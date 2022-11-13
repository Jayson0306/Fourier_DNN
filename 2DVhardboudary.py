"""
@author: LXA
 Date: 2021 年 11 月 11 日
 Modifying on 2022 年 9月 2 日 ~~~~ 2022 年 9月 12 日
@cooperate author:DJX
 Date :2022 年 10月
"""
import os
import sys
import torch
import torch.nn as tn
import numpy as np
import matplotlib
import platform
import shutil
import time
import DNN_base
import DNN_tools
import dataUtilizer2torch
import MS_LaplaceEqs
import MS_BoltzmannEqs
import General_Laplace
import Load_data2Mat
import saveData
import plotData
import DNN_Log_Print
from Load_data2Mat import *
import torchvision
from gen_points import *
from model_distance import *
from scipy.special import erfc


def grad_fun_2D(model_D, XY):
    DNN = model_D(XY).reshape(-1, 1)
    X = torch.reshape(XY[:, 0], shape=[-1, 1])
    grad2DNN = torch.autograd.grad(DNN, XY, grad_outputs=torch.ones_like(X), create_graph=True, retain_graph=True)
    dDNN = grad2DNN[0]
    dDNN2x = torch.reshape(dDNN[:, 0], shape=[-1, 1])
    dDNN2y = torch.reshape(dDNN[:, 1], shape=[-1, 1])
    dDNN2z = torch.reshape(dDNN[:, 2], shape=[-1, 1])
    dDNNxxyz = torch.autograd.grad(dDNN2x, XY, grad_outputs=torch.ones_like(X),
                                   create_graph=True, retain_graph=True)[0]
    dDNNyxyz = torch.autograd.grad(dDNN2y, XY, grad_outputs=torch.ones_like(X),
                                   create_graph=True, retain_graph=True)[0]
    dDNNxx = torch.reshape(dDNNxxyz[:, 0], shape=[-1, 1])
    dDNNyy = torch.reshape(dDNNyxyz[:, 1], shape=[-1, 1])

    return DNN, dDNN2x, dDNNxx, dDNN2y, dDNNyy, dDNN2z




def erfc1(x):
    tmp = x.detach().numpy()
    z = erfc(tmp)
    z = torch.from_numpy(z)
    return z


def AnalyticalSolution(z, t, p, ws, ds, split=1000, inf=-0, Use_Torch=True):
    z = z.reshape(-1, 1)
    t = t.reshape(-1, 1)
    if Use_Torch:
        N = z.shape[0]
        out = torch.zeros_like(z)
    else:
        N = z.size()
        out = np.zeros_like(z)
    for i in range(N):
        out[i] = Eq8(z[i], t[i], p, ws, ds, split, inf, Use_torch=Use_Torch)
    return out


class MscaleDNN(tn.Module):
    def __init__(self, input_dim=4, out_dim=1, hidden_layer=None, Model_name='DNN', name2actIn='relu',
                 name2actHidden='relu', name2actOut='linear', opt2regular_WB='L2', type2numeric='float32',
                 factor2freq=None, sFourier=1.0, use_gpu=False, No2GPU=0, repeat_highFreq=True):
        super(MscaleDNN, self).__init__()
        if 'DNN' == str.upper(Model_name):
            self.DNN = DNN_base.Pure_DenseNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'SCALE_DNN' == str.upper(Model_name) or 'DNN_SCALE' == str.upper(Model_name):
            self.DNN = DNN_base.Dense_ScaleNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                repeat_Highfreq=repeat_highFreq, to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'FOURIER_DNN' == str.upper(Model_name) or 'DNN_FOURIERBASE' == str.upper(Model_name):
            self.DNN = DNN_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                repeat_Highfreq=repeat_highFreq, to_gpu=use_gpu, gpu_no=No2GPU)

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_layer = hidden_layer
        self.Model_name = Model_name
        self.name2actIn = name2actIn
        self.name2actHidden = name2actHidden
        self.name2actOut = name2actOut
        self.factor2freq = factor2freq
        self.sFourier = sFourier
        self.opt2regular_WB = opt2regular_WB

        if type2numeric == 'float32':
            self.float_type = torch.float32
        elif type2numeric == 'float64':
            self.float_type = torch.float64
        elif type2numeric == 'float16':
            self.float_type = torch.float16

        self.use_gpu = use_gpu
        if use_gpu:
            self.opt2device = 'cuda:' + str(No2GPU)
        else:
            self.opt2device = 'cpu'

    def loss2bd(self, XY_bd=None, Ubd_exact=None, if_lambda2Ubd=True, loss_type='l2_loss', scale2lncosh=0.5):
        assert (XY_bd is not None)
        assert (Ubd_exact is not None)

        shape2XY = XY_bd.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 3)
        X_bd = torch.reshape(XY_bd[:, 0], shape=[-1, 1])
        Y_bd = torch.reshape(XY_bd[:, 1], shape=[-1, 1])
        Z_bd = torch.reshape(XY_bd[:, 2], shape=[-1, 1])
        if if_lambda2Ubd:
            Ubd = Ubd_exact(X_bd, Y_bd, Z_bd)
        else:
            Ubd = Ubd_exact

        UNN_bd = self.DNN(XY_bd, scale=self.factor2freq, sFourier=self.sFourier)
        diff_bd = UNN_bd - Ubd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd = torch.mean(torch.square(diff_bd))
        elif str.lower(loss_type) == 'lncosh_loss':
            loss_bd = (1 / scale2lncosh) * torch.mean(torch.log(torch.cosh(scale2lncosh * diff_bd)))
        return loss_bd

    def loss2bd_neumann(self, XY_bd=None, Ubd_exact=None, if_lambda2Ubd=True, loss_type='l2_loss', scale2lncosh=0.5):
        assert (XY_bd is not None)
        assert (Ubd_exact is not None)

        shape2XY = XY_bd.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)
        X = torch.reshape(XY_bd[:, 0], shape=[-1, 1])
        Y = torch.reshape(XY_bd[:, 1], shape=[-1, 1])
        if if_lambda2Ubd:
            U_bd = Ubd_exact(X, Y)
        else:
            U_bd = Ubd_exact

        UNN = self.DNN(XY_bd, scale=self.factor2freq, sFourier=self.sFourier)
        grad2UNN = torch.autograd.grad(UNN, XY_bd, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)
        dUNN = grad2UNN[0]
        dUNN2x = torch.reshape(dUNN[:, 0], shape=[-1, 1])
        diff_bd = dUNN2x - U_bd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd = torch.mean(torch.square(diff_bd))
        elif str.lower(loss_type) == 'lncosh_loss':
            loss_bd = (1 / scale2lncosh) * torch.mean(torch.log(torch.cosh(scale2lncosh * diff_bd)))
        return loss_bd

    def loss_it2ADVDFS(self, XYZ=None, fside=None, if_lambda2fside=True, loss_type='l2_losss',
                       kx=1.4, ky=1.7, vx=1, vy=1):
        assert (XYZ is not None)
        assert (fside is not None)

        shape2XYZ = XYZ.shape
        lenght2XYZ_shape = len(shape2XYZ)
        assert (lenght2XYZ_shape == 2)
        assert (shape2XYZ[-1] == 3)
        X = torch.reshape(XYZ[:, 0], shape=[-1, 1])
        Y = torch.reshape(XYZ[:, 1], shape=[-1, 1])
        Z = torch.reshape(XYZ[:, 2], shape=[-1, 1])

        if if_lambda2fside:
            force_side = fside(X, Y, Z)
        else:
            force_side = fside

        UNN = self.DNN(XYZ, scale=self.factor2freq, sFourier=self.sFourier)
        grad2UNN = torch.autograd.grad(UNN, XYZ, grad_outputs=torch.ones_like(X), create_graph=True, retain_graph=True)
        dUNN = grad2UNN[0]

        dUNN2x = torch.reshape(dUNN[:, 0], shape=[-1, 1])
        dUNN2y = torch.reshape(dUNN[:, 1], shape=[-1, 1])
        dUNN2z = torch.reshape(dUNN[:, 2], shape=[-1, 1])

        dUNNxxyz = torch.autograd.grad(dUNN2x, XYZ, grad_outputs=torch.ones_like(X),
                                       create_graph=True, retain_graph=True)[0]
        dUNNyxyz = torch.autograd.grad(dUNN2y, XYZ, grad_outputs=torch.ones_like(X),
                                       create_graph=True, retain_graph=True)[0]

        dUNNxx = torch.reshape(dUNNxxyz[:, 0], shape=[-1, 1])
        dUNNyy = torch.reshape(dUNNyxyz[:, 1], shape=[-1, 1])
        # du/dt -kx * du/dxx - ky * du/dyy + vx * du/dx + vy * du/dy = f(x,y,t)
        res = -kx * dUNNxx -ky * dUNNyy + vx + dUNN2x + vy * dUNN2y + dUNN2z - force_side

        if str.lower(loss_type) == 'ritz_loss' or str.lower(loss_type) == 'variational_loss':
            dUNN_2Norm = torch.reshape(torch.sum(torch.mul(dUNN, dUNN), dim=-1), shape=[-1, 1])  # 按行求和
            loss_it_ritz = (1.0 / 2) * dUNN_2Norm - torch.mul(torch.reshape(force_side, shape=[-1, 1]), UNN)
            loss_it = torch.mean(loss_it_ritz)
        elif str.lower(loss_type) == 'l2_loss':

            square_loss_it = torch.mul(res, res)
            loss_it = torch.mean(square_loss_it)
        return UNN, loss_it

    def loss_it2Dhardpinn(self, XYZ=None, fside=None, if_lambda2fside=True, loss_type='l2_losss',
                       kx=1.4, ky=1.7, vx=1, vy=1, model_D=None, model_G=None,model_g_is_None =True):
        assert (XYZ is not None)
        assert (fside is not None)

        shape2XYZ = XYZ.shape
        lenght2XYZ_shape = len(shape2XYZ)
        assert (lenght2XYZ_shape == 2)
        assert (shape2XYZ[-1] == 3)
        X = torch.reshape(XYZ[:, 0], shape=[-1, 1])
        Y = torch.reshape(XYZ[:, 1], shape=[-1, 1])
        Z = torch.reshape(XYZ[:, 2], shape=[-1, 1])

        if if_lambda2fside:
            force_side = fside(X, Y, Z)
        else:
            force_side = fside

        UNN = self.DNN(XYZ, scale=self.factor2freq, sFourier=self.sFourier)
        grad2UNN = torch.autograd.grad(UNN, XYZ, grad_outputs=torch.ones_like(X), create_graph=True, retain_graph=True)
        dUNN = grad2UNN[0]

        dUNN2x = torch.reshape(dUNN[:, 0], shape=[-1, 1])
        dUNN2y = torch.reshape(dUNN[:, 1], shape=[-1, 1])
        dUNN2z = torch.reshape(dUNN[:, 2], shape=[-1, 1])

        dUNNxxyz = torch.autograd.grad(dUNN2x, XYZ, grad_outputs=torch.ones_like(X),
                                       create_graph=True, retain_graph=True)[0]
        dUNNyxyz = torch.autograd.grad(dUNN2y, XYZ, grad_outputs=torch.ones_like(X),
                                       create_graph=True, retain_graph=True)[0]

        dUNNxx = torch.reshape(dUNNxxyz[:, 0], shape=[-1, 1])
        dUNNyy = torch.reshape(dUNNyxyz[:, 1], shape=[-1, 1])

        if model_g_is_None:
            DNN, dDNN2x, dDNNxx, dDNN2y, dDNNyy, dDNN2z = grad_fun_2D(model_D, XYZ)
            # du/dt -kx * du/dxx - ky * du/dyy + vx * du/dx + vy * du/dy = f(x,y,t)
            dxx = torch.mul(dDNNxx, UNN) + 2 * torch.mul(dDNN2x, dUNN2x) + torch.mul(dUNNxx, DNN)
            dyy = torch.mul(dDNNyy, UNN) + 2 * torch.mul(dDNN2y, dUNN2y) + torch.mul(dUNNyy, DNN)
            d2x = torch.mul(dDNN2x, UNN) + torch.mul(dUNN2x, DNN)
            d2y = torch.mul(dDNN2y, UNN) + torch.mul(dUNN2y, DNN)
            d2z = torch.mul(dDNN2z, UNN) + torch.mul(dUNN2z, DNN)
            res = -kx * dxx - ky * dyy + vx + d2x + vy * d2y + d2z - force_side
            HPNN = torch.mul(DNN, UNN)
        else:
            DNN, dDNN2x, dDNNxx, dDNN2y, dDNNyy, dDNN2z = grad_fun_2D(model_D, XYZ)
            GNN, dGNN2x, dGNNxx, dGNN2y, dGNNyy, dGNN2z = grad_fun_2D(model_G, XYZ)
            # du/dt -kx * du/dxx - ky * du/dyy + vx * du/dx + vy * du/dy = f(x,y,t)
            dxx = dGNNxx + torch.mul(dDNNxx, UNN) + 2 * torch.mul(dDNN2x, dUNN2x) + torch.mul(dUNNxx, DNN)
            dyy = dGNNyy + torch.mul(dDNNyy, UNN) + 2 * torch.mul(dDNN2y, dUNN2y) + torch.mul(dUNNyy, DNN)
            d2x = dGNN2x + torch.mul(dDNN2x, UNN) + torch.mul(dUNN2x, DNN)
            d2y = dGNN2y + torch.mul(dDNN2y, UNN) + torch.mul(dUNN2y, DNN)
            d2z = dGNN2z + torch.mul(dDNN2z, UNN) + torch.mul(dUNN2z, DNN)
            res = -kx * dxx -ky * dyy + vx + d2x + vy * d2y + d2z - force_side

            HPNN = GNN + torch.mul(DNN, UNN)
        if str.lower(loss_type) == 'ritz_loss' or str.lower(loss_type) == 'variational_loss':
            dUNN_2Norm = torch.reshape(torch.sum(torch.mul(dUNN, dUNN), dim=-1), shape=[-1, 1])  # 按行求和
            loss_it_ritz = (1.0 / 2) * dUNN_2Norm - torch.mul(torch.reshape(force_side, shape=[-1, 1]), UNN)
            loss_it = torch.mean(loss_it_ritz)
        elif str.lower(loss_type) == 'l2_loss':
            square_loss_it = torch.mul(res, res)
            loss_it = torch.mean(square_loss_it)
        return HPNN, loss_it

    def get_regularSum2WB(self):
        sum2WB = self.DNN.get_regular_sum2WB(self.opt2regular_WB)
        return sum2WB

    def evalue_MscaleDNN(self, XY_points=None):
        assert (XY_points is not None)
        shape2XY = XY_points.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 3)

        UNN = self.DNN(XY_points, scale=self.factor2freq, sFourier=self.sFourier)
        return UNN


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']  # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)  # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2train', R['name2act_hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    # DNN_Log_Print.dictionary_out2file(R, log_fileout)

    # 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']

    bd_penalty_init = R['init_boundary_penalty']  # Regularization parameter for boundary conditions
    penalty2WB = R['penalty2weight_biases']  # Regularization parameter for weights and biases
    learning_rate = R['learning_rate']

    input_dim = R['input_dim']
    out_dim = R['output_dim']
    if R['PDE_type'] == '1DV':
        if R['equa_name'] == 'eq1':
            # -laplace u = f
            region_lb = 0.0
            region_rt = 1.0
            a = 1
            beta = 0.1
            vx = -1
            vy = -1
            kx = 1.4
            ky = 1.7
            cx = (vx + np.sqrt(vx ** 2 + 4 * beta * kx)) / (2 * kx)
            cy = (vy + np.sqrt(vy ** 2 + 4 * beta * ky)) / (2 * ky)
            tempx = lambda x: torch.mul(cx, x)
            tempy = lambda y: torch.mul(cy, y)
            f_side = lambda x, y, t: torch.zeros_like(x)
            u_true = lambda x, y, t: a * torch.mul(torch.exp(beta * t), (torch.exp(-tempx(x)) + torch.exp(-tempy(y))))
            # u_left = lambda x, y, t: a * torch.exp(beta * t) * (1 + torch.exp(-tempy(y)))
            # u_right = lambda x, y, t: a * torch.exp(beta * t) * (np.exp(-cx) + torch.exp(-tempy(y)))
            # u_bottom = lambda x, y, t: a * torch.exp(beta * t) * (1 + torch.exp(-tempx(x)))
            # u_top = lambda x, y, t: a * torch.exp(beta * t) * (torch.exp(-tempx(x)) + np.exp(-cy))
            # u_init = lambda x, y, t: a * (torch.exp(-tempx(x)) + torch.exp(-tempy(y)))
            model_distance = lambda x: torch.mul(x[:, 2], torch.mul(torch.mul(x[:, 0], 1-x[:, 0]),
                                                                    torch.mul(x[:, 1], 1-x[:, 1])))
            model_g = lambda x: a * torch.mul(torch.exp(beta * x[:, 2]),
                                              (torch.exp(-tempx(x[:, 0])) + torch.exp(-tempy(x[:, 1]))))
        elif R['equa_name'] == 'eq2':
            # -laplace u = f
            region_lb = 0.0
            region_rt = 1.0
            a = 1
            beta = 0.1
            vx = np.cos(np.pi/8)
            vy = np.sin(np.pi/8)    # 一次项
            kx = 0.02  # 二次项
            ky = 0.02  # 二次项
            f_side = lambda x, y, t: torch.zeros_like(x)
            u_true = lambda x, y, t: torch.mul(t, torch.mul(torch.mul(x, y), torch.mul(1-x,1-y))) * torch.exp(x+y)
            u_left = lambda x, y, t: 0
            u_right = lambda x, y, t: 0
            u_bottom = lambda x, y, t: 0
            u_top = lambda x, y, t: 0
            u_init = lambda x, y, t: 0
        elif R['equa_name'] == 'eq3':
            # -laplace u = f
            region_lb = 0.0
            region_rt = 1.0
            a = 1
            beta = 0.1
            vx = np.cos(np.pi*2)
            vy = np.sin(np.pi*2)    # 一次项
            kx = 0.02  # 二次项
            ky = 0.02  # 二次项
            temp1 = lambda t: torch.div(1, 4*t+1)
            temp2 = lambda x, t: torch.div(torch.square(x-torch.cos(vx*t)), temp1(t)/-kx)
            temp3 = lambda y, t: torch.div(torch.square(y-torch.cos(vy*t)), temp1(t)/-ky)
            f_side = lambda x, y, t: torch.zeros_like(x)
            u_true = lambda x, y, t: torch.mul(temp1(t), torch.exp(temp2(x, t)+temp3(y, t)))
            model_distance = lambda x: torch.mul(x[:, 2], torch.mul(torch.mul(x[:, 0], 1-x[:, 0]),
                                                                    torch.mul(x[:, 1], 1-x[:, 1])))
            model_g = lambda x: torch.mul(temp1(x[:, 2]), torch.exp(temp2(x[:, 0], x[:, 2])+temp3(x[:, 1], x[:, 2])))
            model_g_is_None = False
        elif R['equa_name'] == 'eq4':
            # -laplace u = f
            region_lb = 0.0
            region_rt = 1.0
            vx = 10
            vy = 10  # 一次项
            kx = 1  # 二次项
            ky = 1  # 二次项
            temp1 = lambda x: torch.mul(x, 1-x)
            temp2 = lambda x, y: 12 * (temp1(x) + temp1(y)) - 41 * torch.mul(x, y) + \
                                 21 * (torch.mul(torch.square(x), y) + torch.mul(torch.square(y), x)) - \
                                 torch.mul(torch.square(x), torch.square(y))
            f_side = lambda x, y, t: torch.mul(torch.exp(-t), temp2(x, y))
            u_true = lambda x, y, t: torch.mul(torch.exp(-t), torch.mul(temp1(x), temp1(y)))
            model_distance = lambda x: torch.mul(x[:, 2], torch.mul(torch.mul(x[:, 0], 1-x[:, 0]),
                                                                    torch.mul(x[:, 1], 1-x[:, 1])))
            model_g_is_None =True
            model_g = None

    mscalednn = MscaleDNN(input_dim=R['input_dim'], out_dim=R['output_dim'], hidden_layer=R['hidden_layers'],
                          Model_name=R['model2NN'], name2actIn=R['name2act_in'], name2actHidden=R['name2act_hidden'],
                          name2actOut=R['name2act_out'], opt2regular_WB='L0', repeat_highFreq=R['repeat_highFreq'],
                          type2numeric='float32', factor2freq=R['freq'], use_gpu=R['use_gpu'], No2GPU=R['gpuNo'])
    if True == R['use_gpu']:
        mscalednn = mscalednn.cuda(device='cuda:' + str(R['gpuNo']))

    params2Net = mscalednn.DNN.parameters()

    # 定义优化方法，并给定初始学习率
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr)                     # SGD
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr, momentum=0.8)       # momentum
    # optimizer = torch.optim.RMSprop(params2Net, lr=init_lr, alpha=0.95)     # RMSProp
    optimizer = torch.optim.Adam(params2Net, lr=learning_rate)  # Adam

    # 定义更新学习率的方法
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.995)

    loss_it_all, loss_bd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    test_epoch = []
    loss_init_all = []

    if R['testData_model'] == 'random_generate':
        # # 生成测试数据，用于测试训练后的网络
        # test_bach_size = 1600
        # size2test = 40
        # test_x_torch = dataUtilizer2torch.rand_it(test_bach_size, 1, region_lb, region_rt,
        #                                           to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
        #                                           use_grad2x=True)
        # test_y_torch = dataUtilizer2torch.rand_it(test_bach_size, 1, region_lb, region_rt,
        #                                           to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
        #                                           use_grad2x=True)
        # # test_z_torch = dataUtilizer2torch.rand_it(test_bach_size, 1, region_lb, region_rt,
        # #                                           to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
        # #                                           use_grad2x=True)
        # 生成测试数据，用于测试训练后的网络
        test_bach_size = 100
        size2test = 100
        # test_bach_size = 6400
        # size2test = 80
        # test_bach_size = 10000
        # size2test = 100

        # ---------------------------------------------------------------------------------#
        test_xy_bach_x = np.linspace(0, 1, test_bach_size).reshape(-1, 1)
        test_xy_bach_y = np.linspace(0, 1, test_bach_size).reshape(-1, 1)
        x_repeat = np.repeat(test_xy_bach_x, test_bach_size).reshape(-1, 1)
        t2 = list(test_xy_bach_y)
        t1 = list(test_xy_bach_y)
        for i in range(test_bach_size - 1):
            t2.extend(t1)
        t_repeat = np.array(t2)
        test_xy_bach = np.concatenate([x_repeat, t_repeat], -1)
        test_x_torch = torch.from_numpy(x_repeat)
        test_y_torch = torch.from_numpy(t_repeat)
        test_xy_torch = torch.from_numpy(test_xy_bach)
        test_z_torch = torch.ones_like(test_x_torch) * 0.1
        test_xyz_torch = torch.cat([test_x_torch, test_y_torch, test_z_torch], 1)
        test_xyz_torch = torch.as_tensor(test_xyz_torch, dtype=torch.float)

    if R['testData_model'] == 'load_data':
        size2test = 258
        mat_data_path = 'D:/Matlab/bin'
        x_test, t_test = get_randomData2mat(dim=2, data_path=mat_data_path)
        x_test = x_test.reshape(-1, 1)
        t_test = t_test.reshape(-1, 1)
        test_xy_bach = np.concatenate([x_test, t_test], -1)

    saveData.save_testData_or_solus2mat(test_xy_bach, dataName='testXY', outPath=R['FolderName'])
    # 生成test data的真实值
    Utrue2test = u_true(torch.reshape(test_xyz_torch[:, 0], shape=[-1, 1]),
                        torch.reshape(test_xyz_torch[:, 1], shape=[-1, 1]),
                        torch.reshape(test_xyz_torch[:, 2], shape=[-1, 1]))
    if True == R['use_gpu']:
        test_xyz_torch = test_xyz_torch.cuda(device='cuda:' + str(R['gpuNo']))

    t0 = time.time()
    for i_epoch in range(R['max_epoch'] + 1):
        if R['PDE_type'] == '1DV':
            xyz_it_batch = dataUtilizer2torch.rand_it(batchsize_it, R['input_dim'], region_a=region_lb,
                                                      region_b=region_rt,
                                                      to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'],
                                                      use_grad2x=True)
            xyz_bottom_batch, xyz_top_batch, xyz_left_batch, xyz_right_batch, xyz_front_batch, xyz_behind_batch = \
                dataUtilizer2torch.rand_bd_3D(batchsize_bd, R['input_dim'], region_a=region_lb, region_b=region_rt,
                                              to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'])

        if R['PDE_type'] == '1DV':
            if R['activate_penalty2bd_increase'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 10 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 50 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 100 * bd_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 200 * bd_penalty_init
                else:
                    temp_penalty_bd = 500 * bd_penalty_init
            else:
                temp_penalty_bd = bd_penalty_init

            UNN2train, loss_it = mscalednn.loss_it2Dhardpinn(XYZ=xyz_it_batch, loss_type=R['loss_type'], kx=kx, ky=ky,
                                                            vx=vx, vy=vy, fside=f_side,model_g_is_None=model_g_is_None,
                                                            model_D=model_distance,model_G=model_g)
            PWB = mscalednn.get_regularSum2WB()
            loss_bd = loss_it * 0
            loss = loss_it
            Uexact2train = u_true(torch.reshape(xyz_it_batch[:, 0], shape=[-1, 1]),
                                  torch.reshape(xyz_it_batch[:, 1], shape=[-1, 1]),
                                  torch.reshape(xyz_it_batch[:, 2], shape=[-1, 1]))
            train_mse = torch.mean(torch.square(UNN2train - Uexact2train))
            train_rel = train_mse / torch.mean(torch.square(Uexact2train))

        loss_it_all.append(loss_it.item())
        loss_bd_all.append(loss_bd.item())
        loss_all.append(loss.item())

        optimizer.zero_grad()  # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()  # 对loss关于Ws和Bs求偏导
        optimizer.step()  # 更新参数Ws和Bs
        scheduler.step()

        if R['PDE_type'] == '1DV':
            train_mse = torch.mean(torch.square(UNN2train - Uexact2train))
            train_rel = train_mse / torch.mean(torch.square(Uexact2train))

        train_mse_all.append(train_mse.item())
        train_rel_all.append(train_rel.item())

        if i_epoch % 1000 == 0:
            run_times = time.time() - t0
            tmp_lr = optimizer.param_groups[0]['lr']
            DNN_tools.print_and_log_train_one_epoch_pinn_1(i_epoch, run_times, tmp_lr, loss_it.item(), loss_bd.item(),
                                                           loss.item(), train_mse.item(), train_rel.item(),
                                                           log_out=log_fileout)

            test_epoch.append(i_epoch / 1000)
            if R['PDE_type'] == '1DV':
                if model_g_is_None:
                    TEMP1 = model_distance(test_xyz_torch).reshape(-1, 1)
                    TEMP3 = mscalednn.evalue_MscaleDNN(XY_points=test_xyz_torch)
                    UNN2test = torch.mul(TEMP1, TEMP3)
                else:
                    TEMP1 = model_distance(test_xyz_torch).reshape(-1, 1)
                    TEMP2 = model_g(test_xyz_torch).reshape(-1, 1)
                    TEMP3 = mscalednn.evalue_MscaleDNN(XY_points=test_xyz_torch)
                    UNN2test = TEMP2 + torch.mul(TEMP1, TEMP3)

            else:
                UNN2test = mscalednn.evalue_MscaleDNN(XY_points=test_xyz_torch)

            point_square_error = torch.square(Utrue2test - UNN2test.cpu())
            test_mse = torch.mean(point_square_error)
            test_rel = test_mse / torch.mean(torch.square(Utrue2test))
            test_mse_all.append(test_mse.item())
            test_rel_all.append(test_rel.item())
            DNN_tools.print_and_log_test_one_epoch(test_mse.item(), test_rel.item(), log_out=log_fileout)

    # ------------------- save the training results into mat file and plot them -------------------------
    saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=R['activate_func'],
                                         outPath=R['FolderName'])
    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=R['activate_func'], outPath=R['FolderName'])

    plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=R['activate_func'], outPath=R['FolderName'])
    plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=R['activate_func'], seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    # ----------------------  save testing results to mat files, then plot them --------------------------------
    if True == R['use_gpu']:
        utrue2test_numpy = Utrue2test.cpu().detach().numpy()
        unn2test_numpy = UNN2test.cpu().detach().numpy()
        point_square_error_numpy = point_square_error.cpu().detach().numpy()
    else:
        utrue2test_numpy = Utrue2test.detach().numpy()
        unn2test_numpy = UNN2test.detach().numpy()
        point_square_error_numpy = point_square_error.detach().numpy()

    saveData.save_2testSolus2mat(utrue2test_numpy, unn2test_numpy, actName='utrue',
                                 actName1=R['name2act_hidden'], outPath=R['FolderName'])

    plotData.plot_Hot_solution2test(utrue2test_numpy, size_vec2mat=size2test, actName='Utrue',
                                    seedNo=R['seed'], outPath=R['FolderName'])
    plotData.plot_Hot_solution2test(unn2test_numpy, size_vec2mat=size2test, actName=R['name2act_hidden'],
                                    seedNo=R['seed'], outPath=R['FolderName'])

    saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=R['name2act_hidden'], outPath=R['FolderName'])
    plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=R['activate_func'],
                              seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)

    saveData.save_test_point_wise_err2mat(point_square_error_numpy, actName=R['name2act_hidden'],
                                          outPath=R['FolderName'])

    # plotData.plot_Hot_point_wise_err(point_square_error_numpy, size_vec2mat=size2test,
    #                                  actName=R['activate_func'], seedNo=R['seed'], outPath=R['FolderName'])


if __name__ == "__main__":
    R = {}
    R['gpuNo'] = 0
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

    # 文件保存路径设置
    # store_file = 'Laplace2D'
    # store_file = 'pLaplace2D'
    # store_file = 'Boltzmann2D'
    store_file = 'Soft PINN3D'
    # store_file = 'Hard PINN'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])  # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  复制并保存当前文件 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if platform.system() == 'Windows':
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    # step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    step_stop_flag = 0
    # R['activate_stop'] = int(step_stop_flag)
    R['activate_stop'] = 0
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 2000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of multi-scale problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    R['input_dim'] = 3  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1  # 输出维数

    # if store_file == 'Advec':
    if store_file == 'Soft PINN3D':
        R['PDE_type'] = '1DV'
        # R['equa_name'] = 'eq1'
        R['equa_name'] = 'eq3'

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 训练集的设置(内部和边界)
    if R['PDE_type'] == '1DV':
        R['batch_size2interior'] = 5000  # 内部训练数据的批大小
        R['batch_size2boundary'] = 1000  # 边界训练数据的批大小

    # 装载测试数据模式
    # R['testData_model'] = 'load_data'
    R['testData_model'] = 'random_generate'

    R['loss_type'] = 'L2_loss'  # loss类型:L2 loss
    # R['loss_type'] = 'variational_loss'                      # loss类型:PDE变分
    # R['loss_type'] = 'lncosh_loss'
    R['lambda2lncosh'] = 0.5

    R['optimizer_name'] = 'Adam'  # 优化器
    R['learning_rate'] = 2e-4  # 学习率
    R['learning_rate_decay'] = 5e-5  # 学习率 decay
    R['train_model'] = 'union_training'
    # R['train_model'] = 'group2_training'
    # R['train_model'] = 'group3_training'

    # 正则化权重和偏置的模式
    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'#x
    # R['regular_wb_model'] = 'L2'
    # R['penalty2weight_biases'] = 0.000                    # Regularization parameter for weights
    R['penalty2weight_biases'] = 0.001  # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025                 # Regularization parameter for weights

    # 边界的惩罚处理方式,以及边界的惩罚因子
    R['activate_penalty2bd_increase'] = 1
    # R['activate_penalty2bd_increase'] = 0
    # R['init_boundary_penalty'] = 1000                   # Regularization parameter for boundary conditions
    R['activate_penalty2init_increase'] = 1
    # R['activate_penalty2init_increase'] = 0
    R['init_boundary_penalty'] = 100  # Regularization parameter for boundary conditions
    # 网络的频率范围设置
    # R['freq'] = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
    # R['freq'] = np.random.normal(0, 100, 100)
    R['freq'] = np.concatenate(([1], np.arange(1, 30 - 1)), axis=0)
    R['repeat_highFreq'] = False

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['model2NN'] = 'DNN'
    # R['model2NN'] = 'Scale_DNN'
    # R['model2NN'] = 'Adapt_scale_DNN'
    R['model2NN'] = 'Fourier_DNN'
    # R['model2NN'] = 'Wavelet_DNN'

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model2NN'] == 'Fourier_DNN':
        # R['hidden_layers'] = (125, 250, 200, 100, 50)
        # 125, 200, 200, 100, 100, 80) # 1*125+250*200+200*200+200*100+100*100+100*50+50*1=128205
        R['hidden_layers'] = (50, 80, 60, 60, 40)
    else:
        # R['hidden_layers'] = (100, 80, 80, 60, 40, 40)
        # R['hidden_layers'] = (200, 100, 80, 50, 30)
        R['hidden_layers'] = (
            250, 200, 200, 100, 100, 80)  # 1*250+250*200+200*200+200*100+100*100+100*50+50*1=128330
        # R['hidden_layers'] = (500, 400, 300, 200, 100)
        # R['hidden_layers'] = (500, 400, 300, 300, 200, 100)

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['name2act_in'] = 'relu'
    # R['name2act_in'] = 'tanh'
    R['name2act_in'] = 'Enh_tanh'
    # R['name2act_in'] = 's2relu'

    # R['name2act_hidden'] = 'relu'#x
    # R['name2act_hidden'] = 'tanh'#效果不好
    # R['name2act_hidden']' = leaky_relu'#x
    # R['name2act_hidden'] = 'srelu'#x
    # R['name2act_hidden'] = 's2relu'#x
    # R['name2act_hidden'] = 'sin'
    R['name2act_hidden'] = 'sinAddcos'
    # R['name2act_hidden'] = 'elu'x
    # R['name2act_hidden'] = 'phi'x
    R['activate_func'] = R['name2act_hidden']
    R['name2act_out'] = 'linear'

    if R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 'tanh':
        # R['sfourier'] = 0.5
        R['sfourier'] = 1.0
    elif R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 's2relu':
        R['sfourier'] = 0.5
        # R['sfourier'] = 1.0
    elif R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 'sinAddcos':
        # R['sfourier'] = 0.5
        R['sfourier'] = 1.0
    elif R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 'sin':
        # R['sfourier'] = 0.5
        R['sfourier'] = 1.0
    else:
        R['sfourier'] = 1.0
        # R['sfourier'] = 5.0
        # R['sfourier'] = 0.75

    if R['model2NN'] == 'Wavelet_DNN':
        # R['freq'] = np.concatenate(([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], np.arange(1, 100 - 9)), axis=0)
        # R['freq'] = np.concatenate(([0.25, 0.5, 0.6, 0.7, 0.8, 0.9], np.arange(1, 100 - 6)), axis=0)
        # R['freq'] = np.concatenate(([0.5, 0.6, 0.7, 0.8, 0.9], np.arange(1, 100 - 5)), axis=0)
        # R['freq'] = np.concatenate(([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], np.arange(1, 30-9)), axis=0)
        R['freq'] = np.concatenate(([0.25, 0.5, 0.6, 0.7, 0.8, 0.9], np.arange(1, 100 - 6)), axis=0)
        # R['freq'] = np.arange(1, 100)

    R['use_gpu'] = True

    solve_Multiscale_PDE(R)
