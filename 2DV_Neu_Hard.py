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
        self.mat2XY = torch.tensor([[1, 0, 0],
                                    [0, 1, 0]], dtype=self.float_type, device=self.opt2device)  # 2 行 3 列
        self.mat2U = torch.tensor([[0, 0, 1]], dtype=self.float_type, device=self.opt2device)  # 1 行 3 列
        self.mat2T = torch.tensor([[0, 0, 1]], dtype=self.float_type, device=self.opt2device)  # 1 行 3 列

    def loss_it2_hardbd(self, XY=None, t=None, fside=None, if_lambda2fside=True, loss_type='l2_losss',
                    kx=1.4, ky=1.7, vx=1, vy=1,model_D = None,model_G = None):
        '''

        Args:
            XY: 输入的内部点
            t: 时间
            fside: 原函数
            if_lambda2fside: Bool：原函数为Lambda函数
            loss_type: 损失函数类型
            kx: 二阶导的系数
            ky: 二阶导的系数
            vx: 一阶导的系数
            vy: 一阶导的系数
            model_D: 距离函数
            model_G: 拓展函数

        Returns:
            LNN : 预测值
            loss : 损失

        '''
        assert (XY is not None)
        assert (t is not None)
        assert (fside is not None)

        shape2XY = XY.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        shape2t = t.shape
        lenght2t_shape = len(shape2t)
        assert (lenght2t_shape == 2)
        assert (shape2t[-1] == 1)
        # 输入数据切分 + 与时间T 做结合
        X = torch.reshape(XY[:, 0], shape=[-1, 1])
        Y = torch.reshape(XY[:, 1], shape=[-1, 1])
        # T = torch.reshape(t, shape=[-1, 1])
        XYT = torch.matmul(XY, self.mat2XY) + torch.matmul(t, self.mat2T)

        # 生成对应的源项
        if if_lambda2fside:
            force_side = fside(X, Y, t)
        else:
            force_side = fside
        # 神经网络输出
        UNN = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)
        # dt
        grad2UNN2t = torch.autograd.grad(UNN, t, grad_outputs=torch.ones_like(t), create_graph=True, retain_graph=True)
        dUNN2t = grad2UNN2t[0]
        # dx dy dxx dyy
        grad2UNN = torch.autograd.grad(UNN, XY, grad_outputs=torch.ones_like(X), create_graph=True, retain_graph=True)
        dUNN = grad2UNN[0]
        dUNN2x = torch.reshape(dUNN[:, 0], shape=[-1, 1])
        dUNN2y = torch.reshape(dUNN[:, 1], shape=[-1, 1])

        dUNNxxy = torch.autograd.grad(dUNN2x, XY, grad_outputs=torch.ones_like(X),
                                      create_graph=True, retain_graph=True)[0]
        dUNNyxy = torch.autograd.grad(dUNN2y, XY, grad_outputs=torch.ones_like(X),
                                      create_graph=True, retain_graph=True)[0]

        dUNNxx = torch.reshape(dUNNxxy[:, 0], shape=[-1, 1])
        dUNNyy = torch.reshape(dUNNyxy[:, 1], shape=[-1, 1])

        # 神经网络输出
        DNN = model_D(X, Y, t)
        # dt
        grad2DNN2t = torch.autograd.grad(DNN, t, grad_outputs=torch.ones_like(t), create_graph=True, retain_graph=True)
        dDNN2t = grad2DNN2t[0]
        # dx dy dxx dyy
        grad2DNN = torch.autograd.grad(DNN, XY, grad_outputs=torch.ones_like(X), create_graph=True, retain_graph=True)
        dDNN = grad2DNN[0]
        dDNN2x = torch.reshape(dDNN[:, 0], shape=[-1, 1])
        dDNN2y = torch.reshape(dDNN[:, 1], shape=[-1, 1])

        dDNNxxy = torch.autograd.grad(dDNN2x, XY, grad_outputs=torch.ones_like(X),
                                      create_graph=True, retain_graph=True)[0]
        dDNNyxy = torch.autograd.grad(dDNN2y, XY, grad_outputs=torch.ones_like(X),
                                      create_graph=True, retain_graph=True)[0]

        dDNNxx = torch.reshape(dDNNxxy[:, 0], shape=[-1, 1])
        dDNNyy = torch.reshape(dDNNyxy[:, 1], shape=[-1, 1])

        GNN = model_G(X, Y, t)
        # dt
        grad2GNN2t = torch.autograd.grad(GNN, t, grad_outputs=torch.ones_like(t), create_graph=True, retain_graph=True, allow_unused=True)
        dGNN2t = grad2GNN2t[0]
        # dx dy dxx dyy
        grad2GNN = torch.autograd.grad(GNN, XY, grad_outputs=torch.ones_like(X), create_graph=True, retain_graph=True)
        dGNN = grad2GNN[0]
        dGNN2x = torch.reshape(dGNN[:, 0], shape=[-1, 1])
        dGNN2y = torch.reshape(dGNN[:, 1], shape=[-1, 1])

        dGNNxxy = torch.autograd.grad(dGNN2x, XY, grad_outputs=torch.ones_like(X),
                                      create_graph=True, retain_graph=True)[0]
        dGNNyxy = torch.autograd.grad(dGNN2y, XY, grad_outputs=torch.ones_like(X),
                                      create_graph=True, retain_graph=True)[0]

        dGNNxx = torch.reshape(dGNNxxy[:, 0], shape=[-1, 1])
        dGNNyy = torch.reshape(dGNNyxy[:, 1], shape=[-1, 1])

        # du/dt -kx * du/dxx - ky * du/dyy + vx * du/dx + vy * du/dy = f(x,y,t)
        DT = torch.mul(DNN, dUNN2t) + torch.mul(dDNN2t, UNN)
        DX = dGNN2x + torch.mul(DNN, dUNN2x) + torch.mul(dDNN2x, UNN)
        DY = dGNN2y + torch.mul(DNN, dUNN2y) + torch.mul(dDNN2y, UNN)
        DXX = dGNNxx + torch.mul(DNN, dUNNxx) + 2 * torch.mul(dDNN2x, dUNN2x) + torch.mul(dDNNxx, UNN)
        DYY = dGNNyy + torch.mul(DNN, dUNNyy) + 2 * torch.mul(dDNN2y, dUNN2y) + torch.mul(dDNNyy, UNN)
        res = DT + vx * DX + vy * DY - kx * DXX - ky * DYY- force_side

        LNN = GNN + torch.mul(DNN, UNN)
        if str.lower(loss_type) == 'ritz_loss' or str.lower(loss_type) == 'variational_loss':
            dUNN_2Norm = torch.reshape(torch.sum(torch.mul(dUNN, dUNN), dim=-1), shape=[-1, 1])  # 按行求和
            loss_it_ritz = (1.0 / 2) * dUNN_2Norm - torch.mul(torch.reshape(force_side, shape=[-1, 1]), UNN)
            loss_it = torch.mean(loss_it_ritz)
        elif str.lower(loss_type) == 'l2_loss':

            square_loss_it = torch.mul(res, res)
            loss_it = torch.mean(square_loss_it)
        return LNN, loss_it

    def loss2bd_neumann_x(self, XY_bd=None, T_bd=None, Ubd_exact=None, if_lambda2Ubd=True, loss_type='l2_loss',
                          model_D=None, model_G=None):
        assert (XY_bd is not None)
        assert (Ubd_exact is not None)

        shape2XY = XY_bd.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)
        X = torch.reshape(XY_bd[:, 0], shape=[-1, 1])
        Y = torch.reshape(XY_bd[:, 1], shape=[-1, 1])
        T_bd = torch.reshape(T_bd, shape=[-1, 1])
        if if_lambda2Ubd:
            U_bd = Ubd_exact(X, Y, T_bd)
        else:
            U_bd = Ubd_exact

        XYT = torch.matmul(XY_bd, self.mat2XY) + torch.matmul(T_bd, self.mat2T)

        # 神经网络输出
        UNN = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)
        # dx dy dxx dyy
        grad2UNN = torch.autograd.grad(UNN, XY_bd, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)
        dUNN = grad2UNN[0]
        dUNN2x = torch.reshape(dUNN[:, 0], shape=[-1, 1])
        # 神经网络输出
        GNN = model_G(X, Y, T_bd)
        # dx dy dxx dyy
        grad2GNN = torch.autograd.grad(GNN, XY_bd, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)
        dGNN = grad2GNN[0]
        dGNN2x = torch.reshape(dGNN[:, 0], shape=[-1, 1])

        DNN = model_D(X, Y, T_bd)
        grad2DNN = torch.autograd.grad(DNN, XY_bd, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)
        dDNN = grad2DNN[0]
        dDNN2x = torch.reshape(dDNN[:, 0], shape=[-1, 1])

        # du/dt -kx * du/dxx - ky * du/dyy + vx * du/dx + vy * du/dy = f(x,y,t)
        DX = dGNN2x + torch.mul(DNN, dUNN2x) + torch.mul(dDNN2x, UNN)

        res = DX - U_bd

        square_loss_it = torch.mul(res, res)
        loss_it = torch.mean(square_loss_it)
        return loss_it

    def loss2bd_neumann_y(self, XY_bd=None, T_bd=None, Ubd_exact=None, if_lambda2Ubd=True, loss_type='l2_loss',
                          model_D=None, model_G=None):
        assert (XY_bd is not None)
        assert (Ubd_exact is not None)

        shape2XY = XY_bd.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)
        X = torch.reshape(XY_bd[:, 0], shape=[-1, 1])
        Y = torch.reshape(XY_bd[:, 1], shape=[-1, 1])
        T_bd = torch.reshape(T_bd, shape=[-1, 1])
        if if_lambda2Ubd:
            U_bd = Ubd_exact(X, Y, T_bd)
        else:
            U_bd = Ubd_exact

        XYT = torch.matmul(XY_bd, self.mat2XY) + torch.matmul(T_bd, self.mat2T)

        # 神经网络输出
        UNN = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)
        # dx dy dxx dyy
        grad2UNN = torch.autograd.grad(UNN, XY_bd, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)
        dUNN = grad2UNN[0]
        dUNN2y = torch.reshape(dUNN[:, 1], shape=[-1, 1])
        # 神经网络输出
        GNN = model_G(X, Y, T_bd)
        # dx dy dxx dyy
        grad2GNN = torch.autograd.grad(GNN, XY_bd, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)
        dGNN = grad2GNN[0]
        dGNN2y = torch.reshape(dGNN[:, 1], shape=[-1, 1])

        DNN = model_D(X, Y, T_bd)
        grad2DNN = torch.autograd.grad(DNN, XY_bd, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)
        dDNN = grad2DNN[0]
        dDNN2y = torch.reshape(dDNN[:, 1], shape=[-1, 1])
        DX = dGNN2y + torch.mul(DNN, dUNN2y) + torch.mul(dDNN2y, UNN)

        res = DX - U_bd

        square_loss_it = torch.mul(res, res)
        loss_it = torch.mean(square_loss_it)
        return loss_it

    def get_regularSum2WB(self):
        sum2WB = self.DNN.get_regular_sum2WB(self.opt2regular_WB)
        return sum2WB

    def evalue_MscaleDNN(self, XY_points=None, t_points=None):
        assert (XY_points is not None)
        shape2XY = XY_points.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)
        XYT = torch.matmul(XY_points, self.mat2XY) + torch.matmul(t_points, self.mat2T)
        UNN = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)
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
    batchsize_init = R['batch_size2init']

    bd_penalty_init = R['init_boundary_penalty']  # Regularization parameter for boundary conditions
    init_penalty_init = R['init_init_penalty']
    penalty2WB = R['penalty2weight_biases']  # Regularization parameter for weights and biases
    learning_rate = R['learning_rate']

    input_dim = R['input_dim']
    out_dim = R['output_dim']
    if R['PDE_type'] == '1DV':
        # 二维非稳态对流扩散方程的高阶紧致差分格式_魏剑英
        # Ut +vx * Ux +vy *Uy - kx * Uxx -ky * Uyy = f
        region_l = 0.0
        region_r = 1.0
        region_b = 0.0
        region_t = 1.0
        init_time = 0
        end_time = 5
        vx = 10
        vy = 10  # 一次项
        kx = 1  # 二次项
        ky = 1  # 二次项
        temp1 = lambda x: torch.mul(x, 1 - x)
        temp2 = lambda x, y: 12 * (temp1(x) + temp1(y)) - 41 * torch.mul(x, y) + \
                             21 * (torch.mul(torch.square(x), y) + torch.mul(torch.square(y), x)) - \
                             torch.mul(torch.square(x), torch.square(y))
        f_side = lambda x, y, t: torch.mul(torch.exp(-t), temp2(x, y))
        u_true = lambda x, y, t: torch.mul(torch.exp(-t), torch.mul(temp1(x), temp1(y)))
        u_left = lambda x, y, t: torch.mul(torch.exp(-t), temp1(y))
        u_right = lambda x, y, t: torch.mul(torch.exp(-t), -1.0 * temp1(y))
        u_bottom = lambda x, y, t: torch.mul(torch.exp(-t), temp1(x))
        u_top = lambda x, y, t: torch.mul(torch.exp(-t), -1.0 * temp1(x))
        u_init = lambda x, y: torch.mul(temp1(x), temp1(y))
        model_D  = lambda x, y, t: torch.mul(torch.mul(temp1(x), temp1(y)), t)
        model_G = lambda x, y, t: u_init(x, y)


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
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.995)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.99)

    loss_it_all, loss_bd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    test_epoch = []
    loss_init_all = []

    if R['testData_model'] == 'random_generate':
        # # 生成测试数据，用于测试训练后的网络
        # test_bach_size = 1600
        # size2test = 40
        # test_x_torch = dataUtilizer2torch.rand_it(test_bach_size, 1, region_l, region_r,
        #                                           to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
        #                                           use_grad2x=True)
        # test_y_torch = dataUtilizer2torch.rand_it(test_bach_size, 1, region_b, region_t,
        #                                           to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
        #                                           use_grad2x=True)
        # test_z_torch = dataUtilizer2torch.rand_it(test_bach_size, 1, init_time, end_time,
        #                                           to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
        #                                           use_grad2x=True)
        #
        # test_xy_torch = torch.cat([test_x_torch, test_y_torch], 1)
        # test_xy_bach = test_xy_torch.detach().numpy()
        # test_xyz_torch = torch.cat([test_x_torch, test_y_torch, test_z_torch], 1)
        # test_xyz_torch = torch.as_tensor(test_xyz_torch, dtype=torch.float)

        # 线性差值
        # 生成测试数据，用于测试训练后的网络
        test_bach_size = 100
        size2test = 100
        # test_bach_size = 6400
        # size2test = 80
        # test_bach_size = 10000
        # size2test = 100

        # ---------------------------------------------------------------------------------#
        test_xy_bach_x = np.linspace(0, region_r, test_bach_size).reshape(-1, 1)
        test_xy_bach_y = np.linspace(0, region_t, test_bach_size).reshape(-1, 1)
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
        test_z_torch = torch.ones_like(test_x_torch) * 1
        # test_xyz_torch = torch.cat([test_x_torch, test_y_torch, test_z_torch], 1)
        # test_xyz_torch = torch.as_tensor(test_xyz_torch, dtype=torch.float)
        test_xy_torch = torch.as_tensor(test_xy_torch, dtype=torch.float)
        test_z_torch = torch.as_tensor(test_z_torch, dtype=torch.float)
    if R['testData_model'] == 'load_data':
        size2test = 258
        mat_data_path = 'D:/Matlab/bin'
        x_test, t_test = get_randomData2mat(dim=2, data_path=mat_data_path)
        x_test = x_test.reshape(-1, 1)
        t_test = t_test.reshape(-1, 1)
        test_xy_bach = np.concatenate([x_test, t_test], -1)

    saveData.save_testData_or_solus2mat(test_xy_bach, dataName='testXY', outPath=R['FolderName'])
    # 生成test data的真实值
    Utrue2test = u_true(torch.reshape(test_xy_torch[:, 0], shape=[-1, 1]),
                        torch.reshape(test_xy_torch[:, 1], shape=[-1, 1]),
                        torch.reshape(test_z_torch, shape=[-1, 1]))
    if True == R['use_gpu']:
        test_xy_torch = test_xy_torch.cuda(device='cuda:' + str(R['gpuNo']))
        test_z_torch = test_z_torch.cuda(device='cuda:' + str(R['gpuNo']))

    t0 = time.time()
    t_init_batch = np.ones(shape=[batchsize_init, 1], dtype=np.float32) * init_time
    t_init_batch = torch.from_numpy(t_init_batch)
    if True == R['use_gpu']:
        t_init_bd = t_init_batch.cuda(device='cuda:' + str(R['gpuNo']))


    for i_epoch in range(R['max_epoch'] + 1):
        if R['PDE_type'] == '1DV':
            # xy_it_batch = dataUtilizer2torch.rand_bd_2D1(batchsize_it, R['input_dim']-1,
            #                                              region_a=region_l, region_b=region_r, init_l=region_b, init_r=region_t,
            #                                              to_torch=True, to_float=True, to_cuda=True,
            #                                              gpu_no=0, use_grad=True)
            #生成训练的时候的内部点
            xy_it_batch = dataUtilizer2torch.rand2D_2(batchsize_it, R['input_dim'] - 1,
                                                      region_l=region_l, region_r=region_r, region_b=region_b,
                                                      region_t=region_t,
                                                      to_torch=True, to_float=True, to_cuda=True,
                                                      gpu_no=0, use_grad=True)
            t_it_batch = dataUtilizer2torch.rand_it(batchsize_it, 1, region_a=init_time, region_b=end_time,
                                                    to_torch=True, to_float=True, to_cuda=True,
                                                    gpu_no=0, use_grad2x=True)
            xy_left_bd, xy_right_bd, xy_bottom_bd, xy_top_bd = dataUtilizer2torch.rand_bd_2D1(batch_size=batchsize_bd,
                                                                                              variable_dim=R['input_dim'] - 1,
                                                                                              region_a=region_l,
                                                                                              region_b=region_r,
                                                                                              init_l=region_b,
                                                                                              init_r=region_t,
                                                                                              to_torch=True,
                                                                                              to_float=True,
                                                                                              to_cuda=True,
                                                                                              use_grad=True, gpu_no=0)
            t_bd_batch = dataUtilizer2torch.rand_it(batchsize_bd, 1, region_a=init_time, region_b=end_time,
                                                    to_torch=True, to_float=True, to_cuda=True, gpu_no=0,
                                                    use_grad2x=False)

            #计算各个损失来源
            #PDE损失
            UNN2train, loss_it = mscalednn.loss_it2_hardbd(XY=xy_it_batch, t=t_it_batch, loss_type=R['loss_type'], kx=kx,
                                                       ky=ky, vx=vx, vy=vy, fside=f_side,model_D=model_D,model_G=model_G)
            # 边界点损失
            loss_bd2left = mscalednn.loss2bd_neumann_x(XY_bd=xy_left_bd, T_bd=t_bd_batch, Ubd_exact=u_left,
                                                       loss_type=R['loss_type'],model_D=model_D,model_G=model_G)

            loss_bd2right = mscalednn.loss2bd_neumann_x(XY_bd=xy_right_bd, T_bd=t_bd_batch, Ubd_exact=u_right,
                                                        loss_type=R['loss_type'],model_D=model_D,model_G=model_G)

            loss_bd2bottom = mscalednn.loss2bd_neumann_y(XY_bd=xy_bottom_bd, T_bd=t_bd_batch, Ubd_exact=u_bottom,
                                                         loss_type=R['loss_type'],model_D=model_D,model_G=model_G)

            loss_bd2top = mscalednn.loss2bd_neumann_y(XY_bd=xy_top_bd, T_bd=t_bd_batch, Ubd_exact=u_top,
                                                      loss_type=R['loss_type'],model_D=model_D,model_G=model_G)
            #边界损失点之和
            loss_bd = loss_bd2left + loss_bd2right +loss_bd2bottom +loss_bd2top
            PWB = mscalednn.get_regularSum2WB()
            # loss = loss_it + loss_bd + loss_init
            loss = loss_it + loss_bd
            Uexact2train = u_true(torch.reshape(xy_it_batch[:, 0], shape=[-1, 1]),
                                  torch.reshape(xy_it_batch[:, 1], shape=[-1, 1]),
                                  torch.reshape(t_it_batch, shape=[-1, 1]))
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
            DNN_tools.print_and_log_train_one_epoch_pinn_1(i_epoch, run_times, tmp_lr, loss_it.item(), loss_bd.item() ,
                                                           loss.item(), train_mse.item(), train_rel.item(),
                                                           log_out=log_fileout)

            test_epoch.append(i_epoch / 1000)

            t1 = model_G(x=test_xy_torch[:, 0].reshape(-1, 1), y=test_xy_torch[:, 1].reshape(-1, 1), t=test_z_torch)
            t2 = model_D(x=test_xy_torch[:, 0].reshape(-1, 1), y=test_xy_torch[:, 1].reshape(-1, 1), t=test_z_torch)
            t3 = mscalednn.evalue_MscaleDNN(XY_points=test_xy_torch, t_points=test_z_torch)
            UNN2test = t1 + torch.mul(t2, t3)
            point_square_error = torch.square(Utrue2test - UNN2test.cpu())
            test_mse = torch.mean(point_square_error)
            test_rel = test_mse / torch.mean(torch.square(Utrue2test))
            test_mse_all.append(test_mse.item())
            test_rel_all.append(test_rel.item())
            DNN_tools.print_and_log_test_one_epoch(test_mse.item(), test_rel.item(), log_out=log_fileout)

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
                                 actName1='Usin', outPath=R['FolderName'])



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
    store_file = 'Hard PINN3D'
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
    R['max_epoch'] = 100000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of multi-scale problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    R['input_dim'] = 3  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1  # 输出维数

    # if store_file == 'Advec':
    if store_file == 'Hard PINN3D':
        R['PDE_type'] = '1DV'
        # R['equa_name'] = 'eq1'
        R['equa_name'] = 'eq4'

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 训练集的设置(内部和边界)
    if R['PDE_type'] == '1DV':
        R['batch_size2interior'] = 8000  # 内部训练数据的批大小
        R['batch_size2boundary'] = 2000  # 边界训练数据的批大小
        R['batch_size2init'] = 2000

    # 装载测试数据模式
    # R['testData_model'] = 'load_data'
    R['testData_model'] = 'random_generate'

    R['loss_type'] = 'L2_loss'  # loss类型:L2 loss
    # R['loss_type'] = 'variational_loss'                      # loss类型:PDE变分
    # R['loss_type'] = 'lncosh_loss'
    R['lambda2lncosh'] = 0.5

    R['optimizer_name'] = 'Adam'  # 优化器
    # R['learning_rate'] = 2e-4  # 学习率
    R['learning_rate'] = 0.001  # 学习率
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
    # R['activate_penalty2bd_increase'] = 1
    R['activate_penalty2bd_increase'] = 0
    # R['init_boundary_penalty'] = 1000                   # Regularization parameter for boundary conditions
    # R['activate_penalty2init_increase'] = 1
    R['activate_penalty2init_increase'] = 0
    R['init_boundary_penalty'] = 100  # Regularization parameter for boundary conditions
    R['init_init_penalty'] = 100  # Regularization parameter for boundary conditions

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['model2NN'] = 'DNN'
    # R['model2NN'] = 'Scale_DNN'
    # R['model2NN'] = 'Adapt_scale_DNN'
    R['model2NN'] = 'Fourier_DNN'
    # R['model2NN'] = 'Wavelet_DNN'

    # 网络的频率范围设置
    R['freq'] = np.concatenate(([1], np.arange(1, 40 - 1)), axis=0)
    R['repeat_highFreq'] = True

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['model2NN'] = 'DNN'
    # R['model2NN'] = 'Scale_DNN'
    R['model2NN'] = 'Fourier_DNN'

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model2NN'] == 'Fourier_DNN':
        # R['hidden_layers'] = (100, 150, 80, 80, 50)
        R['hidden_layers'] = (125, 200, 100, 100, 80)  # 1*125+250*200+200*200+200*100+100*100+100*50+50*1=128205
        # R['hidden_layers'] = (50, 10, 10, 10)
        # R['hidden_layers'] = (50, 80, 60, 60, 40)
    else:
        R['hidden_layers'] = (100, 150, 80, 80, 50)
        # R['hidden_layers'] = (200, 100, 80, 50, 30)
        # R['hidden_layers'] = (250, 200, 100, 100, 80)  # 1*250+250*200+200*200+200*100+100*100+100*50+50*1=128330
        # R['hidden_layers'] = (500, 400, 300, 200, 100)
        # R['hidden_layers'] = (500, 400, 300, 300, 200, 100)

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['name2act_in'] = 'relu'
    R['name2act_in'] = 'sin'
    # R['name2act_in'] = 'tanh'
    # R['name2act_in'] = 's2relu'
    # R['name2act_in'] = 'Enh_tanh'

    # R['name2act_hidden'] = 'relu'
    # R['name2act_hidden'] = 'tanh'
    # R['name2act_hidden']' = leaky_relu'
    # R['name2act_hidden'] = 'srelu'
    # R['name2act_hidden'] = 's2relu'
    # R['name2act_hidden'] = 'sin'
    # R['name2act_hidden'] = 'sinAddcos'
    R['name2act_hidden'] = 'Enh_tanh'
    # R['name2act_hidden'] = 'elu'
    # R['name2act_hidden'] = 'phi'

    R['activate_func'] = R['name2act_in']
    R['name2act_out'] = 'linear'

    R['sfourier'] = 1.0

    R['use_gpu'] = True

    solve_Multiscale_PDE(R)
