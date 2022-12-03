"""
@author: LXA
 Date: 2021 年 11 月 11 日
 Modifying on 2022 年 9月 2 日 ~~~~ 2022 年 9月 12 日
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

import saveData
import plotData
import DNN_Log_Print
from Load_data2Mat import *
from scipy.special import erfc


class MscaleDNN(tn.Module):
    def __init__(self, input_dim=2, out_dim=1, hidden_layer=None, Model_name='DNN', name2actIn='relu',
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

        self.use_gpu = use_gpu
        if use_gpu:
            self.opt2device = 'cuda:' + str(No2GPU)
        else:
            self.opt2device = 'cpu'
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
        self.mat2XYZ = torch.tensor([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0]], dtype=self.float_type, device=self.opt2device)  # 3 行 4 列
        self.mat2U = torch.tensor([[0, 0, 0, 1]], dtype=self.float_type, device=self.opt2device)  # 1 行 3 列
        self.mat2T = torch.tensor([[0, 0, 0, 1]], dtype=self.float_type, device=self.opt2device)  # 1 行 3 列

    def loss_3DV(self, XYZ=None, T=None, loss_type='l2_loss', if_lambda2fside=True, fside = None,
                 p=1, q=1, r=1, a=1):
        '''

        Args:
            XYZ: 输入的空间上的随机点
            T: 输入的时间上的随机点
            loss_type: 损失类型
            if_lambda2fside : 边界是否为lambda 函数
            p: Ux 的系数
            q: Uy 的系数
            r: Uz 的系数
            a: 二阶导的系数

        Returns:

        '''

        # 判断 XYZ T 非空
        assert (XYZ is not None)
        assert (fside is not None)

        shape2XYZ = XYZ.shape
        lenght2XYZ_shape = len(shape2XYZ)
        shape2T = T.shape
        lenght2T_shape = len(shape2T)

        # 判断 XYZ 是三列的变量  T 为 1 列的向量
        assert (lenght2XYZ_shape == 2)
        assert (shape2XYZ[-1] == 3)
        assert (lenght2T_shape == 2)
        assert (shape2T[-1] == 1)

        # 将XYZ 摘出来
        X = torch.reshape(XYZ[:, 0], shape=[-1, 1])
        Y = torch.reshape(XYZ[:, 1], shape=[-1, 1])
        Z = torch.reshape(XYZ[:, 2], shape=[-1, 1])

        # 生成源项
        if if_lambda2fside:
            force_side = fside(XYZ, T)
        else:
            force_side = fside

        XYZT = torch.matmul(XYZ, self.mat2XYZ) + torch.matmul(T, self.mat2T)
        # XYZT = torch.cat([XYZ, T], 1)
        # 求UNN XYZ的偏导项
        UNN = self.DNN(XYZT, scale=self.factor2freq, sFourier=self.sFourier)
        grad2UNN = torch.autograd.grad(UNN, XYZ, grad_outputs=torch.ones_like(X), create_graph=True, retain_graph=True)
        dUNN = grad2UNN[0]

        dUNN2x = torch.reshape(dUNN[:, 0], shape=[-1, 1])
        dUNN2y = torch.reshape(dUNN[:, 1], shape=[-1, 1])
        dUNN2z = torch.reshape(dUNN[:, 2], shape=[-1, 1])

        dUNNxxyz = torch.autograd.grad(dUNN2x, XYZ, grad_outputs=torch.ones_like(X),
                               create_graph=True, retain_graph=True)[0]
        dUNNyxyz = torch.autograd.grad(dUNN2y, XYZ, grad_outputs=torch.ones_like(X),
                               create_graph=True, retain_graph=True)[0]
        dUNNzxyz = torch.autograd.grad(dUNN2z, XYZ, grad_outputs=torch.ones_like(X),
                               create_graph=True, retain_graph=True)[0]

        dUNNxx = torch.reshape(dUNNxxyz[:, 0], shape=[-1, 1])
        dUNNyy = torch.reshape(dUNNyxyz[:, 1], shape=[-1, 1])
        dUNNzz = torch.reshape(dUNNzxyz[:, 2], shape=[-1, 1])

        # 求T的偏导项
        grad2UNN = torch.autograd.grad(UNN, T, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True)
        dUNN2t = grad2UNN[0]


        # du/dt = kx * du/dxx + ky * du/dyy + vx * du/dx + vy * du/dy
        res = dUNN2t + p * dUNN2x + q * dUNN2y + r * dUNN2z - a * (dUNNxx + dUNNyy + dUNNzz)

        if str.lower(loss_type) == 'ritz_loss' or str.lower(loss_type) == 'variational_loss':
            dUNN_2Norm = torch.reshape(torch.sum(torch.mul(dUNN, dUNN), dim=-1), shape=[-1, 1])  # 按行求和
            loss_it_ritz = (1.0/2)*dUNN_2Norm-torch.mul(torch.reshape(force_side, shape=[-1, 1]), UNN)
            loss_it = torch.mean(loss_it_ritz)
        elif str.lower(loss_type) == 'l2_loss':

            square_loss_it = torch.mul(res, res)
            loss_it = torch.mean(square_loss_it)
        return UNN, loss_it

    def loss2bd_neumann(self, X_bd=None, T_bd=None, Ubd_exact=None, if_lambda2Ubd=True,
                        loss_type='l2_loss', scale2lncosh=0.5):
        assert (X_bd is not None)
        assert (Ubd_exact is not None)

        shape2XY = X_bd.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 1)

        XT_bd = torch.mul(X_bd, self.mat2X) + torch.mul(T_bd, self.mat2T)

        if if_lambda2Ubd:
            U_bd = Ubd_exact(X_bd, T_bd)
        else:
            U_bd = Ubd_exact

        # 用神经网络求 dUNN2x 以及 UNN
        UNN = self.DNN(XT_bd, scale=self.factor2freq, sFourier=self.sFourier)
        grad2UNN = torch.autograd.grad(UNN, X_bd, grad_outputs=torch.ones_like(X_bd), create_graph=True,
                                       retain_graph=True, allow_unused=True)
        dUNN2x = grad2UNN[0]

        diff_bd = dUNN2x - U_bd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd = torch.mean(torch.square(diff_bd))
        elif str.lower(loss_type) == 'lncosh_loss':
            loss_bd = (1 / scale2lncosh) * torch.mean(torch.log(torch.cosh(scale2lncosh * diff_bd)))
        return loss_bd

    def loss2bd_dirichlet_3D(self, XYZ_bd=None, T_bd=None, Ubd_exact=None, if_lambda2Ubd=True,
                          loss_type='l2_loss', scale2lncosh=0.5):
        # 判断 XYZ T 非空
        assert (XYZ_bd is not None)
        assert (Ubd_exact is not None)

        shape2XYZ = XYZ_bd.shape
        lenght2XYZ_shape = len(shape2XYZ)
        shape2T = T_bd.shape
        lenght2T_shape = len(shape2T)

        # 判断 XYZ 是三列的变量  T 为 1 列的向量
        assert (lenght2XYZ_shape == 2)
        assert (shape2XYZ[-1] == 3)
        assert (lenght2T_shape == 2)
        assert (shape2T[-1] == 1)

        # 将XYZ 摘出来
        X = torch.reshape(XYZ_bd[:, 0], shape=[-1, 1])
        Y = torch.reshape(XYZ_bd[:, 1], shape=[-1, 1])
        Z = torch.reshape(XYZ_bd[:, 2], shape=[-1, 1])

        # 生成源项
        if if_lambda2Ubd:
            U_bd = Ubd_exact(XYZ_bd, T_bd)
        else:
            U_bd = Ubd_exact

        XYZT = torch.matmul(XYZ_bd, self.mat2XYZ) + torch.mul(T_bd, self.mat2T)

        UNN = self.DNN(XYZT, scale=self.factor2freq, sFourier=self.sFourier)

        diff_bd = UNN - U_bd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd = torch.mean(torch.square(diff_bd))
        elif str.lower(loss_type) == 'lncosh_loss':
            loss_bd = (1 / scale2lncosh) * torch.mean(torch.log(torch.cosh(scale2lncosh * diff_bd)))
        return loss_bd

    def loss2init(self, XYZ=None, T_init=None, Uinit_exact=None, if_lambda2Uinit=True,
                          loss_type='l2_loss', scale2lncosh=0.5):
        # 判断 XYZ T 非空
        assert (XYZ is not None)
        assert (Uinit_exact is not None)

        shape2XYZ = XYZ.shape
        lenght2XYZ_shape = len(shape2XYZ)
        shape2T = T_init.shape
        lenght2T_shape = len(shape2T)

        # 判断 XYZ 是三列的变量  T 为 1 列的向量
        assert (lenght2XYZ_shape == 2)
        assert (shape2XYZ[-1] == 3)
        assert (lenght2T_shape == 2)
        assert (shape2T[-1] == 1)

        # 将XYZ 摘出来
        X = torch.reshape(XYZ[:, 0], shape=[-1, 1])
        Y = torch.reshape(XYZ[:, 1], shape=[-1, 1])
        Z = torch.reshape(XYZ[:, 2], shape=[-1, 1])

        if if_lambda2Uinit:
            U_init = Uinit_exact(XYZ, T_init)
        else:
            U_init = Uinit_exact

        XYZT = torch.matmul(XYZ, self.mat2XYZ) + torch.matmul(T_init, self.mat2T)
        UNN = self.DNN(XYZT, scale=self.factor2freq, sFourier=self.sFourier)

        diff_bd = UNN - U_init
        if str.lower(loss_type) == 'l2_loss':
            loss_bd = torch.mean(torch.square(diff_bd))
        elif str.lower(loss_type) == 'lncosh_loss':
            loss_bd = (1 / scale2lncosh) * torch.mean(torch.log(torch.cosh(scale2lncosh * diff_bd)))
        return loss_bd


    def get_regularSum2WB(self):
        sum2WB = self.DNN.get_regular_sum2WB(self.opt2regular_WB)
        return sum2WB

    def evalue_MscaleDNN(self, XY_points=None):
        assert (XY_points is not None)
        shape2XY = XY_points.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 4)

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
    batchsize_init = R['batch_size2init']

    bd_penalty_init = R['init_boundary_penalty']  # Regularization parameter for boundary conditions
    init_penalty_init = R['init_penalty2init']
    penalty2WB = R['penalty2weight_biases']  # Regularization parameter for weights and biases
    learning_rate = R['learning_rate']

    input_dim = R['input_dim']
    equation = 1


    if equation == 1:
        # equation 1
        #三维对流扩散方程的高精度紧致差分格式
        region_l = 0.0
        region_r = 1.0
        region_b = 0.0
        region_t = 1.0
        region_f = 0.0
        region_be = 1.0
        init_time = 0.0
        end_time = 1
        p = 1
        q = 1
        r = 1
        a = 1/(np.square(np.pi)*3)
        temp1 = lambda x: torch.sin(np.pi * x)
        temp2 = lambda xyz: torch.mul(torch.mul(temp1(xyz[:, 0]), temp1(xyz[:, 1])), temp1(xyz[:, 2]))
        temp3 = lambda x: torch.cos(np.pi * x)
        temp4 = lambda xyz: torch.mul(torch.mul(temp3(xyz[:, 0]), temp1(xyz[:, 1])), temp1(xyz[:, 2]))
        temp5 = lambda xyz: torch.mul(torch.mul(temp1(xyz[:, 0]), temp3(xyz[:, 1])), temp1(xyz[:, 2]))
        temp6 = lambda xyz: torch.mul(torch.mul(temp1(xyz[:, 0]), temp1(xyz[:, 1])), temp3(xyz[:, 2]))
        f = lambda x, t: np.pi * torch.mul(torch.exp(-t), (temp4(x)+temp5(x)+temp6(x)))
        u_true = lambda xyz, t: torch.mul(torch.exp(-t), temp2(xyz))
        u_left = lambda x, t: torch.zeros_like(x[:, 0])
        u_right = lambda x, t: torch.zeros_like(x[:, 0])
        u_init = lambda X, t: temp2(X)
    elif equation == 2:
        # equation 1
        #三维对流扩散方程的高精度紧致差分格式
        region_l = 0.0
        region_r = np.pi
        region_b = 0.0
        region_t = np.pi
        region_f = 0.0
        region_be = np.pi
        init_time = 0.0
        end_time = 1
        p = 1
        q = 1
        r = 1
        a = 1
        temp1 = lambda x: torch.sin(x)
        temp2 = lambda xyz: torch.mul(torch.mul(temp1(xyz[:, 0]), temp1(xyz[:, 1])), temp1(xyz[:, 2]))
        temp3 = lambda x: torch.cos(x)
        temp4 = lambda xyz: torch.mul(torch.mul(temp3(xyz[:, 0]), temp1(xyz[:, 1])), temp1(xyz[:, 2]))
        temp5 = lambda xyz: torch.mul(torch.mul(temp1(xyz[:, 0]), temp3(xyz[:, 1])), temp1(xyz[:, 2]))
        temp6 = lambda xyz: torch.mul(torch.mul(temp1(xyz[:, 0]), temp1(xyz[:, 1])), temp3(xyz[:, 2]))
        f = lambda x, t: torch.mul(torch.exp(-t), (temp4(x)+temp5(x)+temp6(x)))
        u_true = lambda xyz, t: torch.mul(torch.exp(-t), temp2(xyz))
        u_left = lambda x, t: torch.zeros_like(x[:, 0])
        u_right = lambda x, t: torch.zeros_like(x[:, 0])
        u_init = lambda X, t: temp2(X)


    model = MscaleDNN(input_dim=R['input_dim'] + 1, out_dim=R['output_dim'], hidden_layer=R['hidden_layers'],
                      Model_name=R['model2NN'], name2actIn=R['name2act_in'], name2actHidden=R['name2act_hidden'],
                      name2actOut=R['name2act_out'], opt2regular_WB='L0', repeat_highFreq=R['repeat_highFreq'],
                      type2numeric='float32', factor2freq=R['freq'], use_gpu=R['use_gpu'], No2GPU=R['gpuNo'])
    if True == R['use_gpu']:
        model = model.cuda(device='cuda:' + str(R['gpuNo']))

    params2Net = model.DNN.parameters()

    # 定义优化方法，并给定初始学习率
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr)                     # SGD
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr, momentum=0.8)       # momentum
    # optimizer = torch.optim.RMSprop(params2Net, lr=init_lr, alpha=0.95)     # RMSProp
    optimizer = torch.optim.Adam(params2Net, lr=learning_rate)  # Adam

    # 定义更新学习率的方法
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.99)

    loss_it_all, loss_bd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    test_epoch = []
    loss_init_all = []

    if R['testData_model'] == 'random_generate':
        # 生成测试数据，用于测试训练后的网络
        test_bach_size = 100
        size2test = 100
        # test_bach_size = 6400
        # size2test = 80
        # test_bach_size = 10000
        # size2test = 100

        # ---------------------------------------------------------------------------------#
        test_xy_bach_x = np.linspace(region_l, region_r, test_bach_size).reshape(-1, 1)
        test_xy_bach_y = np.linspace(region_b, region_t, test_bach_size).reshape(-1, 1)
        x_repeat = np.repeat(test_xy_bach_x, test_bach_size).reshape(-1, 1)
        t2 = list(test_xy_bach_y)
        t1 = list(test_xy_bach_y)
        for i in range(test_bach_size - 1):
            t2.extend(t1)
        y_repeat = np.array(t2)
        z_bach = np.ones_like(x_repeat) * 0.25
        t_bach = np.ones_like(x_repeat) * 0.5
        test_xyz_bach = np.concatenate([x_repeat, y_repeat, z_bach], -1)
        test_xyzt_bach = np.concatenate([x_repeat, y_repeat, z_bach, t_bach], -1)
        # ------------------------------------------#

    saveData.save_testData_or_solus2mat(test_xyzt_bach, dataName='testXY', outPath=R['FolderName'])

    test_xyz_bach = test_xyz_bach.astype(np.float32)
    test_xyz_torch = torch.from_numpy(test_xyz_bach).reshape(test_bach_size*test_bach_size, 3)
    test_xyzt_bach = test_xyzt_bach.astype(np.float32)
    test_xyzt_torch = torch.from_numpy(test_xyzt_bach).reshape(test_bach_size*test_bach_size, 4)

    # 生成左右边界
    # xl_bd_batch = np.ones(shape=[batchsize_bd, 1], dtype=np.float32) * region_l
    # xl_bd_batch = torch.from_numpy(xl_bd_batch)
    # xr_bd_batch = np.ones(shape=[batchsize_bd, 1], dtype=np.float32) * region_r
    # xr_bd_batch = torch.from_numpy(xr_bd_batch)
    t_init_batch = np.ones(shape=[batchsize_init, 1], dtype=np.float32) * init_time
    t_init_batch = torch.from_numpy(t_init_batch)

    if True == R['use_gpu']:
        test_xyzt_torch = test_xyzt_torch.cuda(device='cuda:' + str(R['gpuNo']))
        test_xyz_torch = test_xyz_torch.cuda(device='cuda:' + str(R['gpuNo']))
        # xl_bd_batch = xl_bd_batch.cuda(device='cuda:' + str(R['gpuNo']))
        # xr_bd_batch = xr_bd_batch.cuda(device='cuda:' + str(R['gpuNo']))
        t_init_batch = t_init_batch.cuda(device='cuda:' + str(R['gpuNo']))

    # xl_bd_batch.requires_grad_(True)
    # xr_bd_batch.requires_grad_(True)
    Utrue2test = u_true(test_xyz_torch, test_xyzt_torch[:, 3].reshape(-1, 1))
    t0 = time.time()
    for i_epoch in range(R['max_epoch'] + 1):
        # 内部点
        x_in_batch = dataUtilizer2torch.rand_it(batchsize_it, input_dim, region_a=region_l, region_b=region_r,
                                                to_float=True, to_cuda=R['use_gpu'],
                                                gpu_no=R['gpuNo'], use_grad2x=True)
        t_in_batch = dataUtilizer2torch.rand_it(batchsize_it, 1, region_a=init_time, region_b=end_time,
                                                to_float=True, to_cuda=R['use_gpu'],
                                                gpu_no=R['gpuNo'], use_grad2x=True)

        # 边界点 左右边界时间点取相同
        t_bd_batch = dataUtilizer2torch.rand_it(batchsize_bd, 1, region_a=init_time, region_b=end_time,
                                                to_float=True, to_cuda=R['use_gpu'],
                                                gpu_no=R['gpuNo'], use_grad2x=False)
        x_init_batch = dataUtilizer2torch.rand_it(batchsize_init, 3, region_a=region_l, region_b=region_r,
                                                  to_float=True, to_cuda=R['use_gpu'],
                                                  gpu_no=R['gpuNo'], use_grad2x=False)
        xl_bd_batch,xr_bd_batch,xb_bd_batch,xt_bd_batch,xf_bd_batch,xbe_bd_batch = dataUtilizer2torch.rand_bd_3D_2(batch_size=batchsize_bd, variable_dim=3,
                                                                  region_l=region_l, region_r=region_r,
                                                                  region_b=region_b, region_t=region_t,
                                                                  region_f=region_f, region_be=region_be,
                                                                  to_float=True, to_cuda=R['use_gpu'],
                                                                  gpu_no=R['gpuNo'], use_grad=False)

        # 计算损失函数
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

        if R['activate_penalty2bd_increase'] == 1:
            if i_epoch < int(R['max_epoch'] / 10):
                temp_penalty_init = init_penalty_init
            elif i_epoch < int(R['max_epoch'] / 5):
                temp_penalty_init = 10 * init_penalty_init
            elif i_epoch < int(R['max_epoch'] / 4):
                temp_penalty_init = 50 * init_penalty_init
            elif i_epoch < int(R['max_epoch'] / 2):
                temp_penalty_init = 100 * init_penalty_init
            elif i_epoch < int(3 * R['max_epoch'] / 4):
                temp_penalty_init = 200 * init_penalty_init
            else:
                temp_penalty_init = 500 * init_penalty_init
        else:
            temp_penalty_init = init_penalty_init

        # 内部点损失 用pinn就没有初始点的选取
        UNN2train, loss_it = model.loss_3DV(XYZ=x_in_batch, T=t_in_batch, loss_type=R['loss_type'], p=p, q=q, r=r,
                                            a=a, fside=f, if_lambda2fside=True)
        # if equation == 4:
        #     loss_bd2left = model.loss2bd_neumann(X_bd=xl_bd_batch, T_bd=t_bd_batch, Ubd_exact=u_left,
        #                                          if_lambda2Ubd=True, loss_type=R['loss_type'])
        #     loss_bd2right = model.loss2bd_neumann(X_bd=xr_bd_batch, T_bd=t_bd_batch, Ubd_exact=u_right,
        #                                       if_lambda2Ubd=True, loss_type=R['loss_type'])
        # else:
        loss_bd2left = model.loss2bd_dirichlet_3D(XYZ_bd=xl_bd_batch, T_bd=t_bd_batch, Ubd_exact=u_left,
                                               if_lambda2Ubd=True, loss_type=R['loss_type'])
        loss_bd2right = model.loss2bd_dirichlet_3D(XYZ_bd=xr_bd_batch, T_bd=t_bd_batch, Ubd_exact=u_right,
                                                if_lambda2Ubd=True, loss_type=R['loss_type'])
        loss_bd2bottom = model.loss2bd_dirichlet_3D(XYZ_bd=xb_bd_batch, T_bd=t_bd_batch, Ubd_exact=u_left,
                                               if_lambda2Ubd=True, loss_type=R['loss_type'])
        loss_bd2top= model.loss2bd_dirichlet_3D(XYZ_bd=xt_bd_batch, T_bd=t_bd_batch, Ubd_exact=u_right,
                                                if_lambda2Ubd=True, loss_type=R['loss_type'])
        loss_bd2front = model.loss2bd_dirichlet_3D(XYZ_bd=xf_bd_batch, T_bd=t_bd_batch, Ubd_exact=u_left,
                                               if_lambda2Ubd=True, loss_type=R['loss_type'])
        loss_bd2back = model.loss2bd_dirichlet_3D(XYZ_bd=xbe_bd_batch, T_bd=t_bd_batch, Ubd_exact=u_right,
                                                if_lambda2Ubd=True, loss_type=R['loss_type'])
        loss_init = model.loss2init(XYZ=x_init_batch, T_init=t_init_batch, Uinit_exact=u_init,
                                    if_lambda2Uinit=True, loss_type=R['loss_type'])

        loss_bd = loss_bd2left + loss_bd2right + loss_bd2back + loss_bd2front + loss_bd2bottom + loss_bd2top
        # PWB = penalty2WB * model.get_regularSum2WB()
        loss = loss_it + loss_bd * temp_penalty_bd + loss_init * temp_penalty_init

        loss_it_all.append(loss_it.item())
        loss_all.append(loss.item())

        optimizer.zero_grad()  # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()  # 对loss关于Ws和Bs求偏导
        optimizer.step()  # 更新参数Ws和Bs
        scheduler.step()



        if i_epoch % 1000 == 0:
            # 把这个计算的放进来
            Uexact2train = u_true(x_in_batch, t_in_batch)
            train_mse = torch.mean(torch.square(UNN2train - Uexact2train))
            train_rel = train_mse / torch.mean(torch.square(Uexact2train))
            train_mse_all.append(train_mse.item())
            train_rel_all.append(train_rel.item())

            run_times = time.time() - t0
            PWB = 0.0
            tmp_lr = optimizer.param_groups[0]['lr']
            DNN_Log_Print.print_and_log_train_one_epoch2Ocean(
                i_epoch, run_times, tmp_lr, temp_penalty_bd, temp_penalty_init, PWB, loss_it.item(), loss_bd.item(),
                loss_init.item(), loss.item(),
                train_mse.item(), train_rel.item(), log_out=log_fileout)

            test_epoch.append(i_epoch / 1000)


            #
            # point_square_error = torch.square(Utrue2test - UNN2test)
            # test_mse = torch.mean(point_square_error)
            # test_rel = test_mse / torch.mean(torch.square(Utrue2test))
            # test_mse_all.append(test_mse.item())
            # test_rel_all.append(test_rel.item())
            # DNN_tools.print_and_log_test_one_epoch(test_mse.item(), test_rel.item(), log_out=log_fileout)
    UNN2test = model.evalue_MscaleDNN(XY_points=test_xyzt_torch)
    # # ------------------- save the training results into mat file and plot them -------------------------
    # # saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=R['activate_func'],
    # #                                      outPath=R['FolderName'])
    # # saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=R['activate_func'], outPath=R['FolderName'])
    # #
    # # plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
    # # plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
    # #                                   yaxis_scale=True)
    # # plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])
    # #
    # # saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=R['activate_func'], outPath=R['FolderName'])
    # # plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=R['activate_func'], seedNo=R['seed'],
    # #                                      outPath=R['FolderName'], yaxis_scale=True)
    #
    # # ----------------------  save testing results to mat files, then plot them --------------------------------
    if True == R['use_gpu']:
        utrue2test_numpy = Utrue2test.cpu().detach().numpy()
        unn2test_numpy = UNN2test.cpu().detach().numpy()
        unn2test_numpy = unn2test_numpy.reshape((size2test, size2test))
    else:
        utrue2test_numpy = Utrue2test.detach().numpy()
        unn2test_numpy = UNN2test.detach().numpy()

    saveData.save_2testSolus2mat(utrue2test_numpy, unn2test_numpy, actName='utrue',
                                 actName1='sin', outPath=R['FolderName'])
    #
    # # plotData.plot_Hot_solution2test(utrue2test_numpy, size_vec2mat=size2test**2, actName='Utrue',
    # #                                 seedNo=R['seed'], outPath=R['FolderName'])
    # plotData.plot_Hot_solution2test(unn2test_numpy, size_vec2mat=size2test, actName=R['name2act_hidden'],
    #                                 seedNo=R['seed'], outPath=R['FolderName'])
    #
    # # saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=R['name2act_hidden'], outPath=R['FolderName'])
    # # plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=R['activate_func'],
    # #                           seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)
    # #
    # # saveData.save_test_point_wise_err2mat(point_square_error_numpy, actName=R['name2act_hidden'],
    # #                                       outPath=R['FolderName'])
    #
    # # plotData.plot_Hot_point_wise_err(point_square_error_numpy, size_vec2mat=size2test,
    # #                                  actName=R['activate_func'], seedNo=R['seed'], outPath=R['FolderName'])


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
    store_file = 'Soft PINN 3DV'
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

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  复制并保存当前文件 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    R['max_epoch'] = 10000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of multi-scale problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    R['input_dim'] = 3  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1  # 输出维数

    R['PDE_type'] = 'PINN'
    R['equa_name'] = 'PINN'

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 训练集的设置(内部和边界)
    R['batch_size2interior'] = 8000  # 内部训练数据的批大小
    R['batch_size2boundary'] = 4000  # 边界训练数据的批大小
    R['batch_size2init'] = 3000

    # 装载测试数据模式
    # R['testData_model'] = 'loadData'
    R['testData_model'] = 'random_generate'

    R['loss_type'] = 'L2_loss'  # loss类型:L2 loss
    # R['loss_type'] = 'variational_loss'                      # loss类型:PDE变分
    # R['loss_type'] = 'lncosh_loss'
    R['lambda2lncosh'] = 0.5

    R['optimizer_name'] = 'Adam'  # 优化器
    R['learning_rate'] = 2e-4  # 学习率
    # R['learning_rate'] = 0.001              # 学习率
    R['learning_rate_decay'] = 5e-5  # 学习率 decay
    R['train_model'] = 'union_training'

    # 正则化权重和偏置的模式
    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'
    # R['penalty2weight_biases'] = 0.000                    # Regularization parameter for weights
    R['penalty2weight_biases'] = 0.001  # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025                 # Regularization parameter for weights

    # 边界的惩罚处理方式,以及边界的惩罚因子
    R['activate_penalty2bd_increase'] = 1
    # R['activate_penalty2bd_increase'] = 0
    R['init_boundary_penalty'] = 20

    R['activate_penalty2init_increase'] = 1
    # R['activate_penalty2init_increase'] = 0
    R['init_penalty2init'] = 20

    # 网络的频率范围设置
    R['freq'] = np.concatenate(([1], np.arange(1, 40 - 1)), axis=0)
    R['repeat_highFreq'] = True

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['model2NN'] = 'DNN'
    # R['model2NN'] = 'Scale_DNN'
    R['model2NN'] = 'Fourier_DNN'

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model2NN'] == 'Fourier_DNN':
        R['hidden_layers'] = (100, 150, 80, 80, 50)
        # R['hidden_layers'] = (125, 200, 100, 100, 80)  # 1*125+250*200+200*200+200*100+100*100+100*50+50*1=128205
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
    R['name2act_hidden'] = 'sinAddcos'
    # R['name2act_hidden'] = 'Enh_tanh'
    # R['name2act_hidden'] = 'elu'
    # R['name2act_hidden'] = 'phi'

    R['name2act_out'] = 'linear'

    R['sfourier'] = 1.0

    R['use_gpu'] = True

    solve_Multiscale_PDE(R)
