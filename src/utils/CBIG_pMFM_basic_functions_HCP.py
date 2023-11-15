# /usr/bin/env python
# Written by Kong Xiaolu, Zhang Shaoshi and CBIG under MIT license:
# https://github.com/ThomasYeolab/CBIG/blob/master/LICENSE.md

import os
import csv
import math
from pathlib import Path
import time
import numpy as np
import torch
import scipy.io as sio
from scipy.optimize import fsolve
import pandas as pd
'''********************  Functions for computing simulated BOLD signals   *************************'''


def CBIG_mfm_multi_simulation(parameter, sc_mat, t_epochlong, n_dup, countloop, d_t, is_memory_insufficient=0):
    '''
    Function used to generate the simulated BOLD signal using mean field model and hemodynamic model
    Each parameter set is ussed to simulated multiple times to get stable result
    Args:
        parameter:  (N*3+1)*M matrix.
                    N is the number of ROI
                    M is the number of candidate parameter sets. 
                    Each column of matrix presents a parameter set, where:
                    parameter[0:N]: recurrent strength within excitatory population (wEE)
                    parameter[N:2*N]: connection strength from excitatory population to inhibitory population (wEI)
                    parameter[2*N]: Gloable constant G
                    parameter[2*N+1:3*N+1]: noise amplitude sigma
        sc_mat:     N*N structural connectivity matrix
        t_epochlong:total simulated time (exclude burn-in period)
        n_dup:      Number of times each parameter set is simulated
        countloop:  An integer used to keep track of which iteration CMAES is currently at
        d_t:        Time step size for Euler integration (6ms for training, 0.5ms for validation)
        is_memory_insufficient:
                    A binary flag indicating if the entire simulated time series can be fit into GPU memory

    Returns:
        bold_d:     simulated BOLD signal
        S_E_all:    temporal average of excitatory synaptic gating variable
        S_I_all:    temporal average of inhibitory synaptic gating variable
        r_E_all:    temporal average of excitatory firing rate
        J_I:        feedback inhibition control (FIC) strength from inhibitory population to excitatoty population
    '''

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    else:
        torch.set_default_tensor_type('torch.DoubleTensor')

    # Initializing system parameters
    preheat = 5000
    kstart = 0.
    t_pre = 60 * 2.4
    kend = t_pre + 60 * t_epochlong
    t_bold = 0.72

    # Setting sampling ratio
    k_p = torch.arange(kstart, kend + d_t, d_t)
    n_num = parameter.shape[1]
    n_set = n_dup * n_num
    parameter = parameter.repeat(1, n_dup)
    n_nodes = sc_mat.shape[0]
    n_samples = k_p.shape[0]

    # Initializing neural activity
    S_E = torch.zeros((n_nodes, n_set))
    S_I = torch.zeros((n_nodes, n_set))
    I_I_ave = 0.296385800197336 * torch.ones(n_nodes, n_set)

    # Initializing hemodynamic activity
    f_mat = torch.ones((n_nodes, n_set, 4))
    z_t = torch.zeros((n_nodes, n_set))
    f_t = torch.ones((n_nodes, n_set))
    v_t = torch.ones((n_nodes, n_set))
    q_t = torch.ones((n_nodes, n_set))
    f_mat[:, :, 0] = z_t
    S_E[:, :] = 0.1641205151
    S_I[:, :] = 0.1433408985

    # Wiener process
    w_coef = parameter[2 * n_nodes + 1:3 * n_nodes + 1, :]
    w_l = k_p.shape[0]
    if is_memory_insufficient == 0:
        d_w = math.sqrt(d_t) * torch.randn(n_dup, n_nodes, w_l + preheat)

    p_costant = 0.34
    v_0 = 0.02
    k_1 = 4.3 * 28.265 * 3 * 0.0331 * p_costant
    k_2 = 0.47 * 110 * 0.0331 * p_costant
    k_3 = 0.53
    count = 0
    y_bold = torch.zeros((n_nodes, n_set, int(n_samples / (t_bold / d_t) + 1)))

    # solve I_I_ave
    S_E_ave = 0.1641205151
    I_E_ave = 0.3772259651
    J_NMDA = 0.15
    w_EE = parameter[0:n_nodes, :]
    G = parameter[2 * n_nodes, :]
    w_EI = parameter[n_nodes:2 * n_nodes, :]
    W_E = 1.
    I0 = 0.382

    # Parameters for firing rate
    a_I = 615.
    b_I = 177.
    d_I = 0.087

    # Parameters for synaptic activity/currents
    tau_I = 0.01
    w_EI = w_EI.cpu().numpy()
    I_I_ave = I_I_ave.cpu().numpy()
    for i in range(n_set):
        I_I_ave_one_set = np.atleast_2d(I_I_ave[:, i]).T
        w_EI_one_set = np.atleast_2d(w_EI[:, i]).T
        I_I_ave[:, i], infodict, ier, mseg = fsolve(I_I_fixed_pt, I_I_ave_one_set, args=w_EI_one_set, full_output=True)
    I_I_ave = torch.from_numpy(I_I_ave).type(torch.DoubleTensor)
    if torch.cuda.is_available():
        I_I_ave = I_I_ave.cuda()
    S_I_ave = tau_I * (a_I * I_I_ave - b_I) / (1 - torch.exp(-d_I * (a_I * I_I_ave - b_I)))

    # calculate J_I
    J_I = torch.div(
        W_E * I0 + w_EE * J_NMDA * S_E_ave + G * J_NMDA * torch.sum(sc_mat, 1).view(-1, 1).repeat(1, n_set) * S_E_ave -
        I_E_ave, S_I_ave)

    # Warm up
    start = time.time()
    print('Warm Up Euler multi...')
    r_E_all = torch.zeros(n_nodes, n_set)
    S_E_all = torch.zeros(n_nodes, n_set)
    S_I_all = torch.zeros(n_nodes, n_set)
    for i in range(preheat):
        dS_E, dS_I, r_E = CBIG_mfm_rfMRI_ode(S_E, S_I, J_I, parameter, sc_mat)
        if is_memory_insufficient == 0:
            noise_level = d_w[:, :, i].repeat(1, 1, n_num).contiguous().view(-1, n_nodes)
            S_E = S_E + dS_E * d_t + w_coef * torch.transpose(
                noise_level, 0, 1)  # training time step size 6ms; validation + testing time step size 0.5ms;
            S_I = S_I + dS_I * d_t + w_coef * torch.transpose(noise_level, 0, 1)
        else:
            S_E = S_E + dS_E * d_t + w_coef * torch.randn(n_nodes, n_dup).repeat(1, n_num) * math.sqrt(d_t)
            S_I = S_I + dS_I * d_t + w_coef * torch.randn(n_nodes, n_dup).repeat(1, n_num) * math.sqrt(d_t)

    # Main body: calculation
    print('Main Body Euler multi...')
    for i in range(n_samples):
        print(i, 'sample in the main body')
        dS_E, dS_I, r_E = CBIG_mfm_rfMRI_ode(S_E, S_I, J_I, parameter, sc_mat)
        if is_memory_insufficient == 0:
            noise_level = d_w[:, :, i].repeat(1, 1, n_num).contiguous().view(-1, n_nodes)
            S_E = S_E + dS_E * d_t + w_coef * torch.transpose(noise_level, 0, 1)
            S_I = S_I + dS_I * d_t + w_coef * torch.transpose(noise_level, 0, 1)
        else:
            S_E = S_E + dS_E * d_t + w_coef * torch.randn(n_nodes, n_dup).repeat(1, n_num) * math.sqrt(d_t)
            S_I = S_I + dS_I * d_t + w_coef * torch.randn(n_nodes, n_dup).repeat(1, n_num) * math.sqrt(d_t)

        d_f = CBIG_mfm_rfMRI_BW_ode(S_E, f_mat)
        f_mat = f_mat + d_f * d_t
        z_t, f_t, v_t, q_t = torch.chunk(f_mat, 4, dim=2)
        y_bold_temp = 100 / p_costant * v_0 * (k_1 * (1 - q_t) + k_2 * (1 - q_t / v_t) + k_3 * (1 - v_t))
        y_bold[:, :, count] = y_bold_temp[:, :, 0]
        count = count + ((i + 1) % (int(round(t_bold / d_t))) == 0) * 1

        r_E_all = r_E_all + r_E
        S_E_all = S_E_all + S_E
        S_I_all = S_I_all + S_I

    elapsed = time.time() - start
    r_E_all = r_E_all / n_samples
    S_E_all = S_E_all / n_samples
    S_I_all = S_I_all / n_samples

    # an extreme approach, if at the last frame, the excitatory firing rate of any node
    # is below 2.7Hz or above 3.3Hz, the first node of the last frame of BOLD is set to nan
    # (essentially ignore this set of parameters)
    for i in range(n_set):
        if (2.7 > r_E_all[:, i]).any() or (r_E_all[:, i] > 3.3).any():
            y_bold[:, i, :] = float('nan')

    # Downsampling
    cut_index = int(t_pre / t_bold)
    bold_d = y_bold[:, :, cut_index + 1:y_bold.shape[2]]

    print('The time used for calculating simulated BOLD signal is: ', elapsed)

    return bold_d, S_E_all, S_I_all, r_E_all, J_I


def CBIG_mfm_single_simulation(parameter, sc_mat, t_epochlong, d_t=0.0005, extrapolation=0):
    '''
    Function used to generate the simulated BOLD signal using mean field model and hemodynamic model
    Each parameter set is ussed to simulated one time
    Args:
        parameter:  (N*3+1)*M matrix.
                    N is the number of ROI
                    M is the number of candidate parameter sets.
                    Each column of matrix presents a parameter set, where:
                    parameter[0:N]: recurrent strength within excitatory population (wEE)
                    parameter[N:2*N]: connection strength from excitatory population to inhibitory population (wEI)
                    parameter[2*N]: Gloable constant G
                    parameter[2*N+1:3*N+1]: noise amplitude sigma
        sc_mat:     N*N structural connectivity matrix
        t_epochlong:total simulated time (exclude burn-in period)
        d_t:        Time step size for Euler integration (0.5ms as default for testing)
        extrapolation:
                    A binary flag indicating if the goal is to get an extrapolated E/I ratio. If '1', the forward simulation
                    is skipped and only an estimated E/I ratio is returned

    Returns:
        if extrapolate == 0
            bold_d:     simulated BOLD signal
            S_E_all:    temporal average of excitatory synaptic gating variable
            S_I_all:    temporal average of inhibitory synaptic gating variable
            r_E_all:    temporal average of excitatory firing rate
            J_I:        feedback inhibition control (FIC) strength from inhibitory population to excitatoty population
        if extrapolate == 1
            S_E_ave/S_I_ave: an estimate of E/I ratio
    '''

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    else:
        torch.set_default_tensor_type('torch.DoubleTensor')

    # Initializing system parameters
    kstart = 0.
    t_pre = 60 * 2.4
    kend = t_pre + 60 * t_epochlong
    t_bold = 0.72

    # sampling ratio
    k_p = torch.arange(kstart, kend + d_t, d_t)
    n_nodes = sc_mat.shape[0]
    n_samples = k_p.shape[0]
    n_set = parameter.shape[1]

    # Initializing neural activity
    S_E = torch.zeros((n_nodes, n_set))
    S_I = torch.zeros((n_nodes, n_set))
    I_I_ave = 0.296385800197336 * torch.ones(n_nodes, n_set)

    # Initializing hemodynamic activity
    f_mat = torch.ones((n_nodes, n_set, 4))
    z_t = torch.zeros((n_nodes, n_set))
    f_t = torch.ones((n_nodes, n_set))
    v_t = torch.ones((n_nodes, n_set))
    q_t = torch.ones((n_nodes, n_set))
    f_mat[:, :, 0] = z_t
    S_E[:, :] = 0.1641205151
    S_I[:, :] = 0.1433408985

    # Wiener process
    w_coef = parameter[2 * n_nodes + 1:3 * n_nodes + 1, :]
    p_costant = 0.34
    v_0 = 0.02
    k_1 = 4.3 * 28.265 * 3 * 0.0331 * p_costant
    k_2 = 0.47 * 110 * 0.0331 * p_costant
    k_3 = 0.53
    count = 0
    y_bold = torch.zeros((n_nodes, n_set, int(n_samples / (t_bold / d_t) + 1)))

    # solve I_I_ave
    S_E_ave = 0.1641205151
    I_E_ave = 0.3772259651
    J_NMDA = 0.15
    w_EE = parameter[0:n_nodes, :]
    G = parameter[2 * n_nodes, :]
    w_EI = parameter[n_nodes:2 * n_nodes, :]
    W_E = 1.
    I0 = 0.382

    # Parameters for firing rate
    a_I = 615.
    b_I = 177.
    d_I = 0.087

    # Parameters for synaptic activity/currents
    tau_I = 0.01
    w_EI = w_EI.cpu().numpy()
    I_I_ave = I_I_ave.cpu().numpy()
    for i in range(n_set):
        I_I_ave_one_set = np.atleast_2d(I_I_ave[:, i]).T
        w_EI_one_set = np.atleast_2d(w_EI[:, i]).T
        I_I_ave[:, i], infodict, ier, mseg = fsolve(I_I_fixed_pt, I_I_ave_one_set, args=w_EI_one_set, full_output=True)
    I_I_ave = torch.from_numpy(I_I_ave).type(torch.DoubleTensor)
    if torch.cuda.is_available():
        I_I_ave = I_I_ave.cuda()
    print(mseg)
    S_I_ave = tau_I * (a_I * I_I_ave - b_I) / (1 - torch.exp(-d_I * (a_I * I_I_ave - b_I)))
    if extrapolation == 1:
        return S_E_ave / S_I_ave

    # calculate J_I
    J_I = torch.div(
        W_E * I0 + w_EE * J_NMDA * S_E_ave + G * J_NMDA * torch.sum(sc_mat, 1).view(-1, 1).repeat(1, n_set) * S_E_ave -
        I_E_ave, S_I_ave)

    # Warm up
    start = time.time()
    print('Warm Up Euler single ...')
    r_E_all = torch.zeros(n_nodes, n_set)
    S_E_all = torch.zeros(n_nodes, n_set)
    S_I_all = torch.zeros(n_nodes, n_set)
    for i in range(5000):
        dS_E, dS_I, r_E = CBIG_mfm_rfMRI_ode(S_E, S_I, J_I, parameter, sc_mat)
        S_E = S_E + dS_E * d_t + w_coef * torch.randn(n_nodes, n_set) * math.sqrt(d_t)
        S_I = S_I + dS_I * d_t + w_coef * torch.randn(n_nodes, n_set) * math.sqrt(d_t)

    # Main body: calculation
    print('Main Body Euler single ...')
    for i in range(n_samples):
        dS_E, dS_I, r_E = CBIG_mfm_rfMRI_ode(S_E, S_I, J_I, parameter, sc_mat)
        S_E = S_E + dS_E * d_t + w_coef * torch.randn(n_nodes, n_set) * math.sqrt(d_t)
        S_I = S_I + dS_I * d_t + w_coef * torch.randn(n_nodes, n_set) * math.sqrt(d_t)

        d_f = CBIG_mfm_rfMRI_BW_ode(S_E, f_mat)
        f_mat = f_mat + d_f * d_t
        z_t, f_t, v_t, q_t = torch.chunk(f_mat, 4, dim=2)
        y_bold_temp = 100 / p_costant * v_0 * (k_1 * (1 - q_t) + k_2 * (1 - q_t / v_t) + k_3 * (1 - v_t))
        y_bold[:, :, count] = y_bold_temp[:, :, 0]
        count = count + ((i + 1) % (int(round(t_bold / d_t))) == 0) * 1

        r_E_all = r_E_all + r_E
        S_E_all = S_E_all + S_E
        S_I_all = S_I_all + S_I

    elapsed = time.time() - start
    r_E_all = r_E_all / n_samples
    S_E_all = S_E_all / n_samples
    S_I_all = S_I_all / n_samples

    for i in range(n_set):
        if (2.7 > r_E_all[:, i]).any() or (r_E_all[:, i] > 3.3).any():
            y_bold[:, i, :] = float('nan')

    print('The time used for calculating simulated BOLD signal is: ', elapsed, flush=True)

    # Downsampling
    cut_index = int(t_pre / t_bold)
    bold_d = y_bold[:, :, cut_index + 1:y_bold.shape[2]]

    return bold_d, S_E_all, S_I_all, r_E_all, J_I


def CBIG_mfm_rfMRI_ode(S_E, S_I, J_I, parameter, sc_mat):
    '''
    This function is an implementation of Deco 2014 FIC model
    Args: 
        S_E:        N*M matrix excitatory synaptic gating variable
        S_I:        N*M matrix inhibitory synaptic gating variable
        J_I:        N*M martix feedback inhibition control
        parameter:  (N*3+1)*M matrix. 
                    N is the number of ROI
                    M is the number of candidate parameter sets. 
                    Each column of matrix presents a parameter set, where:
                    parameter[0:N]: recurrent strength within excitatory population (wEE)
                    parameter[N:2*N]: connection strength from excitatory population to inhibitory population (wEI)
                    parameter[2*N]: Gloable constant G
                    parameter[2*N+1:3*N+1]: noise amplitude sigma
        sc_mat:     N*N structural connectivity matrix
    Returns:
        dy:         N*M matrix represents derivatives of synaptic gating variable S
    '''

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    else:
        torch.set_default_tensor_type('torch.DoubleTensor')

    # Parameters for inputs and couplings
    number_roi = sc_mat.shape[0]
    n_set = parameter.shape[1]
    J_NMDA = 0.15
    w_EE = parameter[0:number_roi, :]
    G = parameter[2 * number_roi, :]
    w_EI = parameter[number_roi:2 * number_roi, :]
    W_E = 1.
    W_I = 0.7
    I0 = 0.382

    # Parameters for firing rate
    a_E = 310.
    b_E = 125.
    d_E = 0.16

    a_I = 615.
    b_I = 177.
    d_I = 0.087

    # Parameters for synaptic activity/currents
    tau_E = 0.1
    tau_I = 0.01
    gamma = 0.641

    # I_E
    I_E = W_E * I0 + J_NMDA * w_EE * S_E + J_NMDA * G.repeat(number_roi, 1) * torch.mm(sc_mat, S_E) - J_I * S_I

    # I_I
    I_I = W_I * I0 + J_NMDA * w_EI * S_E - S_I

    # r_E
    r_E = (a_E * I_E - b_E) / (1 - torch.exp(-d_E * (a_E * I_E - b_E)))

    # r_I
    r_I = (a_I * I_I - b_I) / (1 - torch.exp(-d_I * (a_I * I_I - b_I)))

    # dS_E (noise is added at the forward simulation step)
    dS_E = -S_E / tau_E + (1 - S_E) * gamma * r_E

    # dS_I (noise is added at the forward simulation step)
    dS_I = -S_I / tau_I + r_I

    print('r_E:', r_E)
    if torch.isnan(dS_E).any() or torch.isnan(dS_I).any() or torch.isnan(r_E).any() or torch.isnan(r_I).any():
        # dS_E = torch.zeros(S_E.shape)
        # dS_I = torch.zeros(S_I.shape)
        print('NaN!!!!!')
        print('dS_E:', dS_E)
        print('dS_I:', dS_I)
        print('r_E:', r_E)
        print('r_I:', r_I)
        return

    return dS_E, dS_I, r_E


def I_I_fixed_pt(I_I_ave, w_EI):
    '''
    Given an initial guess of the fixed point and wEI, compute the fixed point of inhibitory current 
    Args:
        I_I_ave:    an initial guess
        w_EI:       connection strength from excitatory population to inhibitory population
    Returns:
        I_I_ave_fixed_point:
                    computed inhibitory current fixed point
    '''
    I_I_ave = np.atleast_2d(I_I_ave).T
    W_I = 0.7
    J_NMDA = 0.15
    S_E_ave = 0.1641205151
    a_I = 615.
    b_I = 177.
    d_I = 0.087
    tau_I = 0.01
    I0 = 0.382
    I_I_ave_fixed_point = -I_I_ave + W_I * I0 + J_NMDA * w_EI * S_E_ave - (a_I * I_I_ave - b_I) / (
        1 - np.exp(-d_I * (a_I * I_I_ave - b_I))) * tau_I
    return I_I_ave_fixed_point.T.flatten()


def CBIG_mfm_rfMRI_BW_ode(y_t, F):
    '''
    This fucntion is to implement the hemodynamic model
    Args:
        y_t:        N*M matrix represents excitatory synaptic gating variable
        F:          Hemodynamic activity variables
    Returns:
        dF:         Derivatives of hemodynamic activity variables
    '''

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    else:
        torch.set_default_tensor_type('torch.DoubleTensor')

    # Hemodynamic model parameters
    beta = 0.65
    gamma = 0.41
    tau = 0.98
    alpha = 0.33
    p_constant = 0.34
    n_nodes = y_t.shape[0]
    n_set = y_t.shape[1]

    # Calculate derivatives
    dF = torch.zeros((n_nodes, n_set, 4))
    dF[:, :, 0] = y_t - beta * F[:, :, 0] - gamma * (F[:, :, 1] - 1)
    dF[:, :, 1] = F[:, :, 0]
    dF[:, :, 2] = 1 / tau * (F[:, :, 1] - F[:, :, 2]**(1 / alpha))
    dF[:, :, 3] = 1/tau*(F[:, :, 1]/p_constant*(1-(1-p_constant)**(1/F[:, :, 1])) - \
      F[:, :, 3]/F[:, :, 2]*F[:, :, 2]**(1/alpha))
    return dF


def CBIG_FCcorrelation_multi_simulation(emp_fc, bold_d, n_dup):
    '''
    This function is to calculate the FC correlation cost for multiple simulation
    BOLD signal results
    Args:
        emp_fc:     N*N group level FC matrix
                    N is number of ROI
        bold_d:     simulated BOLD signal
        n_dup:      Number of times each parameter set is simulated
    Returns:
        corr_cost:  FC correlation cost
        L1_cost:    FC L1 cost
        emp_fc_mean:
                    mean of empirical FC (upper triangle)  
    '''

    fc_timestart = time.time()

    # Calculate vectored simulated FC
    n_set = bold_d.shape[1]
    n_num = int(n_set / n_dup)
    n_nodes = emp_fc.shape[0]
    fc_mask = torch.triu(torch.ones(n_nodes, n_nodes), 1) == 1
    vect_len = int(n_nodes * (n_nodes - 1) / 2)
    sim_fc_vector = torch.zeros(n_set, vect_len)
    for i in range(n_set):
        sim_fc = torch_corr(bold_d[:, i, :])
        sim_fc_vector[i, :] = sim_fc[fc_mask]

    # Average the simulated FCs with same parameter set
    sim_fc_vector[sim_fc_vector != sim_fc_vector] = 0
    sim_fc_num = torch.zeros(n_num, vect_len)
    sim_fc_den = torch.zeros(n_num, 1)
    for k in range(n_dup):
        sim_fc_num = sim_fc_num + sim_fc_vector[k * n_num:(k + 1) * n_num, :]
        sim_fc_den = sim_fc_den + (sim_fc_vector[k * n_num:(k + 1) * n_num, 0:1] != 0).float()

    sim_fc_den[sim_fc_den == 0] = np.nan
    sim_fc_ave = sim_fc_num / sim_fc_den

    # Calculate FC correlation
    emp_fcm = emp_fc[fc_mask].repeat(n_num, 1)
    emp_fc_mean = torch.mean(emp_fc[fc_mask])
    corr_mass = torch_corr2(sim_fc_ave, emp_fcm)
    corr_cost = torch.diag(corr_mass)
    corr_cost = corr_cost.cpu().numpy()
    corr_cost = 1 - corr_cost
    corr_cost[np.isnan(corr_cost)] = 10

    # L1 cost
    L1_cost = torch.abs(torch.mean(sim_fc_ave, 1) - torch.mean(emp_fcm, 1))
    L1_cost = L1_cost.cpu().numpy()
    L1_cost[np.isnan(L1_cost)] = 5

    fc_elapsed = time.time() - fc_timestart
    print('Time using for calculating FC cost: ', fc_elapsed)

    return 1 * corr_cost, 1 * L1_cost, emp_fc_mean.cpu().numpy()


def CBIG_FCcorrelation_single_simulation(emp_fc, bold_d, n_dup):
    '''
    This function is to calculate the FC correlation cost for single simulation
    BOLD signal result
    Args:
        emp_fc:     N*N group level FC matrix
                    N is number of ROI
        bold_d:     simulated BOLD signal
        n_dup:      Number of times each parameter set is simulated
    Returns:
        corr_cost:  FC correlation cost
        L1_cost:    FC L1 cost
    '''

    fc_timestart = time.time()

    # Calculate vectored simulated FC
    n_set = bold_d.shape[1]
    n_nodes = emp_fc.shape[0]
    fc_mask = torch.triu(torch.ones(n_nodes, n_nodes), 1) == 1
    vect_len = int(n_nodes * (n_nodes - 1) / 2)
    sim_fc_vector = torch.zeros(n_set, vect_len)
    for i in range(n_set):
        sim_fc = torch_corr(bold_d[:, i, :])
        sim_fc_vector[i, :] = sim_fc[fc_mask]

    # Calculate FC correlation
    sim_fc_numpy = sim_fc_vector.cpu().numpy()
    emp_fc_numpy = emp_fc[fc_mask].cpu().numpy()
    time_dup = int(n_set / n_dup)
    corr_cost = np.zeros(time_dup)
    L1_cost = np.zeros(time_dup)

    for t in range(time_dup):
        sim_fc_numpy_temp = sim_fc_numpy[t * n_dup:(t + 1) * n_dup, :]
        sim_fc_mean = np.nanmean(sim_fc_numpy_temp, 0)
        corrmean_temp = np.corrcoef(sim_fc_mean, emp_fc_numpy)
        corr_cost[t] = 1 - corrmean_temp[1, 0]
        L1_cost[t] = np.abs(np.mean(sim_fc_mean) - np.mean(emp_fc_numpy))

    fc_elapsed = time.time() - fc_timestart
    print('Time using for calcualting FC correlation cost: ', fc_elapsed)
    return 1 * corr_cost, 1 * L1_cost


def CBIG_FCDKSstat_multi_simulation(emp_ks, bold_d, n_dup):
    '''
    This function is to calculate the FCD KS statistics cost for multiple simulation
    BOLD signal results
    Args:
        emp_cdf:    Group level KS statistics for empirical data
        bold_d:     simulated BOLD signal
        n_dup:      Number of times each parameter set is simulated
    Returns:
        ks_cost:   FCD KS statistics cost
    '''

    fcd_timestart = time.time()

    # Initializing the FC and FCD masks
    n_set = bold_d.shape[1]
    n_num = int(n_set / n_dup)
    n_nodes = bold_d.shape[0]
    window_size = 83
    time_lengh = 1200 - window_size + 1
    sub_num = 10
    resid_num = n_set % sub_num
    fc_edgenum = int(n_nodes * (n_nodes - 1) / 2)
    fc_mask = torch.triu(torch.ones(n_nodes, n_nodes), 1) == 1

    fc_maskm = torch.zeros(n_nodes * sub_num, n_nodes * sub_num)
    if torch.cuda.is_available():
        fc_maskm = fc_maskm.type(torch.cuda.ByteTensor)
    else:
        fc_maskm = fc_maskm.type(torch.ByteTensor)

    for i in range(sub_num):
        fc_maskm[n_nodes * i:n_nodes * (i + 1), n_nodes * i:n_nodes * (i + 1)] = fc_mask

    fc_mask_resid = torch.zeros(n_nodes * resid_num, n_nodes * resid_num)
    if torch.cuda.is_available():
        fc_mask_resid = fc_mask_resid.type(torch.cuda.ByteTensor)
    else:
        fc_mask_resid = fc_mask_resid.type(torch.ByteTensor)
    for i in range(resid_num):
        fc_mask_resid[n_nodes * i:n_nodes * (i + 1), n_nodes * i:n_nodes * (i + 1)] = fc_mask

    fcd_mask = torch.triu(torch.ones(time_lengh, time_lengh), 1) == 1

    # Calculating CDF for simualted FCD matrices
    fcd_hist = np.ones([10000, n_set])
    fc_mat = torch.zeros(fc_edgenum, sub_num, time_lengh)
    batch_num = math.floor(n_set / sub_num)
    fc_resid = torch.zeros(fc_edgenum, resid_num, time_lengh)

    for b in range(batch_num):
        bold_temp = bold_d[:, b * sub_num:(b + 1) * sub_num, :]
        bold_tempm = bold_temp.transpose(0, 1).contiguous().view(-1, 1200)
        for i in range(0, time_lengh):
            bold_fc = torch_corr(bold_tempm[:, i:i + window_size])
            cor_temp = bold_fc[fc_maskm]
            fc_mat[:, :, i] = torch.transpose(cor_temp.view(sub_num, fc_edgenum), 0, 1)

        for j in range(0, sub_num):
            fcd_temp = torch_corr(torch.transpose(fc_mat[:, j, :], 0, 1))
            fcd_hist_temp = np.histogram(fcd_temp[fcd_mask].cpu().numpy(), bins=10000, range=(-1., 1.))
            fcd_hist[:, j + b * sub_num] = fcd_hist_temp[0]

    if resid_num != 0:
        bold_temp = bold_d[:, batch_num * sub_num:n_set, :]
        bold_tempm = bold_temp.transpose(0, 1).contiguous().view(-1, 1200)
        for i in range(time_lengh):
            bold_fc = torch_corr(bold_tempm[:, i:i + window_size])
            cor_temp = bold_fc[fc_mask_resid]
            fc_resid[:, :, i] = torch.transpose(cor_temp.view(resid_num, fc_edgenum), 0, 1)

        for j in range(resid_num):
            fcd_temp = torch_corr(torch.transpose(fc_resid[:, j, :], 0, 1))
            fcd_hist_temp = np.histogram(fcd_temp[fcd_mask].cpu().numpy(), bins=10000, range=(-1., 1.))
            fcd_hist[:, j + sub_num * batch_num] = fcd_hist_temp[0]

    fcd_histcum = np.cumsum(fcd_hist, 0)
    fcd_histcumM = fcd_histcum.copy()
    fcd_histcumM[:, fcd_histcum[-1, :] != emp_ks[-1, 0]] = 0

    # Calculating KS statistics cost
    fcd_histcum_temp = np.zeros((10000, n_num))
    fcd_histcum_num = np.zeros((1, n_num))
    for k in range(n_dup):
        fcd_histcum_temp = fcd_histcum_temp + fcd_histcumM[:, k * n_num:(k + 1) * n_num]
        fcd_histcum_num = fcd_histcum_num + (fcd_histcumM[-1, k * n_num:(k + 1) * n_num] == emp_ks[-1, 0])
    fcd_histcum_ave = fcd_histcum_temp / fcd_histcum_num
    ks_diff = np.abs(fcd_histcum_ave - np.tile(emp_ks, [1, n_num]))
    ks_cost = ks_diff.max(0) / emp_ks[-1, 0]
    ks_cost[fcd_histcum_ave[-1, :] != emp_ks[-1, 0]] = 10

    fcd_elapsed = time.time() - fcd_timestart
    print('Time using for calcualting FCD KS statistics cost: ', fcd_elapsed)
    return ks_cost


def CBIG_FCDKSstat_single_simulation(emp_ks, bold_d, n_dup, window_size=83):
    '''
    This function is to calculate the FCD KS statistics cost for single simulation
    BOLD signal results
    Args:
        emp_ks:     Group level KS statistics for empirical data
        bold_d:     simulated BOLD signal
    Returns:
        ks_cost:    FCD KS statistics cost
    '''

    fcd_timestart = time.time()

    # Initializing the FC and FCD masks
    n_set = bold_d.shape[1]
    n_nodes = bold_d.shape[0]
    time_lengh = 1200 - window_size + 1
    sub_num = 10
    resid_num = n_set % sub_num
    fc_edgenum = int(n_nodes * (n_nodes - 1) / 2)
    fc_mask = torch.triu(torch.ones(n_nodes, n_nodes), 1) == 1
    fc_maskm = torch.zeros(n_nodes * sub_num, n_nodes * sub_num)
    if torch.cuda.is_available():
        fc_maskm = fc_maskm.type(torch.cuda.ByteTensor)
    else:
        fc_maskm = fc_maskm.type(torch.ByteTensor)

    for i in range(sub_num):
        fc_maskm[n_nodes * i:n_nodes * (i + 1), n_nodes * i:n_nodes * (i + 1)] = fc_mask

    fc_mask_resid = torch.zeros(n_nodes * resid_num, n_nodes * resid_num)

    if torch.cuda.is_available():
        fc_mask_resid = fc_mask_resid.type(torch.cuda.ByteTensor)
    else:
        fc_mask_resid = fc_mask_resid.type(torch.ByteTensor)
    for i in range(resid_num):
        fc_mask_resid[n_nodes * i:n_nodes * (i + 1), n_nodes * i:n_nodes * (i + 1)] = fc_mask

    fcd_mask = torch.triu(torch.ones(time_lengh, time_lengh), 1) == 1

    # Calculating CDF for simualted FCD matrices
    fcd_hist = np.ones([10000, n_set])
    fc_mat = torch.zeros(fc_edgenum, sub_num, time_lengh)
    batch_num = int(n_set / sub_num)
    fc_resid = torch.zeros(fc_edgenum, resid_num, time_lengh)

    for b in range(batch_num):
        bold_temp = bold_d[:, b * sub_num:(b + 1) * sub_num, :]
        bold_tempm = bold_temp.transpose(0, 1).contiguous().view(-1, 1200)
        for i in range(0, time_lengh):
            bold_fc = torch_corr(bold_tempm[:, i:i + window_size])
            cor_temp = bold_fc[fc_maskm]
            fc_mat[:, :, i] = torch.transpose(cor_temp.view(sub_num, fc_edgenum), 0, 1)

        for j in range(0, sub_num):
            fcd_temp = torch_corr(torch.transpose(fc_mat[:, j, :], 0, 1))
            fcd_hist_temp = np.histogram(fcd_temp[fcd_mask].cpu().numpy(), bins=10000, range=(-1., 1.))
            fcd_hist[:, j + b * sub_num] = fcd_hist_temp[0]

    if resid_num != 0:
        bold_temp = bold_d[:, batch_num * sub_num:n_set, :]
        bold_tempm = bold_temp.transpose(0, 1).contiguous().view(-1, 1200)
        for i in range(time_lengh):
            bold_fc = torch_corr(bold_tempm[:, i:i + window_size])
            cor_temp = bold_fc[fc_mask_resid]
            fc_resid[:, :, i] = torch.transpose(cor_temp.view(resid_num, fc_edgenum), 0, 1)

        for j in range(resid_num):
            fcd_temp = torch_corr(torch.transpose(fc_resid[:, j, :], 0, 1))
            fcd_hist_temp = np.histogram(fcd_temp[fcd_mask].cpu().numpy(), bins=10000, range=(-1., 1.))
            fcd_hist[:, j + sub_num * batch_num] = fcd_hist_temp[0]

    fcd_histcum = np.cumsum(fcd_hist, 0)

    # Calculating KS statistics cost
    time_dup = int(n_set / n_dup)
    ks_cost = np.zeros(time_dup)
    for t in range(time_dup):
        fcd_hist_temp = fcd_histcum[:, t * n_dup:(t + 1) * n_dup]
        fcd_histcum_nn = fcd_hist_temp[:, fcd_hist_temp[-1, :] == emp_ks[-1, 0]]
        fcd_hist_mean = np.mean(fcd_histcum_nn, 1)
        ks_cost[t] = np.max(np.abs(fcd_hist_mean - emp_ks[:, 0]) / emp_ks[-1, 0])

    fcd_elapsed = time.time() - fcd_timestart
    print('Time using for cost function: ', fcd_elapsed)
    return ks_cost


def torch_corr(A):
    '''
    Self implemented correlation function used for GPU
    '''

    Amean = torch.mean(A, 1)
    Ax = A - torch.transpose(Amean.repeat(A.shape[1], 1), 0, 1)
    Astd = torch.mean(Ax**2, 1)
    Amm = torch.mm(Ax, torch.transpose(Ax, 0, 1)) / A.shape[1]
    Aout = torch.sqrt(torch.ger(Astd, Astd))
    Acor = Amm / Aout
    return Acor


def torch_corr2(A, B):
    '''
    Self implemented correlation function used for GPU
    '''

    Amean = torch.mean(A, 1)
    Ax = A - torch.transpose(Amean.repeat(A.shape[1], 1), 0, 1)
    Astd = torch.mean(Ax**2, 1)
    Bmean = torch.mean(B, 1)
    Bx = B - torch.transpose(Bmean.repeat(B.shape[1], 1), 0, 1)
    Bstd = torch.mean(Bx**2, 1)
    numerator = torch.mm(Ax, torch.transpose(Bx, 0, 1)) / A.shape[1]
    denominator = torch.sqrt(torch.ger(Astd, Bstd))
    torch_cor = numerator / denominator
    return torch_cor


'''********************  Functions for computing FC & FCD costs   *************************'''


def CBIG_combined_cost_train(parameter, n_dup, FCD, SC, FC, countloop):
    '''
    This function is implemented to calcualted the FC correlation, L1 and FCD KS 
    statistics combined cost for input parameter sets based on training data
    Args:
        parameter:  (N*3+1)*M matrix. 
                    N is the number of ROI
                    M is the number of candidate parameter sets. 
                    Each column of matrix presents a parameter set, where:
                    parameter[0:N]: recurrent strength within excitatory population (wEE)
                    parameter[N:2*N]: connection strength from excitatory population to inhibitory population (wEI)
                    parameter[2*N]: Gloable constant G
                    parameter[2*N+1:3*N+1]: noise amplitude sigma
        n_dup:      number of times each parameter set is simulated
        FCD:        path to FCD
        SC:         path to SC
        FC:         path to FC
        countloop:  An integer used to keep track of which iteration CMAES is currently at
    
    Returns:
        total_cost: summation of FC correlation cost and FCD KS statistics cost
        fc_corr_cost:  
                    FC correlation cost
        fc_L1_cost:
                    FC L1 cost
        fcd_cost:   FCD KS statistics cost
        bold_d:     simulated BOLD signal
        r_E_all:    temporal average of excitatory firing rate
        emp_fc_mean:
                    mean of empirical FC (upper triangle)
    '''

    # Loading training data

    parameter = torch.from_numpy(parameter).type(torch.DoubleTensor)

    # should be 10000x1
    emp_fcd = sio.loadmat(FCD)
    emp_fcd = np.array(emp_fcd['group_level_FCD'])

    sc_mat_raw = csv_matrix_read(SC)
    sc_mat = sc_mat_raw * 0.02 / sc_mat_raw.max()
    sc_mat = torch.from_numpy(sc_mat).type(torch.DoubleTensor)

    emp_fc = csv_matrix_read(FC)
    emp_fc = torch.from_numpy(emp_fc).type(torch.DoubleTensor)

    if torch.cuda.is_available():
        parameter = parameter.cuda()
        sc_mat = sc_mat.cuda()
        emp_fc = emp_fc.cuda()
    # Calculating simualted BOLD signal using MFM
    bold_d, S_E_all, S_I_all, r_E_all, J_I = CBIG_mfm_multi_simulation(parameter, sc_mat, 14.4, n_dup, countloop, 0.006,
                                                                       0)

    # Calculating FC correlation cost
    fc_corr_cost, fc_L1_cost, emp_fc_mean = CBIG_FCcorrelation_multi_simulation(emp_fc, bold_d, n_dup)

    # Calculating FCD KS statistics cost
    fcd_cost = CBIG_FCDKSstat_multi_simulation(emp_fcd, bold_d, n_dup)

    # Calculating total cost
    total_cost = fc_corr_cost + fc_L1_cost + fcd_cost

    # keep the bold_d
    return total_cost, fc_corr_cost, fc_L1_cost, fcd_cost, bold_d, r_E_all, emp_fc_mean


def CBIG_combined_cost_validation(parameter, n_dup, FCD, SC, FC):
    '''
    This function is implemented to calcualted the FC correlation, L1 and FCD KS 
    statistics combined cost for input parameter sets based on validation data
    Args:
        parameter:  (N*3+1)*M matrix. 
                    N is the number of ROI
                    M is the number of candidate parameter sets. 
                    Each column of matrix presents a parameter set, where:
                    parameter[0:N]: recurrent strength within excitatory population (wEE)
                    parameter[N:2*N]: connection strength from excitatory population to inhibitory population (wEI)
                    parameter[2*N]: Gloable constant G
                    parameter[2*N+1:3*N+1]: noise amplitude sigma
        n_dup:      number of times each parameter set is simulated
        FCD:        path to FCD
        SC:         path to SC
        FC:         path to FC
        countloop:  An integer used to keep track of which iteration CMAES is currently at
    
    Returns:
        total_cost: summation of FC correlation cost and FCD KS statistics cost
        fc_corr_cost:  
                    FC correlation cost
        fc_L1_cost:
                    FC L1 cost
        fcd_cost:   FCD KS statistics cost
        bold_d:     simulated BOLD signal
        r_E_all:    temporal average of excitatory firing rate
    '''

    # Loading validation data
    parameter = torch.from_numpy(parameter).type(torch.DoubleTensor).cuda()

    emp_fcd = sio.loadmat(FCD)
    emp_fcd = np.array(emp_fcd['FCD_validation'])

    sc_mat_raw = csv_matrix_read(SC)
    sc_mat = sc_mat_raw * 0.02 / sc_mat_raw.max()
    sc_mat = torch.from_numpy(sc_mat).type(torch.DoubleTensor).cuda()

    emp_fc = csv_matrix_read(FC)
    emp_fc = torch.from_numpy(emp_fc).type(torch.DoubleTensor).cuda()

    # Calculating simualted BOLD signal using MFM
    bold_d, S_E_all, S_I_all, r_E_all, J_I = CBIG_mfm_multi_simulation(parameter, sc_mat, 14.4, n_dup, 1, 0.0005,
                                                                       1)  # the first '1' here is just a placeholder

    # Calculating FC correlation cost
    fc_cost, fc_L1_cost, _ = CBIG_FCcorrelation_multi_simulation(emp_fc, bold_d, n_dup)

    # Calculating FCD KS statistics cost
    fcd_cost = CBIG_FCDKSstat_multi_simulation(emp_fcd, bold_d, n_dup)

    # Calculating total cost
    total_cost = fc_cost + fc_L1_cost + fcd_cost

    return total_cost, fc_cost, fc_L1_cost, fcd_cost, bold_d, S_E_all, S_I_all, r_E_all


def CBIG_combined_cost_test(parameter, n_dup, FCD, SC, FC):
    '''
    This function is implemented to calcualted the FC correlation, L1 and FCD KS 
    statistics combined cost for input parameter sets based on test data
    Args:
        parameter:  (N*3+1)*M matrix. 
                    N is the number of ROI
                    M is the number of candidate parameter sets. 
                    Each column of matrix presents a parameter set, where:
                    parameter[0:N]: recurrent strength within excitatory population (wEE)
                    parameter[N:2*N]: connection strength from excitatory population to inhibitory population (wEI)
                    parameter[2*N]: Gloable constant G
                    parameter[2*N+1:3*N+1]: noise amplitude sigma
        n_dup:      number of times each parameter set is simulated
        FCD:        path to FCD
        SC:         path to SC
        FC:         path to FC
        countloop:  An integer used to keep track of which iteration CMAES is currently at
    
    Returns:
        total_cost: summation of FC correlation cost and FCD KS statistics cost
        fc_corr_cost:  
                    FC correlation cost
        fc_L1_cost:
                    FC L1 cost
        fcd_cost:   FCD KS statistics cost
        bold_d:     simulated BOLD signal
        r_E_all:    temporal average of excitatory firing rate
        J_I:        feedback inhibition control (FIC) from inhibitory to excitatory population
    '''

    # Loading test data
    parameter = np.tile(parameter, [1, n_dup])
    parameter = torch.from_numpy(parameter).type(torch.DoubleTensor).cuda()

    emp_fcd = sio.loadmat(FCD)
    emp_fcd = np.array(emp_fcd['FCD_test'])

    sc_mat_raw = csv_matrix_read(SC)
    sc_mat = sc_mat_raw * 0.02 / sc_mat_raw.max()
    sc_mat = torch.from_numpy(sc_mat).type(torch.DoubleTensor).cuda()

    emp_fc = csv_matrix_read(FC)
    emp_fc = torch.from_numpy(emp_fc).type(torch.DoubleTensor).cuda()

    # Calculating simualted BOLD signal using MFM
    bold_d, S_E_all, S_I_all, r_E_all, J_I = CBIG_mfm_single_simulation(parameter, sc_mat, 14.4)

    # Calculating FC correlation cost
    fc_corr_cost, fc_L1_cost = CBIG_FCcorrelation_single_simulation(emp_fc, bold_d, n_dup)

    # Calculating FCD KS statistics cost
    fcd_cost = CBIG_FCDKSstat_single_simulation(emp_fcd, bold_d, n_dup)

    # Calculating total cost
    total_cost = fc_corr_cost + fc_L1_cost + fcd_cost

    return total_cost, fc_corr_cost, fc_L1_cost, fcd_cost, bold_d, S_E_all, S_I_all, r_E_all, J_I


def csv_matrix_read(filename):
    '''
    Convert a .csv file to numpy array
    Args:
        filename:   input path of a .csv file
    Returns:
        out_array:  a numpy array
    '''
    csv_file = open(filename, "r")
    read_handle = csv.reader(csv_file)
    out_list = []
    R = 0
    for row in read_handle:
        out_list.append([])
        for col in row:
            out_list[R].append(float(col))
        R = R + 1
    out_array = np.array(out_list)
    csv_file.close()
    return out_array


##############################
# NEW FUNCTIONS ADDED by Tian Fang
##############################


def bold2fc(bold_d, n_dup):
    """
    Convert BOLD signal to FC
    """

    # Calculate vectored simulated FC
    n_set = bold_d.shape[1]
    n_nodes = bold_d.shape[0]
    n_num = int(n_set / n_dup)
    fc_mask = torch.triu(torch.ones(n_nodes, n_nodes), 1) == 1
    vect_len = int(n_nodes * (n_nodes - 1) / 2)
    sim_fc_vector = torch.zeros(n_set, vect_len)
    for i in range(n_set):
        sim_fc = torch_corr(bold_d[:, i, :])
        sim_fc_vector[i, :] = sim_fc[fc_mask]

    # Average the simulated FCs with same parameter set
    sim_fc_vector[sim_fc_vector != sim_fc_vector] = 0
    sim_fc_num = torch.zeros(n_num, vect_len)
    sim_fc_den = torch.zeros(n_num, 1)
    for k in range(n_dup):
        sim_fc_num = sim_fc_num + sim_fc_vector[k * n_num:(k + 1) * n_num, :]
        sim_fc_den = sim_fc_den + (sim_fc_vector[k * n_num:(k + 1) * n_num, 0:1] != 0).float()

    sim_fc_den[sim_fc_den == 0] = np.nan
    sim_fc_ave = sim_fc_num / sim_fc_den
    sim_fc_ave = sim_fc_ave.cpu().numpy()
    return sim_fc_ave


def get_fc_corr_between(fc1, fc2, n_num=1, use_corr_cost=True):
    if isinstance(fc1, np.ndarray):
        fc1 = torch.from_numpy(fc1).type(torch.DoubleTensor)
    if isinstance(fc2, np.ndarray):
        fc2 = torch.from_numpy(fc2).type(torch.DoubleTensor)

    if fc1.ndim == 1:
        fc1 = fc1.unsqueeze(0)
    if fc2.ndim == 1:
        fc2 = fc2.unsqueeze(0)

    # Calculate FC correlation
    corr_mass = torch_corr2(fc1, fc2)
    corr_cost = torch.diag(corr_mass)
    corr_cost = corr_cost.cpu().numpy()
    if use_corr_cost:
        corr_cost = 1 - corr_cost
    corr_cost[np.isnan(corr_cost)] = 10

    # L1 cost
    L1_cost = torch.abs(torch.mean(fc1, 1) - torch.mean(fc2, 1))
    L1_cost = L1_cost.cpu().numpy()
    L1_cost[np.isnan(L1_cost)] = 5

    return 1 * corr_cost, 1 * L1_cost


def bold2fcd(bold_d, n_dup):
    """
    Convert BOLD signal to FCD CDF
    """
    # Initializing the FC and FCD masks
    n_set = bold_d.shape[1]
    n_nodes = bold_d.shape[0]
    window_size = 83
    time_lengh = 1200 - window_size + 1
    sub_num = 10
    resid_num = n_set % sub_num
    fc_edgenum = int(n_nodes * (n_nodes - 1) / 2)
    fc_mask = torch.triu(torch.ones(n_nodes, n_nodes), 1) == 1
    fc_maskm = torch.zeros(n_nodes * sub_num, n_nodes * sub_num)
    if torch.cuda.is_available():
        fc_maskm = fc_maskm.type(torch.cuda.ByteTensor)
    else:
        fc_maskm = fc_maskm.type(torch.ByteTensor)

    for i in range(sub_num):
        fc_maskm[n_nodes * i:n_nodes * (i + 1), n_nodes * i:n_nodes * (i + 1)] = fc_mask

    fc_mask_resid = torch.zeros(n_nodes * resid_num, n_nodes * resid_num)
    if torch.cuda.is_available():
        fc_mask_resid = fc_mask_resid.type(torch.cuda.ByteTensor)
    else:
        fc_mask_resid = fc_mask_resid.type(torch.ByteTensor)
    for i in range(resid_num):
        fc_mask_resid[n_nodes * i:n_nodes * (i + 1), n_nodes * i:n_nodes * (i + 1)] = fc_mask

    fcd_mask = torch.triu(torch.ones(time_lengh, time_lengh), 1) == 1

    # Calculating CDF for simualted FCD matrices
    fcd_hist = np.ones([10000, n_set])
    fc_mat = torch.zeros(fc_edgenum, sub_num, time_lengh)
    batch_num = math.floor(n_set / sub_num)
    fc_resid = torch.zeros(fc_edgenum, resid_num, time_lengh)

    for b in range(batch_num):
        bold_temp = bold_d[:, b * sub_num:(b + 1) * sub_num, :]
        bold_tempm = bold_temp.transpose(0, 1).contiguous().view(-1, 1200)
        for i in range(0, time_lengh):
            bold_fc = torch_corr(bold_tempm[:, i:i + window_size])
            cor_temp = bold_fc[fc_maskm]
            fc_mat[:, :, i] = torch.transpose(cor_temp.view(sub_num, fc_edgenum), 0, 1)

        for j in range(0, sub_num):
            fcd_temp = torch_corr(torch.transpose(fc_mat[:, j, :], 0, 1))
            fcd_hist_temp = np.histogram(fcd_temp[fcd_mask].cpu().numpy(), bins=10000, range=(-1., 1.))
            fcd_hist[:, j + b * sub_num] = fcd_hist_temp[0]

    if resid_num != 0:
        bold_temp = bold_d[:, batch_num * sub_num:n_set, :]
        bold_tempm = bold_temp.transpose(0, 1).contiguous().view(-1, 1200)
        for i in range(time_lengh):
            bold_fc = torch_corr(bold_tempm[:, i:i + window_size])
            cor_temp = bold_fc[fc_mask_resid]
            fc_resid[:, :, i] = torch.transpose(cor_temp.view(resid_num, fc_edgenum), 0, 1)

        for j in range(resid_num):
            fcd_temp = torch_corr(torch.transpose(fc_resid[:, j, :], 0, 1))
            fcd_hist_temp = np.histogram(fcd_temp[fcd_mask].cpu().numpy(), bins=10000, range=(-1., 1.))
            fcd_hist[:, j + sub_num * batch_num] = fcd_hist_temp[0]

    fcd_histcum = np.cumsum(fcd_hist, 0)
    return fcd_histcum


def get_ks_cost_between(fcd1, fcd2, n_num=1, n_dup=5):
    """
    Calculating KS statistics cost between two FCD cdf
    """
    fcd_histcum_temp = np.zeros((10000, n_num))
    fcd_histcum_num = np.zeros((1, n_num))
    for k in range(n_dup):
        fcd_histcum_temp = fcd_histcum_temp + fcd1[:, k * n_num:(k + 1) * n_num]
        fcd_histcum_num = fcd_histcum_num + (fcd1[-1, k * n_num:(k + 1) * n_num] == fcd2[-1, 0])
    fcd_histcum_ave = fcd_histcum_temp / fcd_histcum_num
    ks_diff = np.abs(fcd_histcum_ave - np.tile(fcd2, [1, n_num]))
    ks_cost = ks_diff.max(0) / fcd2[-1, 0]
    # ks_cost[fcd_histcum_ave[-1, :] != fcd2[-1, 0]] = 10
    ks_cost = np.mean(ks_cost)
    return ks_cost


def get_param_coef(myelin_data, gradient_data, param):
    '''
    This function is implemented to calculate the initial parametrized coefficients
    '''

    n_node = myelin_data.shape[0]
    concat_matrix = np.vstack((np.ones(n_node), myelin_data.T, gradient_data.T)).T  # bias, myelin PC, RSFC gradient PC
    param_coef = np.linalg.inv(concat_matrix.T @ concat_matrix) @ concat_matrix.T @ param
    return param_coef, concat_matrix


def load_group_myelin(split_name, group_index):
    myelin_path = os.path.join('./input_to_pMFM/', split_name, group_index, 'group_level_myelin.csv')
    myelin = pd.read_csv(myelin_path, header=None)
    myelin = myelin.to_numpy()
    return myelin


def load_group_RSFC_gradient(split_name, group_index):
    RSFC_gradient_path = os.path.join('./input_to_pMFM/', split_name, group_index, 'group_level_RSFC_gradient.csv')
    RSFC_gradient = pd.read_csv(RSFC_gradient_path, header=None)
    RSFC_gradient = RSFC_gradient.to_numpy()
    return RSFC_gradient


def load_group_params(split_name, group_index):
    path_to_group_param = os.path.join('./dataset/', split_name, group_index, f'{split_name}_{group_index}.csv')
    params = pd.read_csv(path_to_group_param, header=None)
    params = params.to_numpy()
    return params


def save_coef(coef, split_name, group_index):
    file_name = os.path.join('./dataset/', split_name, group_index, f'{split_name}_{group_index}_coef.csv')
    np.savetxt(file_name, coef, delimiter=',')


def convert_param_to_coef(split_name, group_index):
    print(f'converting {split_name} {group_index} params to coefficients')
    group_index = str(group_index)

    params = load_group_params(split_name, group_index)

    n_param = params.shape[1]
    param_coef = np.zeros((14, n_param))

    group_myelin = load_group_myelin(split_name, group_index)
    group_RSFC_gradient = load_group_RSFC_gradient(split_name, group_index)
    n_node = group_myelin.shape[0]

    for i in range(n_param):
        param_with_cost = params[:, i]
        cost_vector = param_with_cost[:4]
        param = param_with_cost[4:]
        wEE_coef, _ = get_param_coef(group_myelin, group_RSFC_gradient, param[:n_node])
        wEI_coef, _ = get_param_coef(group_myelin, group_RSFC_gradient, param[n_node:2 * n_node])
        sigma_coef, _ = get_param_coef(group_myelin, group_RSFC_gradient, param[2 * n_node + 1:])
        G = np.array([param[2 * n_node]])
        # print(np.concatenate((cost_vector, wEE_coef, wEI_coef, sigma_coef, G)).shape)
        param_coef[:, i] = np.concatenate((cost_vector, wEE_coef, wEI_coef, sigma_coef, G))

    save_coef(param_coef, split_name, group_index)
    return param_coef