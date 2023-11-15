from pathlib import Path
from seaborn import histplot
from matplotlib import pyplot as plt
import numpy as np
import time
import torch
import os
import sys
import csv
import pandas as pd
from torch import optim

sys.path.insert(1, '/home/ftian/storage/pMFM_speedup/')
from src.basic.constants import NUM_REGION  # noqa: E402
from src.testing.testing_lib import MANUAL_SEED, load_split, load_naive_net_no_SC, PATH_TO_TESTING_REPORT, df_to_tensor, get_path_to_group, forward_simulation, all_costs_prediction_vs_actual_cost, pred_and_actual_cost_corr_dist  # noqa: E501, F401, E402
from src.utils.CBIG_pMFM_basic_functions_HCP import CBIG_combined_cost_train  # noqa: E402

################################################################
# CMA-ES Wrapper using pMFM
################################################################


def CBIG_mfm_optimization_desikan_main(path_to_group, random_seed, input_param):
    '''
    This function is to implement the optimization processes of mean field model.
    The objective function is the summation of FC correlation cost and FCD KS statistics cost.
    The optimization process is highly automatic and generate 500 candidate parameter sets for
    main results.

    Args:
        output_path:        output path
        output_file:        output file name
        path_to_group:      path to the folder corresponding to the group, which contains inputs such as FC, SC, etc.
        random_seed:        random seed
    Returns:
        None
    '''
    FCD = path_to_group + 'group_level_FCD.mat'
    SC = path_to_group + 'group_level_SC.csv'
    FC = path_to_group + 'group_level_FC.csv'

    # torch.cuda.set_device(gpu_number)

    # Setting random seed and GPU
    random_seed_cuda = random_seed
    random_seed_np = random_seed
    torch.manual_seed(random_seed_cuda)
    rng = np.random.Generator(np.random.PCG64(random_seed_np))

    # Initializing input parameters
    myelin_data = csv_matrix_read(path_to_group + 'group_level_myelin.csv')
    num_myelin_component = myelin_data.shape[1]

    gradient_data = csv_matrix_read(path_to_group + 'group_level_RSFC_gradient.csv')
    num_gradient_component = gradient_data.shape[1]
    N = 3 * (num_myelin_component + num_gradient_component + 1) + 1  # number of parameterized parameters 10
    N_p = num_myelin_component + num_gradient_component + 1  # nunber of parameterized parameter associated to each parameter 3
    n_node = myelin_data.shape[0]  # 68
    dim = n_node * 3 + 1

    wEE_min, wEE_max, wEI_min, wEI_max = 1, 10, 1, 5
    search_range = np.zeros((dim, 2))
    search_range[0:n_node, :] = [wEE_min, wEE_max]  # search range for w_EE
    search_range[n_node:n_node * 2, :] = [wEI_min, wEI_max]  # search range for w_EI
    search_range[n_node * 2, :] = [0, 3]  # search range for G
    search_range[n_node * 2 + 1:dim, :] = [0.0005, 0.01]  # search range for sigma
    init_para = rng.uniform(0, 1, dim) * (search_range[:, 1] - search_range[:, 0]) + search_range[:, 0]
    start_point_w_EE, template_mat = get_init(myelin_data, gradient_data, num_myelin_component, num_gradient_component,
                                              init_para[0:n_node])
    start_point_w_EI, template_mat = get_init(myelin_data, gradient_data, num_myelin_component, num_gradient_component,
                                              init_para[n_node:n_node * 2])
    start_point_sigma, template_mat = get_init(myelin_data, gradient_data, num_myelin_component, num_gradient_component,
                                               init_para[n_node * 2 + 1:dim])

    # Initializing childrens
    xmean = np.zeros(N)  # size 1 x N
    xmean[0:N_p] = start_point_w_EE
    xmean[1] = start_point_w_EE[1] / 2
    xmean[N_p:2 * N_p] = start_point_w_EI
    xmean[4] = start_point_w_EI[1] / 2
    xmean[2 * N_p] = init_para[2 * n_node]  # G
    xmean[2 * N_p + 1:N] = start_point_sigma

    # Initializing optimization hyper-parameters
    sigma = 0.25  # 'spread' of children
    maxloop = 100
    n_dup = 5  # duplication for pMFM simulation

    # lambda * maxloop * num_of_initialization = number of iterations
    # 36 hours for lambda = 100, max_loop = 100, num_of_initialization = 5
    # we use: lambda = 100, max_loop = 100, num_of_initialization = 1
    # CMA-ES parameters setting
    Lambda = 100  # number of child processes
    mu = 10  # number of top child processes
    all_parameters = np.zeros((209, maxloop * Lambda))  # numpy array to store all parameters and their associated costs

    weights = np.log(mu + 1 / 2) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = np.sum(weights)**2 / np.sum(weights**2)

    # Strategy parameter setting: adaptation
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3)**2 + mueff)
    cmu = np.minimum(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2)**2 + mueff))
    damps = 1 + 2 * np.maximum(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs

    # Initializing dynamic strategy parameters and constants'''
    pc = np.zeros(N)
    ps = np.zeros(N)
    B = np.eye(N)
    D = np.ones(N)
    D[0:N_p] = start_point_w_EE[0] / 2
    D[N_p:2 * N_p] = start_point_w_EI[0] / 2
    D[2 * N_p] = 0.4
    D[2 * N_p + 1:N] = 0.001 / 2

    C = np.dot(np.dot(B, np.diag(np.power(D, 2))), B.T)
    invsqrtC = np.dot(np.dot(B, np.diag(np.power(D, -1))), B.T)
    chiN = N**0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ^ 2))

    # Evolution loop
    countloop = 0
    arx = np.zeros([N, Lambda])
    input_para = np.zeros((dim, Lambda))
    xmin = np.zeros(N + 4)
    while countloop < maxloop:

        start_time = time.time()

        # Generating lambda offspring
        arx[:, 0] = xmean
        j = 0
        infinite_loop_count = 0

        while j < Lambda:
            arx[:, j] = xmean + sigma * np.dot(B, (D * rng.standard_normal(N)))
            input_para[0:n_node, j] = template_mat @ arx[0:N_p, j]
            input_para[n_node:2 * n_node, j] = template_mat @ arx[N_p:2 * N_p, j]
            input_para[2 * n_node:2 * n_node + 1, j] = arx[2 * N_p, j]
            input_para[2 * n_node + 1:dim, j] = template_mat @ arx[2 * N_p + 1:N, j]

            if (input_para[:, j] < search_range[:, 0]).any() or (input_para[:, j] > search_range[:, 1]).any():
                j = j - 1
                infinite_loop_count += 1
                if infinite_loop_count > 20000:
                    print(str(countloop) + ' Infinite Loop')
                    return
            j = j + 1

        # Calculating costs of offspring
        print("Forward simulation starts ...")
        total_cost, fc_corr_cost, fc_L1_cost, fcd_cost, bold_d, r_E, emp_fc_mean = CBIG_combined_cost_train(
            input_para, n_dup, FCD, SC, FC, countloop)
        # print(input_para.shape, total_cost.shape, fc_corr_cost.shape,
        #       fc_L1_cost.shape, fcd_cost.shape, bold_d.shape, r_E.shape)
        # (205, 10) (10,) (10,) (10,) (10,) torch.Size([68, 50, 1200]) torch.Size([68, 50]) when Lambda = 10

        # Storing all parameters and their associated costs
        all_parameters[4:, countloop * Lambda:(countloop + 1) * Lambda] = input_para
        all_parameters[0, countloop * Lambda:(countloop + 1) * Lambda] = fc_corr_cost
        all_parameters[1, countloop * Lambda:(countloop + 1) * Lambda] = fc_L1_cost
        all_parameters[2, countloop * Lambda:(countloop + 1) * Lambda] = fcd_cost
        all_parameters[3, countloop * Lambda:(countloop + 1) * Lambda] = total_cost

        countloop = countloop + 1

        # Sort by total cost and compute weighted mean
        arfitsort = np.sort(total_cost)
        arindex = np.argsort(total_cost)
        xold = xmean
        xmean = np.dot(arx[:, arindex[0:mu]], weights)
        xshow = xmean - xold

        # Cumulation
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(invsqrtC, xshow) / sigma
        hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * countloop)) / chiN < (1.4 + 2 / (N + 1))) * 1
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * xshow / sigma

        # Adapting covariance matrix C
        artmp = (1 / sigma) * (arx[:, arindex[0:mu]] - np.tile(xold, [mu, 1]).T)
        C = (1-c1-cmu)*C+c1*(np.outer(pc, pc)+(1-hsig)*cc*(2-cc)*C) + \
            cmu*np.dot(artmp, np.dot(np.diag(weights), artmp.T))

        # Adapting step size
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        # Decomposition
        if 1 > 1 / (c1 + cmu) / N / 10:
            C = np.triu(C, k=1) + np.triu(C).T
            D, B = np.linalg.eigh(C)
            D = D.real
            B = B.real
            D = np.sqrt(D)
            invsqrtC = np.dot(B, np.dot(np.diag(D**(-1)), B.T))

        # Monitoring the evolution status
        print('******** Generation: ' + str(countloop) + ' ********')
        print('The mean of total cost: ', np.mean(arfitsort[0:mu]))

        xmin[0:N] = arx[:, arindex[0]]
        xmin[N] = fc_corr_cost[arindex[0]]
        xmin[N + 1] = fc_L1_cost[arindex[0]]
        xmin[N + 2] = fcd_cost[arindex[0]]
        xmin[N + 3] = np.min(total_cost)
        print('Best parameter set: ', arindex[0])
        print('Best total cost: ', np.min(total_cost))
        print('FC correlation cost: ', fc_corr_cost[arindex[0]])
        print('FC L1 cost: ', fc_L1_cost[arindex[0]])
        print('FCD KS statistics cost: ', fcd_cost[arindex[0]])
        print('wEI search range: ' + str(wEI_min) + ', ' + str(wEI_max))
        print('wEE search range: ' + str(wEE_min) + ', ' + str(wEE_max))

        elapsed_time = time.time() - start_time
        print('Elapsed time for this evolution is : ', elapsed_time)
        print('******************************************')
    print(str(countloop) + ' Success!')
    return all_parameters[:, -Lambda:]


################################################################
# CMA-ES Wrapper using DL model
################################################################
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


def get_init(myelin_data, gradient_data, init_para):
    '''
    This function is implemented to calculate the initial parametrized coefficients
    '''

    n_node = myelin_data.shape[0]
    concat_matrix = np.vstack((np.ones(n_node), myelin_data.T, gradient_data.T)).T  # bias, myelin PC, RSFC gradient PC
    para = np.linalg.inv(concat_matrix.T @ concat_matrix) @ concat_matrix.T @ init_para
    return para, concat_matrix


def predict_costs(input_para, SC=None, model=load_naive_net_no_SC()):
    '''
    This function is implemented to predict the costs of input parameters using the deep learning model given
    '''
    n_param = input_para.shape[1]  # number of parameters

    batched_param = torch.from_numpy(input_para).type(torch.FloatTensor).transpose(0, 1).to(model.device)
    batched_input = batched_param
    if SC is not None:
        SC = SC.to(model.device)
        batched_SC = SC.type(torch.FloatTensor).unsqueeze(0).repeat((n_param, 1, 1))
        batched_input = (batched_SC, batched_param)

    with torch.no_grad():
        model.eval()
        y_pred = model(batched_input)

    y_pred = y_pred.cpu().numpy()
    return y_pred.sum(1), y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    # prediction = torch.concat((batched_param, y_pred), 1).transpose(0, 1).cpu().numpy()


def save_top_k_param(all_param, file_path, k=1000):
    total_cost = all_param[3, :]
    sorted_indicies = np.argsort(total_cost)
    top_k_param = all_param[:, sorted_indicies[:k]]
    np.savetxt(file_path, top_k_param, delimiter=',')


def cmaes_wrapper_use_coef(myelin_path,
                           gradient_path,
                           SC_path=None,
                           random_seed=MANUAL_SEED,
                           dl_model=load_naive_net_no_SC(),
                           output_dir=os.path.join(PATH_TO_TESTING_REPORT, 'cmaes_wrapper/predicted_best'),
                           output_file='predicted_best.csv'):

    output_dir = os.path.join(output_dir, f'seed_{random_seed}')
    if SC_path is not None:
        SC = pd.read_csv(SC_path, header=None)
        SC = df_to_tensor(SC)
    else:
        SC = None
    # torch.cuda.set_device(gpu_number)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setting random seed and GPU
    random_seed_cuda = random_seed
    random_seed_np = random_seed
    torch.manual_seed(random_seed_cuda)
    rng = np.random.Generator(np.random.PCG64(random_seed_np))

    # Initializing input parameters
    myelin_data = csv_matrix_read(myelin_path)
    num_myelin_component = myelin_data.shape[1]

    gradient_data = csv_matrix_read(gradient_path)
    num_gradient_component = gradient_data.shape[1]
    N = 3 * (num_myelin_component + num_gradient_component + 1) + 1  # number of parameterized parameters 10
    N_p = num_myelin_component + num_gradient_component + 1  # nunber of parameterized parameter associated to each parameter 3
    n_node = NUM_REGION  # 68
    dim = n_node * 3 + 1  # 205

    wEE_min, wEE_max, wEI_min, wEI_max = 1, 10, 1, 5
    search_range = np.zeros((dim, 2))
    search_range[0:n_node, :] = [wEE_min, wEE_max]  # search range for w_EE
    search_range[n_node:n_node * 2, :] = [wEI_min, wEI_max]  # search range for w_EI
    search_range[n_node * 2, :] = [0, 3]  # search range for G
    search_range[n_node * 2 + 1:dim, :] = [0.0005, 0.01]  # search range for sigma
    init_para = rng.uniform(0, 1, dim) * (search_range[:, 1] - search_range[:, 0]) + search_range[:, 0]
    start_point_w_EE, template_mat = get_init(myelin_data, gradient_data, init_para[0:n_node])
    start_point_w_EI, template_mat = get_init(myelin_data, gradient_data, init_para[n_node:n_node * 2])
    start_point_sigma, template_mat = get_init(myelin_data, gradient_data, init_para[n_node * 2 + 1:dim])

    # Initializing childrens
    xmean = np.zeros(N)  # size 1 x N
    xmean[0:N_p] = start_point_w_EE.squeeze()
    xmean[1] = start_point_w_EE[1] / 2
    xmean[N_p:2 * N_p] = start_point_w_EI.squeeze()
    xmean[4] = start_point_w_EI[1] / 2
    xmean[2 * N_p] = init_para[2 * n_node]  # G
    xmean[2 * N_p + 1:N] = start_point_sigma.squeeze()

    # Initializing optimization hyper-parameters
    sigma = 0.25  # 'spread' of children
    maxloop = 500

    # lambda * maxloop * num_of_initialization = number of iterations
    # 36 hours for lambda = 100, max_loop = 100, num_of_initialization = 5
    # we use: lambda = 100, max_loop = 100, num_of_initialization = 1
    # CMA-ES parameters setting
    Lambda = 1000  # number of child processes
    mu = 100  # number of top child processes
    all_parameters = np.zeros((209, maxloop * Lambda))  # numpy array to store all parameters and their associated costs

    weights = np.log(mu + 1 / 2) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = np.sum(weights)**2 / np.sum(weights**2)

    # Strategy parameter setting: adaptation
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3)**2 + mueff)
    cmu = np.minimum(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2)**2 + mueff))
    damps = 1 + 2 * np.maximum(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs

    # Initializing dynamic strategy parameters and constants'''
    pc = np.zeros(N)
    ps = np.zeros(N)
    B = np.eye(N)
    D = np.ones(N)
    D[0:N_p] = start_point_w_EE[0] / 2
    D[N_p:2 * N_p] = start_point_w_EI[0] / 2
    D[2 * N_p] = 0.4
    D[2 * N_p + 1:N] = 0.001 / 2

    C = np.dot(np.dot(B, np.diag(np.power(D, 2))), B.T)
    invsqrtC = np.dot(np.dot(B, np.diag(np.power(D, -1))), B.T)
    chiN = N**0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ^ 2))

    # Evolution loop
    countloop = 0
    arx = np.zeros([N, Lambda])  # all children parameters
    input_para = np.zeros((dim, Lambda))
    xmin = np.zeros(N + 4)

    prev_mean_total_cost = np.inf
    curr_mean_total_cost = 30
    curr_patience = 0
    stopping_threshold = 0.000001
    patience = 3
    while countloop < maxloop and ((prev_mean_total_cost - curr_mean_total_cost > stopping_threshold) or
                                   (curr_patience <= patience)):
        if prev_mean_total_cost - curr_mean_total_cost > stopping_threshold:
            curr_patience = 0
        else:
            curr_patience += 1
        iteration_log = open(os.path.join(output_dir, 'training_iteration.txt'), 'w')
        iteration_log.write(str(countloop))

        start_time = time.time()

        # Generating lambda offspring
        arx[:, 0] = xmean
        j = 0
        infinite_loop_count = 0

        while j < Lambda:
            arx[:, j] = xmean + sigma * np.dot(B, (D * rng.standard_normal(N)))

            input_para[0:n_node, j] = template_mat @ arx[0:N_p, j]
            input_para[n_node:2 * n_node, j] = template_mat @ arx[N_p:2 * N_p, j]
            input_para[2 * n_node:2 * n_node + 1, j] = arx[2 * N_p, j]
            input_para[2 * n_node + 1:dim, j] = template_mat @ arx[2 * N_p + 1:N, j]

            if (input_para[:, j] < search_range[:, 0]).any() or (input_para[:, j] > search_range[:, 1]).any():
                j = j - 1
                infinite_loop_count += 1
                if infinite_loop_count > 20000:
                    iteration_log.write(str(countloop) + ' Infinite Loop')
                    iteration_log.close()
                    return
            j = j + 1

        # Calculating costs of offspring
        print("Predicting using deep learning model ...")
        total_cost, fc_corr_cost, fc_L1_cost, fcd_cost = predict_costs(input_para, SC, model=dl_model)
        # print(input_para.shape, total_cost.shape, fc_corr_cost.shape,
        #       fc_L1_cost.shape, fcd_cost.shape, bold_d.shape, r_E.shape)
        # (205, 10) (10,) (10,) (10,) (10,) torch.Size([68, 50, 1200]) torch.Size([68, 50]) when Lambda = 10

        # Storing all parameters and their associated costs
        all_parameters[4:, countloop * Lambda:(countloop + 1) * Lambda] = input_para
        all_parameters[0, countloop * Lambda:(countloop + 1) * Lambda] = fc_corr_cost
        all_parameters[1, countloop * Lambda:(countloop + 1) * Lambda] = fc_L1_cost
        all_parameters[2, countloop * Lambda:(countloop + 1) * Lambda] = fcd_cost
        all_parameters[3, countloop * Lambda:(countloop + 1) * Lambda] = total_cost

        countloop = countloop + 1

        # Sort by total cost and compute weighted mean
        arfitsort = np.sort(total_cost)
        arindex = np.argsort(total_cost)
        xold = xmean
        xmean = np.dot(arx[:, arindex[0:mu]], weights)
        xshow = xmean - xold

        # Cumulation
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(invsqrtC, xshow) / sigma
        hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * countloop)) / chiN < (1.4 + 2 / (N + 1))) * 1
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * xshow / sigma

        # Adapting covariance matrix C
        artmp = (1 / sigma) * (arx[:, arindex[0:mu]] - np.tile(xold, [mu, 1]).T)
        C = (1-c1-cmu)*C+c1*(np.outer(pc, pc)+(1-hsig)*cc*(2-cc)*C) + \
            cmu*np.dot(artmp, np.dot(np.diag(weights), artmp.T))

        # Adapting step size
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        # Decomposition
        if 1 > 1 / (c1 + cmu) / N / 10:
            C = np.triu(C, k=1) + np.triu(C).T
            D, B = np.linalg.eigh(C)
            D = D.real
            B = B.real
            D = np.sqrt(D)
            invsqrtC = np.dot(B, np.dot(np.diag(D**(-1)), B.T))

        # Monitoring the evolution status
        cost_log = open(os.path.join(output_dir, f'cost_with_seed_{str(random_seed)}.txt'), 'w')
        print('******** Generation: ' + str(countloop) + ' ********')
        cost_log.write('******** Generation: ' + str(countloop) + ' ********' + '\n')
        prev_mean_total_cost = curr_mean_total_cost
        curr_mean_total_cost = np.mean(arfitsort[0:mu])
        print('The mean of total cost: ', curr_mean_total_cost)

        xmin[0:N] = arx[:, arindex[0]]
        xmin[N] = fc_corr_cost[arindex[0]]
        xmin[N + 1] = fc_L1_cost[arindex[0]]
        xmin[N + 2] = fcd_cost[arindex[0]]
        xmin[N + 3] = np.min(total_cost)
        cost_log.write('sigma: ' + str(sigma) + '\n')
        print('Best parameter set: ', arindex[0])
        cost_log.write('Best parameter set: ' + str(arindex[0]) + '\n')
        best_total_cost = np.min(total_cost)
        print('Best total cost: ', best_total_cost)
        cost_log.write('Best total cost: ' + str(best_total_cost) + '\n')
        print('FC correlation cost: ', fc_corr_cost[arindex[0]])
        cost_log.write('FC correlation cost: ' + str(fc_corr_cost[arindex[0]]) + '\n')
        print('FC L1 cost: ', fc_L1_cost[arindex[0]])
        cost_log.write('FC L1 cost: ' + str(fc_L1_cost[arindex[0]]) + '\n')
        print('FCD KS statistics cost: ', fcd_cost[arindex[0]])
        cost_log.write('FCD KS statistics cost: ' + str(fcd_cost[arindex[0]]) + '\n')
        print('wEI search range: ' + str(wEI_min) + ', ' + str(wEI_max))
        print('wEE search range: ' + str(wEE_min) + ', ' + str(wEE_max))
        cost_log.write('wEE search range: ' + str(wEE_min) + ', ' + str(wEE_max) + '\n')
        cost_log.write('wEI search range: ' + str(wEI_min) + ', ' + str(wEI_max) + '\n')

        elapsed_time = time.time() - start_time
        print('Elapsed time for this evolution is : ', elapsed_time)
        cost_log.write('Elapsed time for this evolution is : ' + str(elapsed_time) + '\n')
        print('******************************************')
        cost_log.write('******************************************')
        cost_log.write("\n")

    all_parameters = all_parameters[:, np.any(all_parameters, axis=0)]
    save_top_k_param(all_parameters, os.path.join(output_dir, output_file), k=1)

    iteration_log.write(str(countloop) + ' Success!')
    iteration_log.close()
    cost_log.close()


def cmaes_wrapper(random_seed=MANUAL_SEED,
                  dl_model=load_naive_net_no_SC(),
                  output_dir=os.path.join(PATH_TO_TESTING_REPORT, 'cmaes_wrapper/predicted_best'),
                  output_file='predicted_best.csv'):

    output_dir = os.path.join(output_dir, f'seed_{random_seed}')

    # torch.cuda.set_device(gpu_number)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setting random seed and GPU
    random_seed_cuda = random_seed
    random_seed_np = random_seed
    torch.manual_seed(random_seed_cuda)
    rng = np.random.Generator(np.random.PCG64(random_seed_np))

    # Basic variables
    n_node = 68  # 68
    N = 3 * n_node + 1  # number of parameters 205
    # Note: change N and N_p to 205
    dim = n_node * 3 + 1  # 205

    wEE_min, wEE_max, wEI_min, wEI_max = 1, 10, 1, 5
    search_range = np.zeros((dim, 2))
    search_range[0:n_node, :] = [wEE_min, wEE_max]  # search range for w_EE
    search_range[n_node:n_node * 2, :] = [wEI_min, wEI_max]  # search range for w_EI
    search_range[n_node * 2, :] = [0, 3]  # search range for G
    search_range[n_node * 2 + 1:dim, :] = [0.0005, 0.01]  # search range for sigma
    init_para = rng.uniform(0, 1, dim) * (search_range[:, 1] - search_range[:, 0]) + search_range[:, 0]

    # Initializing childrens
    # Note: N is 205 now, just use the uniform distribution
    xmean = init_para

    # Initializing optimization hyper-parameters
    sigma = 0.25  # 'spread' of children
    maxloop = 500

    # lambda * maxloop * num_of_initialization = number of iterations
    # 36 hours for lambda = 100, max_loop = 100, num_of_initialization = 5
    # we use: lambda = 100, max_loop = 100, num_of_initialization = 1
    # CMA-ES parameters setting
    Lambda = 100  # number of child processes
    mu = 10  # number of top child processes
    all_parameters = np.zeros((209, maxloop * Lambda))  # numpy array to store all parameters and their associated costs

    weights = np.log(mu + 1 / 2) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = np.sum(weights)**2 / np.sum(weights**2)

    # Strategy parameter setting: adaptation
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3)**2 + mueff)
    cmu = np.minimum(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2)**2 + mueff))
    damps = 1 + 2 * np.maximum(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs

    # Initializing dynamic strategy parameters and constants'''
    pc = np.zeros(N)
    ps = np.zeros(N)
    B = np.eye(N)

    D = np.ones(N)  # Note: N is 205 now
    D = init_para / 10
    # D[2 * n_node + 1:] = 0.0005

    C = np.dot(np.dot(B, np.diag(np.power(D, 2))), B.T)
    invsqrtC = np.dot(np.dot(B, np.diag(np.power(D, -1))), B.T)
    chiN = N**0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ^ 2))

    # Evolution loop
    countloop = 0
    arx = np.zeros([N, Lambda])  # all children parameters
    input_para = np.zeros((dim, Lambda))
    xmin = np.zeros(N + 4)

    prev_mean_total_cost = np.inf
    curr_mean_total_cost = 30
    curr_patience = 0
    stopping_threshold = 0.000001
    patience = 3
    while countloop < maxloop and ((prev_mean_total_cost - curr_mean_total_cost > stopping_threshold) or
                                   (curr_patience <= patience)):
        # print(curr_patience)
        if prev_mean_total_cost - curr_mean_total_cost > stopping_threshold:
            curr_patience = 0
        else:
            curr_patience += 1
        iteration_log = open(os.path.join(output_dir, 'training_iteration.txt'), 'w')
        iteration_log.write(str(countloop))

        start_time = time.time()

        # Generating lambda offspring
        arx[:, 0] = xmean
        j = 0
        infinite_loop_count = 0
        # print('xmean', xmean)
        # print(sigma * np.dot(B, (D * rng.standard_normal(N))))
        while j < Lambda:

            arx[:, j] = xmean + sigma * np.dot(B, (D * rng.standard_normal(N)))

            input_para[:, j] = arx[:, j]

            if (input_para[:, j] < search_range[:, 0]).any() or (input_para[:, j] > search_range[:, 1]).any():
                j = j - 1
                infinite_loop_count += 1
                if infinite_loop_count > 20000:
                    iteration_log.write(str(countloop) + ' Infinite Loop')
                    iteration_log.close()
                    return
            j = j + 1

        # Calculating costs of offspring
        print("Predicting using deep learning model ...")
        total_cost, fc_corr_cost, fc_L1_cost, fcd_cost = predict_costs(input_para, None, model=dl_model)
        # print(input_para.shape, total_cost.shape, fc_corr_cost.shape,
        #       fc_L1_cost.shape, fcd_cost.shape, bold_d.shape, r_E.shape)
        # (205, 10) (10,) (10,) (10,) (10,) torch.Size([68, 50, 1200]) torch.Size([68, 50]) when Lambda = 10

        # Storing all parameters and their associated costs
        all_parameters[4:, countloop * Lambda:(countloop + 1) * Lambda] = input_para
        all_parameters[0, countloop * Lambda:(countloop + 1) * Lambda] = fc_corr_cost
        all_parameters[1, countloop * Lambda:(countloop + 1) * Lambda] = fc_L1_cost
        all_parameters[2, countloop * Lambda:(countloop + 1) * Lambda] = fcd_cost
        all_parameters[3, countloop * Lambda:(countloop + 1) * Lambda] = total_cost

        countloop = countloop + 1

        # Sort by total cost and compute weighted mean
        arfitsort = np.sort(total_cost)
        arindex = np.argsort(total_cost)
        xold = xmean
        xmean = np.dot(arx[:, arindex[0:mu]], weights)
        xshow = xmean - xold

        # Cumulation
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(invsqrtC, xshow) / sigma
        hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * countloop)) / chiN < (1.4 + 2 / (N + 1))) * 1
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * xshow / sigma

        # Adapting covariance matrix C
        artmp = (1 / sigma) * (arx[:, arindex[0:mu]] - np.tile(xold, [mu, 1]).T)
        C = (1-c1-cmu)*C+c1*(np.outer(pc, pc)+(1-hsig)*cc*(2-cc)*C) + \
            cmu*np.dot(artmp, np.dot(np.diag(weights), artmp.T))

        # Adapting step size
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        # Decomposition
        if 1 > 1 / (c1 + cmu) / N / 10:
            C = np.triu(C, k=1) + np.triu(C).T
            D, B = np.linalg.eigh(C)
            D = D.real
            B = B.real
            D = np.sqrt(D)
            invsqrtC = np.dot(B, np.dot(np.diag(D**(-1)), B.T))

        # Monitoring the evolution status
        cost_log = open(os.path.join(output_dir, f'cost_with_seed_{str(random_seed)}.txt'), 'w')
        print('******** Generation: ' + str(countloop) + ' ********')
        cost_log.write('******** Generation: ' + str(countloop) + ' ********' + '\n')
        prev_mean_total_cost = curr_mean_total_cost
        curr_mean_total_cost = np.mean(arfitsort[0:mu])
        print('The mean of total cost: ', curr_mean_total_cost)

        xmin[0:N] = arx[:, arindex[0]]
        xmin[N] = fc_corr_cost[arindex[0]]
        xmin[N + 1] = fc_L1_cost[arindex[0]]
        xmin[N + 2] = fcd_cost[arindex[0]]
        xmin[N + 3] = np.min(total_cost)
        cost_log.write('sigma: ' + str(sigma) + '\n')
        print('Best parameter set: ', arindex[0])
        cost_log.write('Best parameter set: ' + str(arindex[0]) + '\n')
        best_total_cost = np.min(total_cost)
        print('Best total cost: ', best_total_cost)
        cost_log.write('Best total cost: ' + str(best_total_cost) + '\n')
        print('FC correlation cost: ', fc_corr_cost[arindex[0]])
        cost_log.write('FC correlation cost: ' + str(fc_corr_cost[arindex[0]]) + '\n')
        print('FC L1 cost: ', fc_L1_cost[arindex[0]])
        cost_log.write('FC L1 cost: ' + str(fc_L1_cost[arindex[0]]) + '\n')
        print('FCD KS statistics cost: ', fcd_cost[arindex[0]])
        cost_log.write('FCD KS statistics cost: ' + str(fcd_cost[arindex[0]]) + '\n')
        print('wEI search range: ' + str(wEI_min) + ', ' + str(wEI_max))
        print('wEE search range: ' + str(wEE_min) + ', ' + str(wEE_max))
        cost_log.write('wEE search range: ' + str(wEE_min) + ', ' + str(wEE_max) + '\n')
        cost_log.write('wEI search range: ' + str(wEI_min) + ', ' + str(wEI_max) + '\n')

        elapsed_time = time.time() - start_time
        print('Elapsed time for this evolution is : ', elapsed_time)
        cost_log.write('Elapsed time for this evolution is : ' + str(elapsed_time) + '\n')
        print('******************************************')
        cost_log.write('******************************************')
        cost_log.write("\n")

    all_parameters = all_parameters[:, np.any(all_parameters, axis=0)]
    save_top_k_param(all_parameters, os.path.join(output_dir, output_file), k=1)

    iteration_log.write(str(countloop) + ' Success!')
    iteration_log.close()
    cost_log.close()


################################################################
# Gradient Descent Wrapper
################################################################
def get_search_range(n_node=68):
    n_param = 3 * n_node + 1
    wEE_min, wEE_max, wEI_min, wEI_max = 1, 10, 1, 5
    search_range = np.zeros((n_param, 2))
    search_range[0:n_node, :] = [wEE_min, wEE_max]  # search range for w_EE
    search_range[n_node:n_node * 2, :] = [wEI_min, wEI_max]  # search range for w_EI
    search_range[n_node * 2, :] = [0, 3]  # search range for G
    search_range[n_node * 2 + 1:n_param, :] = [0.0005, 0.01]  # search range for sigma
    return search_range


def get_init_param(random_seed, n_node=68):

    # n_param = 3 * n_node + 1
    rng = np.random.Generator(np.random.PCG64(random_seed))

    search_range = get_search_range(n_node)
    # range_lengths = search_range[:, 1] - search_range[:, 0]
    # search_range[:, 0] += range_lengths * 0.1
    # search_range[:, 1] -= range_lengths * 0.1
    init_param = rng.uniform(search_range[:, 0], search_range[:, 1])
    return torch.from_numpy(init_param).type(torch.FloatTensor)


def is_within_bounds(param: torch.Tensor, search_range=get_search_range()):
    np_param = param.detach().squeeze().cpu().numpy()
    return np.all(np_param >= search_range[:, 0]) and np.all(np_param <= search_range[:, 1])


def grad_desc_wrapper(random_seed=MANUAL_SEED,
                      n_iteration=1000,
                      dl_model=load_naive_net_no_SC(),
                      total_cost_threshold=0.63,
                      verbose_mode=False,
                      output_dir=os.path.join(PATH_TO_TESTING_REPORT, 'grad_desc_wrapper/predicted_best'),
                      output_file='predicted_best.csv'):
    print("Generating parameters using random seed:", random_seed)

    dl_model.eval()
    loss_fn = torch.nn.MSELoss()
    y_test = torch.tensor([[0, 0, 0]]).type(torch.FloatTensor).to(dl_model.device)

    param = get_init_param(random_seed).unsqueeze(0).to(dl_model.device)
    param.requires_grad = True

    for p in dl_model.parameters():
        p.requires_grad = False

    optimizer = optim.Adam([param], lr=8e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    prev_total_cost = np.inf
    curr_total_cost = 30
    curr_patience = 0
    stopping_threshold = 0.000001
    patience = 4

    for iteration in range(n_iteration):
        if prev_total_cost - curr_total_cost < stopping_threshold:
            if curr_patience > patience:
                break
            else:
                curr_patience += 1
        else:
            curr_patience = 0

        prev_param = param.detach().clone()
        prev_total_cost = curr_total_cost
        optimizer.zero_grad()
        y_pred = dl_model(param)
        curr_total_cost = y_pred.sum()
        loss = loss_fn(y_pred, y_test)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose_mode:
            print(
                f'Iteration: {iteration}, Loss: {loss}, Predicted cost: {y_pred[0, 0]}, {y_pred[0, 1]}, {y_pred[0, 2]}')

        if curr_total_cost > total_cost_threshold:
            if verbose_mode:
                print('Failed to find a good set of parameters')
            return

        if not is_within_bounds(param):
            param = prev_param
            break

    y_pred = y_pred.squeeze().detach().cpu().numpy()
    total_cost = np.array([y_pred.sum()])
    param = param.squeeze().detach().cpu().numpy()
    pred_with_param = np.concatenate((y_pred, total_cost, param))
    file_path = os.path.join(output_dir, f'seed_{str(random_seed)}', output_file)
    Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
    np.savetxt(file_path, pred_with_param, delimiter=',')
    return pred_with_param


def get_meaningful_params(split_name='train',
                          output_dir=os.path.join(PATH_TO_TESTING_REPORT, 'grad_desc_wrapper_given_init/'),
                          output_file='init_params.csv'):
    split_dataset = load_split(split_name)
    all_params = []
    all_performances = []
    for group in split_dataset.subject_groups:
        params, performances = group.sample_k_params(10, use_meaningful_only=True)
        all_params.append(params)
        all_performances.append(performances)
    all_params = torch.cat(all_params, axis=1)
    all_performances = torch.cat(all_performances, axis=1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_total_costs = all_performances[3, :].detach().cpu().numpy()
    histplot(all_total_costs)
    plt.title('distribution of parameters\' total costs')
    plt.xlabel(f'Total Costs (mean={all_total_costs.mean():.4f}, std={all_total_costs.std():.4f})')
    plt.savefig(os.path.join(output_dir, 'init_params_total_costs.png', bbox_inches='tight'), dpi=400)
    plt.clf()

    params_and_performances = torch.cat((all_performances, all_params), axis=0)
    params_and_performances = params_and_performances.detach().cpu().numpy()
    np.savetxt(os.path.join(output_dir, output_file), params_and_performances, delimiter=',')
    return params_and_performances


def get_meaningful_init_params():
    params_and_performances = pd.read_csv(os.path.join(PATH_TO_TESTING_REPORT,
                                                       'grad_desc_wrapper_given_init/init_params.csv'),
                                          header=None).to_numpy()
    init_params = params_and_performances[4:, :]
    return torch.from_numpy(init_params).type(torch.FloatTensor)


def grad_desc_wrapper_given_init(init_params=get_meaningful_init_params(),
                                 n_iteration=1000,
                                 dl_model=load_naive_net_no_SC(),
                                 verbose_mode=False,
                                 output_dir=os.path.join(PATH_TO_TESTING_REPORT,
                                                         'grad_desc_wrapper_given_init/predicted_best'),
                                 output_file='all_predicted_params.csv'):

    dl_model.eval()
    loss_fn = torch.nn.MSELoss()
    y_test = torch.tensor([[0, 0, 0]]).type(torch.FloatTensor).to(dl_model.device)
    n_param = init_params.shape[1]

    all_predicted_params = []
    for i in range(n_param):
        print(f'Optimizing {i}th parameter with gradient descent')
        param = init_params[:, i].unsqueeze(0).to(dl_model.device)
        param.requires_grad = True

        for p in dl_model.parameters():
            p.requires_grad = False

        optimizer = optim.Adam([param], lr=8e-6)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        prev_total_cost = np.inf
        curr_total_cost = 30
        curr_patience = 0
        stopping_threshold = 0.000001
        patience = 4

        for iteration in range(n_iteration):
            if prev_total_cost - curr_total_cost < stopping_threshold:
                if curr_patience > patience:
                    break
                else:
                    curr_patience += 1
            else:
                curr_patience = 0

            prev_param = param.detach().clone()
            prev_total_cost = curr_total_cost
            optimizer.zero_grad()
            y_pred = dl_model(param)
            curr_total_cost = y_pred.sum()
            loss = loss_fn(y_pred, y_test)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if verbose_mode:
                print(
                    f'\tIteration: {iteration}, Loss: {loss}, Predicted cost: {y_pred[0, 0]}, {y_pred[0, 1]}, {y_pred[0, 2]}'
                )

            if not is_within_bounds(param):
                param = prev_param
                break

        y_pred = y_pred.squeeze().detach().cpu().numpy()
        total_cost = np.array([y_pred.sum()])
        param = param.squeeze().detach().cpu().numpy()
        pred_with_param = np.concatenate((y_pred, total_cost, param))
        all_predicted_params.append(pred_with_param)

    all_predicted_params = np.stack(all_predicted_params, axis=1)
    file_path = os.path.join(output_dir, output_file)
    Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
    np.savetxt(file_path, all_predicted_params, delimiter=',')
    return all_predicted_params


def compare_hist(path_to_init_params=os.path.join(PATH_TO_TESTING_REPORT,
                                                  'grad_desc_wrapper_given_init/init_params.csv'),
                 path_to_optimized_params=os.path.join(
                     PATH_TO_TESTING_REPORT,
                     'grad_desc_wrapper_given_init/validation/3/actual_costs_and_prediction.csv'),
                 init_params_fig_file_name='init_params_total_costs.png',
                 optimized_params_fig_file_name='optimized_params_total_costs.png'):
    init_params = pd.read_csv(path_to_init_params, header=None).to_numpy()
    optimized_params = pd.read_csv(path_to_optimized_params, header=None).to_numpy()
    save_dir = os.path.split(path_to_init_params)[0]

    init_total_costs = init_params[3, :]
    optimized_total_costs = optimized_params[3, :]
    optimized_total_costs[optimized_total_costs >= 3] = 3
    num_improved = (optimized_total_costs < init_total_costs).sum()
    optimized_total_costs = optimized_total_costs[optimized_total_costs < 3]

    histplot(init_total_costs)
    plt.title('Distribution of initial parameters\' total costs')
    plt.xlabel(f'Total Costs (mean={init_total_costs.mean():.4f}, std={init_total_costs.std():.4f})')
    plt.savefig(os.path.join(save_dir, init_params_fig_file_name), dpi=400, bbox_inches='tight')
    plt.clf()

    histplot(optimized_total_costs)
    plt.title(f'Distribution of optimized parameters\' total costs\n({num_improved} params\' total costs improved)')
    plt.xlabel(f'Total Costs (mean={optimized_total_costs.mean():.4f}, std={optimized_total_costs.std():.4f})')
    plt.savefig(os.path.join(save_dir, optimized_params_fig_file_name), dpi=400, bbox_inches='tight')
    plt.clf()


def compare_hist_top_k(k=50):
    extract_top_k(os.path.join(PATH_TO_TESTING_REPORT, 'grad_desc_wrapper_given_init/init_params.csv'), k=50)
    compare_hist(path_to_init_params=os.path.join(PATH_TO_TESTING_REPORT,
                                                  f'grad_desc_wrapper_given_init/init_params_top_{k}.csv'),
                 path_to_optimized_params=os.path.join(
                     PATH_TO_TESTING_REPORT, 'grad_desc_wrapper_given_init/test/1/actual_costs_and_prediction.csv'),
                 init_params_fig_file_name=f'init_params_total_costs_top_{k}.png',
                 optimized_params_fig_file_name=f'optimized_params_total_costs_top_{k}.png')


################################################################
# pMFM simulation to get the actual costs
################################################################
def get_all_predicted_best_param(folder_path=os.path.join(PATH_TO_TESTING_REPORT, 'grad_desc_wrapper/predicted_best')):
    predicted_best_param_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if ("predicted_best" in file):
                predicted_best_param_paths.append(os.path.join(root, file))

    print('Number of params: ', len(predicted_best_param_paths))
    params = []
    for path in predicted_best_param_paths:
        params.append(np.loadtxt(path, delimiter=','))

    all_predicted_best_params = np.stack(params, axis=1)
    np.savetxt(os.path.join(folder_path, 'all_predicted_params.csv'), all_predicted_best_params, delimiter=',')


def simulate_with_predicted_best(split_name,
                                 group_index,
                                 param_path=os.path.join(
                                     PATH_TO_TESTING_REPORT,
                                     'grad_desc_wrapper_given_init/predicted_best/all_predicted_params.csv')):
    path_to_group = get_path_to_group(split_name, group_index)
    all_predicted_params = np.loadtxt(param_path, delimiter=',')
    params = all_predicted_params[4:, :]
    predicted_costs = all_predicted_params[:4, :]
    fc_corr_cost, fc_L1_cost, fcd_cost, total_cost = forward_simulation(params, path_to_group)
    actual_costs = np.stack((fc_corr_cost, fc_L1_cost, fcd_cost, total_cost), axis=0)
    actual_costs_and_prediction = np.concatenate((actual_costs, params, predicted_costs), axis=0)

    save_dir = os.path.split((os.path.split(param_path)[0]))[0]
    save_dir = os.path.join(save_dir, split_name, str(group_index))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.savetxt(os.path.join(save_dir, 'actual_costs_and_prediction.csv'), actual_costs_and_prediction, delimiter=',')


def simulate_with_predicted_best_wrapper(param_folder_name='grad_desc_wrapper_given_init'):
    split_name = sys.argv[1]
    group_index = sys.argv[2]
    param_path = os.path.join(PATH_TO_TESTING_REPORT, f'{param_folder_name}/predicted_best/all_predicted_params.csv')
    if split_name == 'test':
        param_path = os.path.join(PATH_TO_TESTING_REPORT, f'{param_folder_name}/validation/top_params_from_val.csv')
    simulate_with_predicted_best(split_name, group_index, param_path=param_path)


def try_diff_seed(seed_range):
    group_path = get_path_to_group('train', '5')
    SC_path = os.path.join(group_path, 'group_level_SC.csv')
    myelin_path = os.path.join(group_path, 'group_level_myelin.csv')
    gradient_path = os.path.join(group_path, 'group_level_RSFC_gradient.csv')

    for random_seed in seed_range:
        # cmaes_wrapper(random_seed=random_seed)
        cmaes_wrapper_use_coef(myelin_path, gradient_path, SC_path, random_seed=random_seed)
        # grad_desc_wrapper(random_seed=random_seed, n_iteration=10000, verbose_mode=True)


################################################################
# get top k params with the best validation costs
################################################################
def replace_high_threshold(file_path):
    all_params = np.loadtxt(file_path, delimiter=',')
    fc_corr_cost = all_params[0, :]
    fc_L1_cost = all_params[1, :]
    fcd_cost = all_params[2, :]
    total_cost = all_params[3, :]

    fc_corr_cost[fc_corr_cost >= 1] = 1
    fc_L1_cost[fc_L1_cost >= 1] = 1
    fcd_cost[fcd_cost >= 1] = 1
    total_cost[total_cost >= 3] = 3
    new_performances = np.stack((fc_corr_cost, fc_L1_cost, fcd_cost, total_cost), axis=0)  # shape: (4, k)

    all_params[:4, :] = new_performances

    np.savetxt(file_path, all_params, delimiter=',')


def extract_top_k(file_path, k=20):
    all_params = np.loadtxt(file_path, delimiter=',')
    actual_total_costs = all_params[3, :]
    sorted_indices = np.argsort(actual_total_costs)
    top_k_params = all_params[:, sorted_indices[:k]]
    save_file_path = file_path.replace('.csv', f'_top_{k}.csv')
    np.savetxt(save_file_path, top_k_params, delimiter=',')


def remove_actual_costs(file_path, save_dir='./'):
    all_params = np.loadtxt(file_path, delimiter=',')
    # actual_costs = all_params[:4, :]
    params = all_params[4:209, :]
    predicted_costs = all_params[-4:, :]
    top_params_from_val = np.concatenate((predicted_costs, params), axis=0)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.savetxt(os.path.join(save_dir, 'top_params_from_val.csv'), top_params_from_val, delimiter=',')


def extract_top_k_in_val(k=50, method_name='grad_desc_wrapper_given_init'):
    for i in range(1, 15):
        file_path = os.path.join(PATH_TO_TESTING_REPORT,
                                 f'{method_name}/validation/{i}/actual_costs_and_prediction.csv')
        replace_high_threshold(file_path)
        extract_top_k(file_path, k=k)


def use_top_params_from(group_index, k, method_name='grad_desc_wrapper_given_init'):
    remove_actual_costs(os.path.join(PATH_TO_TESTING_REPORT,
                                     f'{method_name}/validation/{group_index}/actual_costs_and_prediction_top_{k}.csv'),
                        save_dir=os.path.join(PATH_TO_TESTING_REPORT, f'{method_name}/validation/'))


if __name__ == '__main__':
    method_name = 'grad_desc_wrapper_given_init'
    # ONLY for grad descent with init params
    # get_meaningful_params()
    # grad_desc_wrapper_given_init(verbose_mode=True)

    # Use different random seed to generate params with predicted good costs
    # try_diff_seed(range(1, 601))
    # get_all_predicted_best_param(folder_path=os.path.join(PATH_TO_TESTING_REPORT, f'{method_name}/predicted_best'))

    # Simulate params with predicted best costs with validation set SC
    # simulate_with_predicted_best_wrapper(method_name)
    # extract_top_k_in_val(k=50, method_name=method_name)
    # use_top_params_from('5', k=50, method_name=method_name)

    # Simulate top params from validation with test set SC
    # simulate_with_predicted_best_wrapper(method_name)

    # Generating plots
    # pred_and_actual_cost_corr_dist(
    #     'test',
    #     base_dir=os.path.join(PATH_TO_TESTING_REPORT, method_name),
    #     prediction_file_name='actual_costs_and_prediction.csv')  # generate plot for all groups in split  # noqa: E501
    # all_costs_prediction_vs_actual_cost(os.path.join(PATH_TO_TESTING_REPORT, f'{method_name}/validation/2'),
    #                                     'actual_costs_and_prediction.csv')  # noqa: E501  # generate plot for one group

    # ONLY for grad descent with init params
    # compare_hist()
    # compare_hist_top_k(k=50)
