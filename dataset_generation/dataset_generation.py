# /usr/bin/env python
# Written by Kong Xiaolu, Shaoshi Zhang, Tian Fang and CBIG under MIT license:
# https://github.com/ThomasYeolab/CBIG/blob/master/LICENSE.md

import numpy as np
import random
import time
import torch
import warnings
import sys
import os
from multiprocessing import Pool

sys.path.insert(0, '/home/ftian/storage/pMFM_speedup/')
import src.utils.CBIG_pMFM_basic_functions_HCP as fc


def get_init(myelin_data, gradient_data, num_myelin_component, num_gradient_component, init_para):
    '''
    This function is implemented to calculate the initial parametrized coefficients
    '''

    n_node = myelin_data.shape[0]
    concat_matrix = np.vstack((np.ones(n_node), myelin_data.T, gradient_data.T)).T  # bias, myelin PC, RSFC gradient PC
    para = np.linalg.inv(concat_matrix.T @ concat_matrix) @ concat_matrix.T @ init_para
    return para, concat_matrix


def CBIG_mfm_optimization_desikan_main(input_args):
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
    assert len(input_args) == 4
    output_path = input_args[0]
    output_file = input_args[1]
    path_to_group = input_args[2]
    FCD = path_to_group + 'group_level_FCD.mat'
    SC = path_to_group + 'group_level_SC.csv'
    FC = path_to_group + 'group_level_FC.csv'
    random_seed = input_args[3]

    # torch.cuda.set_device(gpu_number)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Setting random seed and GPU
    random_seed_cuda = random_seed
    random_seed_np = random_seed
    torch.manual_seed(random_seed_cuda)
    rng = np.random.Generator(np.random.PCG64(random_seed_np))

    # Initializing input parameters
    myelin_data = fc.csv_matrix_read(path_to_group + 'group_level_myelin.csv')
    num_myelin_component = myelin_data.shape[1]

    gradient_data = fc.csv_matrix_read(path_to_group + 'group_level_RSFC_gradient.csv')
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
    with open(output_path + output_file, 'w') as f:
        while countloop < maxloop:
            iteration_log = open(output_path + 'training_iteration' + '.txt', 'w')
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
            print("Forward simulation starts ...")
            total_cost, fc_corr_cost, fc_L1_cost, fcd_cost, bold_d, r_E, emp_fc_mean = fc.CBIG_combined_cost_train(
                input_para, n_dup, FCD, SC, FC, countloop)
            # print(input_para.shape, total_cost.shape, fc_corr_cost.shape,
            #       fc_L1_cost.shape, fcd_cost.shape, bold_d.shape, r_E.shape)
            ## (205, 10) (10,) (10,) (10,) (10,) torch.Size([68, 50, 1200]) torch.Size([68, 50]) when Lambda = 10

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
            cost_log = open(output_path + 'cost_' + str(random_seed) + '.txt', 'w')
            print('******** Generation: ' + str(countloop) + ' ********')
            cost_log.write('******** Generation: ' + str(countloop) + ' ********' + '\n')
            print('The mean of total cost: ', np.mean(arfitsort[0:mu]))

            xmin[0:N] = arx[:, arindex[0]]
            xmin[N] = fc_corr_cost[arindex[0]]
            xmin[N + 1] = fc_L1_cost[arindex[0]]
            xmin[N + 2] = fcd_cost[arindex[0]]
            xmin[N + 3] = np.min(total_cost)
            xmin_save = np.reshape(xmin, (-1, N + 4))
            cost_log.write('sigma: ' + str(sigma) + '\n')
            print('Best parameter set: ', arindex[0])
            cost_log.write('Best parameter set: ' + str(arindex[0]) + '\n')
            print('Best total cost: ', np.min(total_cost))
            cost_log.write('Best total cost: ' + str(np.min(total_cost)) + '\n')
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

        np.savetxt(f, all_parameters, delimiter=',')

    iteration_log.write(str(countloop) + ' Success!')
    iteration_log.close()
    cost_log.close()


def get_random_seed_for(split_name, idx_in_split):
    assert split_name in {'train', 'validation', 'test'}
    split_name = str(split_name)
    idx_in_split = int(idx_in_split)

    if split_name == 'train':
        random_seed = random.randint(1, 99999)
    elif split_name == 'validation':
        random_seed = random.randint(100000, 199999)
    else:
        random_seed = random.randint(200000, 299999)

    return random_seed


def generate_parameters_for(split_name, idx_in_split_range):
    split_name = str(split_name)
    input_args = []
    for idx_in_split in idx_in_split_range:
        random_seed = get_random_seed_for(split_name, idx_in_split)
        idx_in_split = str(idx_in_split)
        output_path = '/home/ftian/storage/pMFM_speedup/dataset_generation/dataset/' + split_name + '/' + idx_in_split + '/'
        path_to_group = '/home/ftian/storage/pMFM_speedup/dataset_generation/input_to_pMFM/' + split_name + '/' + idx_in_split + '/'

        input_args.append([output_path, split_name + '_' + idx_in_split + '.csv', path_to_group, random_seed])

    p = Pool(len(idx_in_split_range))
    p.map(CBIG_mfm_optimization_desikan_main, input_args)


def split(r, n_batches):
    k, m = divmod(len(r), n_batches)
    return list((r[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_batches)))


def generate_parameters_for_batch(split_name, batch_idx, n_batches):
    assert split_name in {'train', 'validation', 'test'}
    batch_idx = int(batch_idx)
    n_batches = int(n_batches)

    if split_name == 'train':
        whole_range = range(1, 58)
    elif split_name == 'validation':
        whole_range = range(1, 15)
    else:
        whole_range = range(1, 18)

    idx_in_split_range = split(whole_range, n_batches)[batch_idx]
    generate_parameters_for(split_name, idx_in_split_range)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    split_name = sys.argv[1]
    batch_idx = sys.argv[2]
    n_batches = sys.argv[3]
    generate_parameters_for_batch(split_name, batch_idx, n_batches)
