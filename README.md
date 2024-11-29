# pMFM_speedup

The goal of this project is to use machine learning or deep learning (DL) to speed up the parameter optimization process of the Mean Field Model (MFM), or more specifically parametric MFM (pMFM). This repository has provided various trained models that can predict a pMFM parameter's costs. Additionally, the script for pMFM parameter optimization using the DL model is also provided, which can generate a set of pMFM parameters with decent costs. Compared with the original method for parameter optimization (CMA-ES combined with Euler integration that solves the ODE of the pMFM) which takes several hours to complete, the method proposed by this project would only take a few minutes. The significant speedup can greatly benefit downstream applications/experiments that require good parameters.



* [Introduction](#introduction)
  + [pMFM](#pmfm)
  + [Motivation for pMFM_speedup](#motivation-for-pmfm_speedup)
* [Dataset Creation](#dataset-creation)
  + [Data Availability](#data-availability)
  + [Dataset Generation Process](#dataset-generation-process)
    - [Parameter Selection for each Subject Group](#parameter-selection-for-each-subject-group)
    - [Bad Parameters with out-of-range excitatory firing rates](#bad-parameters-with-out-of-range-excitatory-firing-rates)
  + [Generated Dataset](#generated-dataset)
* [Assumption Validation](#assumption-validation)
  + [Assumption 1: Simulated FC and FCD are not sensitive to changes in SC](#assumption-1-simulated-fc-and-fcd-are-not-sensitive-to-changes-in-sc)
  + [Assumption 2: Simulated FC and FCD are not sensitive to changes in param vector](#assumption-2-simulated-fc-and-fcd-are-not-sensitive-to-changes-in-param-vector)
* [Models](#models)
  + [Common elements for model training](#common-elements-for-model-training)
  + [SC Feature Extractor](#sc-feature-extractor)
  + [Experimented Models](#experimented-models)
    - [Naive Net](#naive-net)
  + [Naive Net with Coefficient Vector](#naive-net-with-coefficient-vector)
    - [Plain GCN](#plain-gcn)
* [Experiment Results](#experiment-results)
  + [Compare Prediction and Actual Costs](#compare-prediction-and-actual-costs)
* [Parameter Optimization using the DL model](#parameter-optimization-using-the-dl-model)
  + [Gradient Descent with DL Model](#gradient-descent-with-dl-model)
    - [Random Initialization](#random-initialization)
    - [Feed in parameters](#feed-in-parameters)
  + [CMA-ES with DL Model](#cma-es-with-dl-model)
  + [CMA-ES with DL Model and Parameterization](#cma-es-with-dl-model-and-parameterization)
* [Limitation](#limitation)
* [Developer Guide](#developer-guide)
  + [Setup](#setup)
  + [Folder Structure](#folder-structure)
  + [Naming Conventions](#naming-conventions)
  + [Dataset Generation](#dataset-generation)
    - [Empirical Group-Level Information](#empirical-group-level-information)
    - [Use CMA-ES with pMFM to Generate Parameters along with their Costs](#use-cma-es-with-pmfm-to-generate-parameters-along-with-their-costs)
  + [Dataset and Models](#dataset-and-models)
    - [Dataset](#dataset)
    - [Models](#models-1)
  + [Model Training](#model-training)
  + [Model Testing and Wrapper Functions](#model-testing-and-wrapper-functions)
  + [Trained Model](#trained-model)
* [Bugs and Questions](#bugs-and-questions)



Below are the detailed descriptions of this project. For the usage and organization of codes in this repository, please refer to the **[Developer Guide](#developer-guide)** at the end of this document.

## Introduction

### pMFM

![pMFM_speedup-pMFM](https://user-images.githubusercontent.com/61874388/168448462-7efcb55a-93f0-4b68-8ad8-00484b76b4fb.png)

Proposed by the NUS Computational Brain Imaging Group (CBIG), the [**parametric Mean Field Model (pMFM)**](https://www.nature.com/articles/s41467-021-26704-y) is a computational model for simulating and
understanding human brain dynamics.

In the context of this project, we consider 68 brain regions (i.e. cortical regions of interest (ROI)). Each human subject has a 68x68 brain structural connectivity (SC) matrix, indicating how strong the physical connections between different brain regions are. Each subject had also gone through the resting-state functional magnetic resonance imaging (rs-fMRI) procedure to generate the corresponding BOLD (Blood Oxygenation Level-Dependent) signal, which can reflect how active each brain regions are. The BOLD signal is then used to generate a 68x68 functional connectivity (FC) matrix (How strong are the correlations between activities of brain region pairs) and an 1118 × 1118 functional connectivity dynamics (FCD) matrix (how FC changes over time). Subjects are formed into groups, with corresponding group-averaged SC, FC, and FCD

The input to the pMFM includes a group-level SC, and 205 parameters (1 + 3 x 68 = 205): There is a global scaling factor G, and each brain region has three parameters wEE, wEI, and 𝜎. After taking in an SC and a parameter vector of length 205, the pMFM will generate a simulated BOLD signal, which is then used to generate simulated FC and FCD. Then, the simulated FC and FCD are compared with the empirical FC and FCD, which comes from the empirical BOLD signals of subjects in the subject group. The metrics we use to compare these matrices are as follows:

- FC_CORR: FC correlation cost, defined by 1 - correlation between the simulated FC and the empirical FC
- FC_L1: The L1 distance between the simulated FC and the empirical FC
- FCD_KS: The KS statistic between the simulated FCD and the empirical FCD

To accurately capture the human brain dynamics, we need to find a set of good parameters for MFM such that the BOLD signals generated have similar FC and FCD as that of the empirical ones.

### Motivation for pMFM_speedup

![pMFM_speedup-Deep Learning Model](https://s2.loli.net/2022/11/10/Rfrw8odSjcPNkVl.png)

The current pMFM has a major downside: the simulation is very slow. The underlying equations of pMFM are nonlinear ordinary differential equations (ODEs) that do not have a closed-form solution. Hence, the forward Euler method (a type of numerical procedure for solving ODEs) is used in pMFM, which has an undesirable simulation speed. Therefore, it requires a significant amount of time to find a good set of parameter (parameter vector of length 205). The goal of pMFM_speedup is to come up with a deep-learning (DL) model to help the parameter selection process. The inputs to pMFM_speedup models are the same as that of pMFM and the pMFM_speedup models will perform a regression task to output a cost vector containing FC_CORR, FC_L1, and FCD_KS indicating how good the input parameter vector is. In this way, the pMFM_speedup can filter out bad parameters with much less time (compared to pMFM), and we only need to use pMFM for those parameters with good costs.

## Dataset Creation

### Data Availability

The raw diffusion MRI, rs-fMRI, and T1w/T2w data are retrieved from the Human Connectome Project ([HCP](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release))

### Dataset Generation Process

<img width="500" alt="pMFM_speedup-Subject Groups" src="https://user-images.githubusercontent.com/61874388/171318768-818c5e3e-afff-4cf4-a003-267dcdcedf05.png">

- 1000 subjects are split into 88 groups
  - There are 50 subjects within each group
  - The train set has 57 groups, the validation set has 14 groups, and the test set has 17 groups
    - Subjects are not shared between train/validation/test sets (to avoid **data snooping/leakage**)
    - Adjacent groups (e.g. train 1 and train 2) share 40 subjects
- Generate group-level SCs from individual SCs and get empirical FCs, empirical FCDs from empirical BOLD signals
- Feed the SC and selected parameters (see below) for each group into pMFM and generate pMFM FC and FCD
- Generate correlation FC_CORR and L1 distance FC_L1 between pMFM FC and the empirical one
- Generate KS statistics FCD_KS between pMFM FCD and the empirical one

#### Parameter Selection for each Subject Group

- Initialize CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with G, wEE, wEI, and 𝜎 in their respective ranges
  - Range for each wEE: [1, 10]
  - Range for each wEI: [1, 5]
  - Range for each 𝜎: [0.0005, 0.01]
  - Range for G: [0, 3]
- Use CMA-ES (100 iterations with each iteration yielding 100 children) to generate 10000 different parameter vectors for each subject group
- We will have 88 SCs, each with 10000 parameter vectors, resulting in 880,000 inputs

#### Bad Parameters with out-of-range excitatory firing rates

Among 10000 parameters for each subject group, some of them are bad parameters with out-of-range excitatory firing rates. Consequently, these bad parameters cannot yield a BOLD time course when fed into the pMFM, let alone generate corresponding FC_CORR, FC_L1, and FCD_KS costs. Hence, we arbitrarily set the FC_CORR, FC_L1, and FCD_KS costs for these bad parameters to be 1, 1, and 1 respectively.

### Generated Dataset

To summarize, the generated dataset contains 880,000 inputs, each containing:

- Input features:

  - Group-level SC

  - wEE, wEI, and 𝜎 for each ROI

  - Global scaling factor G


- Target value/Ground truths:
  - Corresponding FC correlation, FC L1 distance, and FCD KS costs




## Assumption Validation

Below are some interesting findings about FC, FCD, SC, and param vector.

### Assumption 1: Simulated FC and FCD are not sensitive to changes in SC

**Motivation**: The pair-wise correlations (Pearson correlation coefficient) between all group SCs are computed and they are highly similar. Hence changes in SC may not result in large variation in FC or FCD

<img width="500" alt="corr_between_all_SCs" src="https://s2.loli.net/2022/11/14/Vg8DMonXqpWN7cZ.png">

**Validation**: The validation process is described below:

- Randomly choose a parameter vector from the train/validation set
- Fix the parameter vector and use different SCs with that parameter vector to generate corresponding simulated FC and FCD
- For each pair of SC, compute the correlation between SCs and the pair-wise FC correlation/FC L1 cost/FCD KS statistic

<img width="250" alt="costs_vs_SC_correlation" src="https://s2.loli.net/2022/11/14/KSMOrVfNQHdzFBW.png">

The results for one of the param vectors are shown below:![image-20221110162803524](https://s2.loli.net/2022/11/10/EyDq69nSfLCKVP8.png)

![image-20221110162812616](https://s2.loli.net/2022/11/10/J5Pt34ExBDg78Fb.png)

![image-20221110162820430](https://s2.loli.net/2022/11/10/R2gSoKEW7XH6Pa1.png)

This validation process is repeated for different param vectors and the results are similar. As shown above, when SC changes, the simulated FC/FCD are quite similar (high FC correlation, low FC L1 cost, low FCD KS statistic). Hence, the simulated FC and FCD are not sensitive to changes in SC.

### Assumption 2: Simulated FC and FCD are not sensitive to changes in param vector

**Motivation**: After verifying the simulated FC and FCD are not sensitive to changes in SC, we are also interested in whether the simulated FC and FCD are sensitive to changes in the param vector

**Validation**: The validation process is very similar to that of SC:

- Randomly choose an SC from the train/validation set
- Fix the SC and use different parameter vectors with that SC to generate corresponding simulated FC and FCD
- For each pair of parameter vectors, compute the correlation between parameter vectors and the pair-wise FC correlation/FC L1 cost/FCD KS statistic

<img width="250" alt="costs_vs_param_correlation" src="https://s2.loli.net/2022/11/14/g7Dnsl2zy8atUbp.png">

The results for one of the param vectors are shown below:

![image-20221110170219758](https://s2.loli.net/2022/11/10/hpGXD21OUwJknMx.png)

![image-20221110170230210](https://s2.loli.net/2022/11/10/sQCw5dkZjDSOmhX.png)

![image-20221110170235849](https://s2.loli.net/2022/11/10/zywpaSmMdeQBDNR.png)

This validation process is repeated for different param vectors and the results are similar. The above figures demonstrated that when the param vector changes, the simulated FC/FCD are also quite similar (high FC correlation, low FC L1 cost, low FCD KS statistic). Hence, the simulated FC and FCD are not sensitive to changes in the param vector.



## Models

### Common elements for model training

- **Loss function**: Mean Square Error (MSE) Loss

- **Optimizer**: Adam optimizer (initial lr = 5e-4) with exponential decay learning rate scheduler (multiply lr by 0.98 every epoch)

- **Batch Size**: 256

- **Metrics logged**:
  - During **training**: MSE loss across 3 different cost terms
  - During **validation**: MSE loss across 3 different cost terms, and MSE loss for each individual cost

- **Hyperparameter** **tuning**: Optuna (with max epoch equal to 100)

### SC Feature Extractor

Since the SC matrix has many parameters (68x68) and SC matrices from different groups are highly similar. We can use neural nets to extract features/embedding from the SC matrix and proceed further with the SC feature with a much lower dimension. In this way, we have much fewer input parameters, which can address overfitting to some extent.

Two SC feature extractors are provided:

- CNN (Convolutional Neural Network) version: we treat an SC matrix as a 68x68 image and use convolution to extract latent features of the SC matrix

  ![pMFM_speedup-Extract SC Feature CNN](https://user-images.githubusercontent.com/61874388/171318608-9f0e0af0-0045-446d-b3c9-56123c536341.png)

- MLP (Multi-Layer Perceptron) version: we first vectorize the upper triangular part of the SC (without the diagonal entries), and feed the SC vector into an MLP to extract features

  ![pMFM_speedup-Extract SC Feature MLP](https://s2.loli.net/2022/11/10/taW5dM4Vnbsgp7e.png)

The experiments below used MLP as the SC feature extractor.

### Experimented Models

#### Naive Net

![pMFM_speedup-Naive Net](https://user-images.githubusercontent.com/61874388/171318654-6f3837a8-d603-436c-a39f-306427afd5bc.png)

The Naive Net is a straightforward simple MLP network with several ReLU-activated linear layers. If the SC feature is extracted and used as a part of the input, it will be concatenated with the param vector.

### Naive Net with Coefficient Vector

![pMFM_speedup-Naive Net using coefficient](https://s2.loli.net/2022/11/14/ydbT8zifp4tcLkw.png)

Note that the wEE, wEI, and 𝜎 for each ROI can be derived using 9 coefficients, group-level myelin, and group-level RSFC gradient (as wEE, wEI, and 𝜎 have been parameterized by myelin and RSFC gradient). Hence, we have a variant of the Naive Net, where we use the Coefficient Vector (9 linear coefficients and the global scaling factor G) as the input.

#### Plain GCN

![pMFM_speedup-GCN](https://s2.loli.net/2022/11/14/XpMvK8aLSJ1H2Ut.png)

We can treat each brain region as a node, and the parameters associated with a brain region as the corresponding node features. The edge and edge weight come from non-zero entries of a group's SC. The global scaling factor G is used to scale the edge weight, similar to its role in the pMFM (uniformly scaling connections between brain regions). The node features at the end of GCN are concatenated and fed into an MLP for the final prediction.



## Experiment Results

Each of the models is tunned by Optuna: 100 sets of hyperparameters have been tried for each model, and each trial's max number of epochs is 100. Then the top 10 sets of hyperparameters are picked based on validation cost, and their performances (total MSE loss, which is mathematically equivalent to the mean of 3 cost terms' MSEs) are shown in the box plot below.

<img width="500" alt="mse_loss_box_plot" src="https://s2.loli.net/2022/11/10/9C3JVch6tjXxgMl.png">

We can see that Naive Net without SC performs the best (has the lowest validation loss and the lowest variation).

The MSE losses for each cost term are shown below:

|                        FC_CORR’s MSE                         |                         FC_L1’s MSE                          |                         FCD_KS’s MSE                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![FC_CORR’s MSE](https://s2.loli.net/2022/11/10/PykjoqO3VZKX8MC.png) | ![FC_L1’s MSE](https://s2.loli.net/2022/11/10/ifYvMbZ6d7k2mX1.png) | ![FCD_KS’s MSE](https://s2.loli.net/2022/11/10/kAT8wPhGNSqBuV7.png) |

We can see that the individual MSE is consistent with the total MSE. And Naive Net without SC performs consistently better than other models.

The best model (Naive Net without SC) is then tested on the test set, and the MSE loss is 0.0248.

### Compare Prediction and Actual Costs

We then evaluate the best model's (Naive Net without SC) performance on each test subject group by comparing the predictions (output from the DL model) and the actual costs (the ground truth costs derived from comparing pMFM FC and FCD with the empirical FC and FCD). The results for total costs (sum of the 3 cost terms) are shown below. The figure on the left shows predicted total costs and the actual total costs for one test group (dots with an actual cost of 3 represent bad parameters with excitatory firing rate outside 2.7 Hz and 3.3Hz when fed into pMFM). The distribution on the right shows the correlations between the predicted total costs and the actual total costs for different test groups.

| Predicted total costs and actual total costs for one test group | Distribution of correlation between predicted total costs and actual total costs in the test set |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![Predicted costs and actual costs for one test group ](https://s2.loli.net/2022/11/10/B5LZ48UDNMuAJGd.png) | ![image-20221110191738050](https://s2.loli.net/2022/11/10/fzZVhcU5S8dl9FC.png) |

The comparison between the prediction and the actual cost is also performed for individual cost terms and the results can be found in `reports/slides/CBIG_20220810.pptx` as well as in `reports/testing/compare_top_k_params/basic_models/naive_net/no_SC`. 

Overall, the predictions and actual costs are strongly correlated, which shows the effectiveness of the DL model.



## Parameter Optimization using the DL model

Up to the previous section, we are training and testing the DL model on the dataset generated by the CMA-ES with pMFM. To really validate the effectiveness of the DL model on parameter optimization, we should use the DL model alone to generate some param vectors with good predicted costs, feed those params into the pMFM, and check whether those params yield great actual costs.

Three wrapper methods have been tried to optimize parameters using the DL model (Naive Net): Gradient Descent with DL Model, CMA-ES with DL Model, and CMA-ES with DL Model and Parameterization.

> Note that the best model tested so far is the Naive Net without SC. Hence, the parameter optimization using this model will not require group-level SC as an input.

### Gradient Descent with DL Model

Since the prediction is the result of a series of differentiable operations, we can use gradient descent (GD) to update the input parameters. Specifically, we can fix the trained model's weights and use an optimizer to optimize the input parameter. Additionally, we need to make sure the optimized parameters are within the pre-defined range (e.g.,  wEE is in [1, 10]). Hence, the GD will stop when any of the parameters is going out of the pre-defined range.

#### Random Initialization

At first, random initialization is used to generate the initial param vector to be optimized by GD. We tried 80,000 random seeds have been tried, 498 of which yield param vectors with predicted total costs lower than 0.68. However, when fed into pMFM, these parameters are all bad (the excitatory firing rate is not within 2.7 Hz and 3.3Hz).

#### Feed in parameters

As random initialization usually results in bad parameters, feeding in "not bad" parameters that can generate BOLD signals with pMFM is also attempted.

To begin with, 570 param vectors were randomly selected from the train set, the total costs for these initial param vectors are shown on the left of the figure below. Then, these 570 param vectors were optimized using the GD with the DL Model, and the total cost distribution after optimization is shown on the right of the figure below.

| Total costs of the initial 570 param vectors                 | Total cost distribution of the optimized 570 param vectors   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![Total costs of the initial 570 param vectors](https://s2.loli.net/2022/11/14/nSTUeARpdyiujKY.png) | ![Total cost distribution of the optimized 570 parameters](https://s2.loli.net/2022/11/14/71AajHq6xuyOCfr.png) |

Although it seems that the total cost distribution is more 'skewed towards left' after optimization (indicating overall lower total costs), two problems are observed:

Firstly, among 570 param vectors, only 302 param vectors have improved total costs after optimization. Additionally, param vectors with very low total costs are worsened by the GD with the DL Model as the lowest total cost after optimization is higher than that of the initial param vectors.

Second, it is noticed that the GD process quickly stops (usually 10-30 steps with a step size of `8e-6`) as one of the 205 parameters is going out of the pre-defined range. This may be the reason for the first problem as GD did not change the input param vector much.

Overall, GD with the DL model shows limited capability in terms of optimizing pMFM's parameters.

### CMA-ES with DL Model

A straightforward solution to optimizing parameters using the DL model is replacing the pMFM's forward Euler method with the DL model. Specifically, CMA-ES with DL Model:

1. Initialize in the same way as the forward-Euler-method-version CMA-ES
2. Optimize each of the 205 parameters
3. Use trained DL model (instead of forward Euler method) to predict costs of param vectors
4. Run up to 500 CMA-ES iterations, each yielding 100 children
5. Choose 1 param vector with the best-predicted costs

Such a process is repeated with different random seeds to generate 540 parameters. However, when they are fed into the pMFM with an SC, none of the excitatory firing rates are within 2.7 Hz and 3.3 Hz, which means they are all bad parameters.

![pMFM_speedup-DL_version_CMA-ES](https://s2.loli.net/2022/11/11/vGU81soC75mQyLR.png)

Therefore, it might be too difficult to optimize each of the 205 parameters at the same time, which gives rise to the next approach, which uses parameterization similar to the original CMA-ES with pMFM.

### CMA-ES with DL Model and Parameterization

Instead of optimizing 205 parameters individually, CMA-ES with DL Model and Parameterization only optimizes the coefficient vector (9 coefficients and a global scaling factor G). Specifically, this wrapper method:

1. Initialize in the same way as the forward-Euler-method-version CMA-ES and get the associated 9 linear coefficients and G
2. Optimize  the coefficient vector (9 coefficients and G)
3. For each coefficient vector, get the corresponding param vector using the group-level myelin and RSFC gradient. Then use the trained DL model to predict the costs of those param vectors.
4. Run up to 500 CMA-ES iterations, each yielding 1000 children
5. Choose 1 param vector with the best-predicted costs



![pMFM_speedup-DL_version_CMA-ES_with_parameterization](https://s2.loli.net/2022/11/14/gPfdKRNHY7olTFV.png)

Such a process is repeated with different random seeds to generate 200 param vectors. Unlike the previous wrappers, when the 200 params are fed into the pMFM with an SC from a **validation** subject group, almost all of the 200 params can generate simulated BOLD signals. Additionally, most of them have decent actual costs as shown in the figure below (showing only the total cost, can check `reports/slides/CBIG_20221005.pptx` for individual cost terms):

![image-20221114013656877](https://s2.loli.net/2022/11/14/3d7TOfgieFESK4D.png)

As we only want the best of the best, we are interested in those param vectors with the lowest total costs. To test the generalizability of those best param vectors, the top 20 param vectors with the lowest **actual** total costs are chosen and fed into the pMFM with an SC from a **test** subject group. The corresponding actual total costs are quite decent even with a different SC, as shown below:

![image-20221114014220315](https://s2.loli.net/2022/11/14/p3knQqO9IFNg52X.png)

This validates the usefulness of the DL model as the param vectors produced have decent actual costs. Note that the whole process (including trying with different random seeds) shown above would at most take several minutes. Whereas the original method (CMA-ES with pMFM) would take several hours to complete (~6.5 hours for 1 random seed). Hence, this method significantly speeds up the process of finding good param vectors for the pMFM.

## Limitation

This project mainly focuses on the HCP dataset and the effectiveness of this method is not tested on other datasets. If the DL model is not generalizable and one has to re-train the model with a new dataset, then this method is of little practical value. Hence, the generalizability of the DL model may be one of the directions for future work.



## Developer Guide

The Developer Guide covers the usage and organization of codes in this repository. For detailed descriptions of this project, please refer to the sections before the Developer Guide.

### Setup

Install the dependencies by creating a Conda environment:

```
conda env create -f config/pMFM_speedup_Tian_Fang.yml
conda activate pMFM_speedup-torch1.8
```

Change the `PATH_TO_PROJECT` variable under `src.basic.constants.py`, the current value is:

```python
PATH_TO_PROJECT = '/home/ftian/storage/pMFM_speedup/'
```

### Folder Structure

The folder structure for this repository is described as follows:

- `config`: contains yml files to setup the conda environment
- `dataset_generation`: contains code (python and MATLAB) for generating the dataset
- `notebooks`: contains Jupyter notebooks for quick visualization and debugging
- `reports`: contains figures, training logs, testing results, and slides
- `src`: contains the source code for building models, training, and testing
- `submit_to_scheduler`: contains Bash script for submitting the job to the scheduler

### Naming Conventions

In the source code:

- train/validation/test *sets* are often referred to as train/validation/test *splits*. Hence the `split_name` that appeared in the code refers to one of the `'train'`, `'validation'`, and `'test'`.
- The class `ParamPerformance` encapsulates a param vector along with its ground truth costs (FC_CORR, FC_L1, FCD_KS, and total cost). The 'Performance' in the class name is used instead of 'Costs' to avoid the confusion between the ground truth 'costs' and training/validation/testing 'costs'.

### Dataset Generation

#### Empirical Group-Level Information
*Note: Since the dataset is derived from the [HCP](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) dataset, the generated dataset is not uploaded here.*

The 1000 subjects from the [HCP](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) dataset are grouped based on `train_groups.csv`, `validation_groups.csv`, and `test_groups.csv`, where each row contains several subject IDs that define a subject group (you may define your own grouping). The MATLAB function `generate_inputs_wrapper` will generate group-level information for each of the subject groups, including group-level SC, FC, FCD, RSFC gradient, and myelin. The resulting empirical group-level information is stored in `dataset_generation/input_to_pMFM/[train OR validation OR test]/[group index within a set]` (e.g., the group-level information for the first subject group in the train set is stored in `dataset_generation/input_to_pMFM/train/1`).

#### Use CMA-ES with pMFM to Generate Parameters along with their Costs 

The `generate_parameters_for` function within `dataset_generation/dataset_generation.py` can be used to invoke calls to CMA-ES with pMFM which generates 10,000 parameters for each subject group along with those parameters' costs. Each subject group's parameters with costs are stored in subfolders of `dataset_generation/dataset`. For instance, the parameters with costs for the first subject group in the train set are stored in `dataset_generation/dataset/train/1/train_1.csv`. Each column of the CSV file represents a param vector with its costs, where the last 205 entries of a column represent the param vector and the first 4 entries of a column represent the associated costs (FC_CORR, FC_L1, FCD_KS, total cost, respectively).

### Dataset and Models

#### Dataset

The dataset structure is described as follows: Each train/validation/test set is represented by a `ParamDataset` consisting of multiple `SubjectGroup`s. Each `SubjectGroup` contains its group-level SC and 10000 `ParamPerformance`s that encapsulate param vectors along with their ground truth costs (FC_CORR, FC_L1, FCD_KS, and total cost). 

The Graph Neural Network (GNN) used is pytorch_geometric (PyG), which requires data instances to be stored in a `Data` class that represents nodes and edges. PyG also has its customized Dataset and Dataloader that are designed for the `Data` class. As a result, a `GnnParamDataset` is built specifically for the GCN model.

Additionally, you may use the `load_split` in `src/utils/init_utils.py` to load a specific dataset (e.g., train set). The function `load_split` will cache the dataset under `data/processed` which will speed up the second call to `load_split` significantly.

![pMFM_speedup-Code Structure](https://s2.loli.net/2022/11/14/kIOnFDG3fTAb9N7.png)

#### Models

The main framework used in this project to speed up code writing and model training is the PyTorch Lightning framework. All models are subclasses of the `PlModule` which is a subclass of PyTorch Lightning's LightningModule. The `PlModule` defines the common training elements across different models such as optimizer, loss function, and metric logging. The `Gcn` model implements the GCN and was trained on the `GnnParamDataset`. The `NaiveNet` is a simple MLP with several configurations, including whether to use coefficient vectors instead of param vectors and whether to use the SC feature extracted from `ExtractSCFeatMLP`. Note that a CNN version of the SC feature extractor is also provided.

### Model Training

The hyperparameter tuning framework used is Optuna which requires an 'objective' function to be optimized. Those 'objective' functions are written at the end of the models' source files. The actual scripts for running the model training and hyperparameter optimization are located at `src/training/training_script`, which simply call the `tune`/`tune_gnn` function (in the `src/utils/training_utils.py`) with corresponding 'objective' functions. Note that,  `tune`/`tune_gnn` will save a 'study.pkl' file in your specified save directory and you can simply resume tuning and try more sets of hyperparameters by running `tune`/`tune_gnn` again.

### Model Testing and Wrapper Functions

Under the `src/testing` folder, the `compare_top_k_params.py` compares the DL model's predictions and the actual costs as described in the section [Compare Prediction and Actual Costs](#compare-prediction-and-actual-costs). The `wrapper.py` contains several wrapper methods for [parameter optimization using the DL model](#parameter-optimization-using-the-dl-model). To use the recommended wrapper function (CMA-ES with DL Model and Parameterization), you can call the `cmaes_wrapper_use_coef` function in the `wrapper.py` with the group-level myelin and RSFC gradient. Moreover, you can configure the function with different random seeds, different DL models used, and different output directories. Furthermore, an optional `SC_path` may be specified if the DL model used requires SC.

There are other testing functions in `src/utils/test_utils.py` that may be helpful, such as `test_model` which tests a model's performance on a given dataset.

### Trained Model

The trained model optimized by Optuna can be found by calling functions in `src/testing/testing_lib/__init__.py` such as `load_naive_net_no_SC()` for the Naive Net without SC feature extractor.

## Bugs and Questions

Please contact Tian Fang at [tianfangsg@gmail.com](mailto:tianfangsg@gmail.com).
