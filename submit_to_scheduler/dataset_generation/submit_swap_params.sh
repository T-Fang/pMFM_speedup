#!/bin/bash

ngpus='1'
ncpus='4'
mem='35G'
walltime='3:00:00'
name='swap_params'

function submit_job {
    split_name=$1
    idx_in_split=$2

    cmd_script="source deactivate; source activate pMFM_speedup-torch1.8; python /home/ftian/storage/pMFM_speedup/dataset_generation/assumption_validation.py ${split_name} ${idx_in_split}; source deactivate"

    job_path="/home/ftian/storage/pMFM_speedup/dataset_generation/job/swap_params/use_train_1_params/${split_name}_${idx_in_split}/"
    mkdir -p ${job_path}
    job_err="${job_path}swap_params_error.txt"
    job_out="${job_path}swap_params_out.txt"

    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd_script" -walltime $walltime -mem $mem -name $name -joberr $job_err -jobout $job_out -ngpus $ngpus -ncpus $ncpus
}

submit_job 'train' '2'
submit_job 'train' '3'
submit_job 'train' '4'
submit_job 'train' '5'
submit_job 'validation' '1'
submit_job 'validation' '5'
submit_job 'validation' '10'
submit_job 'test' '1'
submit_job 'test' '5'
submit_job 'test' '10'
