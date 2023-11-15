#!/bin/bash

ngpus='1'
ncpus='4'
mem='30G'
walltime='28:00:00'
name='dataset_generation'

function submit_job {
    split_name=$1
    idx_in_split=$2

    cmd_script="source deactivate; source activate pMFM_speedup; python /home/ftian/storage/pMFM_speedup/dataset_generation/single_dataset_generation.py ${split_name} ${idx_in_split}; source deactivate"

    job_path="/home/ftian/storage/pMFM_speedup/dataset_generation/job/${split_name}/${idx_in_split}/"
    mkdir -p ${job_path}
    job_err="${job_path}dataset_generation_error.txt"
    job_out="${job_path}dataset_generation_out.txt"

    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd_script" -walltime $walltime -mem $mem -name $name -joberr $job_err -jobout $job_out -ngpus $ngpus -ncpus $ncpus
}

submit_job 'train' '10'
submit_job 'train' '45'
submit_job 'test' '14'
