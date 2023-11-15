#!/bin/bash

ngpus='1'
ncpus='4'
mem='8G'
walltime='15:00:00'
name='simulated_FCD_with_diff_params'

function submit_job {
    split_name=$1
    group_index=$2
    cmd_script="source deactivate; source activate pMFM_speedup-torch1.8; cd /home/ftian/storage/pMFM_speedup/dataset_generation/; python -c \"from assumption_validation import simulate_with_diff_param; simulate_with_diff_param('${split_name}', '${group_index}')\"; source deactivate"

    job_path="/home/ftian/storage/pMFM_speedup/dataset_generation/job/simulated_FCD_with_diff_params/${split_name}/${group_index}"
    mkdir -p ${job_path}
    job_err="${job_path}${name}_error.txt"
    job_out="${job_path}${name}_out.txt"

    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd_script" -walltime $walltime -mem $mem -name $name -joberr $job_err -jobout $job_out -ngpus $ngpus -ncpus $ncpus
}

submit_job 'train' '10'
# submit_job 'train' '25'
# submit_job 'train' '36'
# submit_job 'validation' '7'
# submit_job 'validation' '13'
