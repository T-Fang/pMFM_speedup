#!/bin/bash

ngpus='1'
ncpus='4'
mem='8G'
walltime='10:00:00'
name='simulated_FCD_with_diff_SC'

function submit_job {
    param_folder_name=$1

    cmd_script="source deactivate; source activate pMFM_speedup-torch1.8; cd /home/ftian/storage/pMFM_speedup/dataset_generation/; python -c \"from assumption_validation import simulate_all_with_diff_SC; simulate_all_with_diff_SC('${param_folder_name}')\"; source deactivate"

    job_path="/home/ftian/storage/pMFM_speedup/dataset_generation/job/simulated_FCD_with_diff_SC/${param_folder_name}/"
    mkdir -p ${job_path}
    job_err="${job_path}${name}_error.txt"
    job_out="${job_path}${name}_out.txt"

    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd_script" -walltime $walltime -mem $mem -name $name -joberr $job_err -jobout $job_out -ngpus $ngpus -ncpus $ncpus
}

# submit_job 'param_0'
# submit_job 'param_1'
# submit_job 'param_2'
# submit_job 'param_3'
submit_job 'param_4'
