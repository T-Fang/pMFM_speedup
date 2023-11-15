#!/bin/bash

ngpus='1'
ncpus='4'
mem='20G'
walltime='01:00:00'
name='wrapper'

function submit_job {
    split_name=$1
    idx_in_split=$2

    cmd_script="source deactivate; source activate pMFM_speedup-torch1.8; python /home/ftian/storage/pMFM_speedup/src/testing/wrapper.py ${split_name} ${idx_in_split}; source deactivate"

    job_path="/home/ftian/storage/pMFM_speedup/submit_to_scheduler/jobs/testing/wrapper/${split_name}_${idx_in_split}/"
    mkdir -p ${job_path}
    job_err="${job_path}${name}_error.txt"
    job_out="${job_path}${name}_out.txt"

    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd_script" -walltime $walltime -mem $mem -name $name -joberr $job_err -jobout $job_out -ngpus $ngpus -ncpus $ncpus
}

# for ((group_index = 2; group_index < 15; group_index++)); do
#     submit_job 'validation' ${group_index}
# done
submit_job 'validation' '1'

# for ((group_index = 2; group_index < 18; group_index++)); do
#     submit_job 'test' ${group_index}
# done
# submit_job 'test' '1'
