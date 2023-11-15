#!/bin/bash

ncpus='1'
mem='10G'
walltime='00:20:00'
name='wrapper'

function submit_job {

    cmd_script="source deactivate; source activate pMFM_speedup-torch1.8; python /home/ftian/storage/pMFM_speedup/src/testing/wrapper.py; source deactivate"

    job_path="/home/ftian/storage/pMFM_speedup/submit_to_scheduler/jobs/testing/wrapper/"
    mkdir -p ${job_path}
    job_err="${job_path}${name}_error.txt"
    job_out="${job_path}${name}_out.txt"

    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd_script" -walltime $walltime -mem $mem -name $name -joberr $job_err -jobout $job_out -ncpus $ncpus
}

submit_job
