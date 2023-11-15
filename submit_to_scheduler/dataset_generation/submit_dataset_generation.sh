#!/bin/bash

ngpus='1'
ncpus='4'
mem='40G'
name='dataset_generation'

function submit_job {
    split_name=$1
    batch_idx=$2
    n_batches=$3

    cmd_script="source deactivate; source activate pMFM_speedup; python /home/ftian/storage/pMFM_speedup/dataset_generation/dataset_generation.py ${split_name} ${batch_idx} ${n_batches}; source deactivate"

    job_path="/home/ftian/storage/pMFM_speedup/dataset_generation/job/${split_name}/batch_${batch_idx}/"
    mkdir -p ${job_path}
    job_err="${job_path}dataset_generation_error.txt"
    job_out="${job_path}dataset_generation_out.txt"

    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd_script" -walltime 80:00:00 -mem $mem -name $name -joberr $job_err -jobout $job_out -ngpus $ngpus -ncpus $ncpus
}

n_batches_train=10
n_batches_validation=2
n_batches_test=3

# Note: batch_idx is zero-based

for ((batch_idx = 0; batch_idx < $n_batches_train; batch_idx++)); do
    submit_job 'train' ${batch_idx} ${n_batches_train}
done

for ((batch_idx = 0; batch_idx < $n_batches_validation; batch_idx++)); do
    submit_job 'validation' ${batch_idx} ${n_batches_validation}
done

for ((batch_idx = 0; batch_idx < $n_batches_test; batch_idx++)); do
    submit_job 'test' ${batch_idx} ${n_batches_test}
done
