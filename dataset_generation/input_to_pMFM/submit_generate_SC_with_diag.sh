#!/bin/bash

ncpus='1'
mem='8G'
name='SC_with_diag'

job_err="/home/ftian/storage/pMFM_speedup/dataset_generation/input_to_pMFM/job/SC_with_diag_err.txt"
job_out="/home/ftian/storage/pMFM_speedup/dataset_generation/input_to_pMFM/job/SC_with_diag_out.txt"

MATLAB=$(which matlab)

cmd="$MATLAB -nosplash -nodisplay -nodesktop -r "
cmd="${cmd} \"addpath('/home/ftian/storage/pMFM_speedup/dataset_generation/input_to_pMFM'); generate_SC_with_diag_wrapper; quit;\" "

temp_script_file1="/home/ftian/storage/pMFM_speedup/dataset_generation/input_to_pMFM/job/temp_script.sh"
echo '#!/bin/bash ' >${temp_script_file1}
echo ${cmd} >>${temp_script_file1}
chmod 755 ${temp_script_file1}

$CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd ${temp_script_file1} -walltime 10:00:00 -mem $mem -name ${name} -joberr ${job_err} -jobout ${job_out} -ncpus ${ncpus}

exit 0
