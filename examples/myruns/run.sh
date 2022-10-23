#!/usr/bin/env bash

module load gcc/10.2.0-zuvaafu cuda/11.2.1-ni46jbc cudnn/8.1.1.33-11.2-px2tiqy
#######################################################################
# source /work/baskarg/bkhara/python_virtual_envs/lightning/bin/activate
# source activate /work/baskarg/bkhara/python_virtual_envs/tf-2.3
source /work/baskarg/bkhara/python_virtual_envs/tf-2.10/bin/activate
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export CUDA_VISIBLE_DEVICES="1"  # specify which GPU(s) to be used

echo "Launch time = `date +"%T"`"
echo "Working directory = ${PWD}"
#######################################################################
########## NO CHANGE ABOVE ############################################
#######################################################################

TRAIN_SCRIPT="poisson_2d_dirichlet.py"
time python ${TRAIN_SCRIPT} #>out_train.txt 2>&1
# time python -m pdb ${TRAIN_SCRIPT} #>out_train.txt 2>&1
