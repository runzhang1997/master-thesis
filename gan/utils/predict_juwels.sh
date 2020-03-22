#!/bin/bash -x
#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 --partition=develgpus

model_name='/p/project/joaiml/remote_sensing/run_zhang/gan/network_data/run_20_wassersteinGAN/weights-400'
test_dir='../../DSen2/data/test32_no_interp/'

module use /usr/local/software/jureca/OtherStages
module load GCC/8.3.0  MVAPICH2/2.3.1-GDR

module load TensorFlow/1.13.1-GPU-Python-3.6.8
module load Horovod/0.16.2-GPU-Python-3.6.8
module load Keras/2.2.4-GPU-Python-3.6.8

source /p/project/joaiml/remote_sensing/run_zhang/.env_juwels/bin/activate
python predict-tensorflow.py --predict_data $test_dir --model $model_name --prediction_file 'wassergan'