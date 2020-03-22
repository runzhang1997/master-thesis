#!/bin/bash
#SBATCH --partition=dp-dam
#SBATCH -N 1
#SBATCH --tasks-per-node 1
#SBATCH -o deep-test-out.%j
#SBATCH -e deep-test-err.%j 
#SBATCH --time=00:40:00

module load GCC/8.3.0 ParaStationMPI/5.4.0-1-CUDA
module load TensorFlow/1.13.1-GPU-Python-3.6.8
module load Horovod/0.16.2-GPU-Python-3.6.8
module load Keras/2.2.4-GPU-Python-3.6.8
module load scikit/2019a-Python-3.6.8
#module load mpi4py/3.0.1-Python-3.6.8

source /p/project/joaiml/remote_sensing/run_zhang/.env_deep/bin/activate

network='residual_dense_gan_tensorflow'
model_name='/p/project/joaiml/remote_sensing/run_zhang/sen2_pansharpening/gan/network_data/run_20_dense_dsen2_self_2skip_20_64_32_02/'
#test_dir='/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/test20_16_no_interp/'
test_dir='/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/test20_64_32/'
#test_dir='/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/test20_32_interp_bilinear/'
test_one='S2A_MSIL1C_20170616T223701_N0205_R072_T59GLL_20170616T223702.SAFE'  
prediction_file='self_skip_subpixel'
batch_size=128
python $network'/'predict-tensorflow.py \
              --predict_data $test_dir --model $model_name --patch_size 32 --batch_size $batch_size
