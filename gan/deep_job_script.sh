#!/bin/bash
#SBATCH --partition=dp-dam
#SBATCH -N 8
#SBATCH --tasks-per-node 1
#SBATCH -o deep-test-out.%j
#SBATCH -e deep-test-err.%j
#SBATCH --time=12:00:00

batch_size=128
epoch=1200
number_of_patches=360000
patch_size=32
#train_data='../DSen2/data/train_no_interp_32_patches_channel_last.tfrecords'
train_data='../../DSen2/data/train20_32_shuffel_channel_last_interp_bilinear_stdnorm.tfrecords'
network='residual_dense_gan_tensorflow'
network_data='dense_norm'
#network='wassersteinGAN'
# 0.0006 to 0.0003
generator_learning_rate=0.0002
discrimator_learning_rate=0.0006
gantype='wgan-gp'
checkpoints='network_data/run_20_dense_norm/'
contloss='L1'
patch_2=32

module load GCC/8.3.0 ParaStationMPI/5.4.0-1-CUDA
module load TensorFlow/1.13.1-GPU-Python-3.6.8
module load Horovod/0.16.2-GPU-Python-3.6.8
module load Keras/2.2.4-GPU-Python-3.6.8
module load scikit/2019a-Python-3.6.8
#module load mpi4py/3.0.1-Python-3.6.8

source /p/project/joaiml/remote_sensing/run_zhang/.env_deep/bin/activate

# nvprof --analysis-metrics -o gpu_performance%p
srun --cpu_bind=none \
        python $network"/"train.py \
       --patch_size $patch_size --contentloss $contloss\
       --epochs $epoch --batch_size $batch_size \
       --number_of_patches $number_of_patches --norm \
       --train_data $train_data --patch_2 $patch_2 --pretrain_generator \
       --generator_learning_rate $generator_learning_rate --discrimator_learning_rate $discrimator_learning_rate --network_data $network_data --gan_type $gantype \
       #--load --checkpoints $checkpoints
	    
