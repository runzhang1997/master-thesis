#!/bin/bash
#SBATCH --account=cstdl
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --output=ganout1.%j  
#SBATCH --error=ganerr1.%j
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4 --partition=develgpus
# patch 256 128 192 96 32 intepolation to the same.
# patch 256 128 192 96 32 intepolation to the same.
batch_size=128
epoch=2000
patch_size=32
#train_data='../../DSen2/data/train_60_20_10_32_interp_divide_2000.tfrecords'
#train_data='../../DSen2/data/train_10_20_32_interp_divide_2000.tfrecords' 
train_data='../../DSen2/data/train_10_20_32_interp_divide_2000_L2A.tfrecords'
valid_data='../../DSen2/data/valid_10_20.tfrecords'
patch_1=32
patch_2=32
patch_3=32
shuffel=20000
network='residual_dense_gan_tensorflow'
#network_data='pretrain_20m_l2A_8GPU_wgan'
#network_data='pretrain_20m_l2A_8GPU_hinge'
network_data='pretrain_20m_l2A_8GPU_rela'
#network_data='pretrain_20m_l2A_8GPU_vanila' 
#network='wassersteinGAN'
#0.0006 to 0.0003
#0.0002, 0.0001, 0.0001, 0.00005, 0.00005, 0.00005 0.00002, 0.00002 0.00001
generator_learning_rate=0.0001
discrimator_learning_rate=0.0002 
adv_weight=0.0002 
#gantype='wgan-gp'
#gantype='Vanila-gan'
#gantype='hinge'
gantype='relativistic'
checkpoints='network_data/run_20_pretrain_20m_bright_skip_l2a_learning_rate_4GPU/'
contloss='L1'
# Load the required modules
module use /usr/local/software/jureca/OtherStages
module load GCC/8.3.0  MVAPICH2/2.3.2-GDR
module load TensorFlow/1.13.1-GPU-Python-3.6.8
module load Horovod/0.16.2-GPU-Python-3.6.8
module load Keras/2.2.4-GPU-Python-3.6.8
#module load h5py/2.9.0-Python-3.6.8
module load CUDA/10.1.105
# Load your virtual environment 
source /p/project/joaiml/remote_sensing/run_zhang/.env_juwels/bin/activate
#--pretrain_generator
# nvprof --analysis-metrics --export-profile gpu_performance
# nvprof --analysis-metrics -o gpu_performance%p
srun python $network"/"train.py \
       --patch_size $patch_size --contentloss $contloss --validation_data $valid_data \
       --epochs $epoch --batch_size $batch_size --shuffel $shuffel \
       --train_data $train_data --patch_1 $patch_1 --patch_2 $patch_2 \
	   --patch_3 $patch_3 --kd 1 --adv_weight $adv_weight \
       --generator_learning_rate $generator_learning_rate \
	   --discrimator_learning_rate $discrimator_learning_rate \
	   --network_data $network_data --gan_type $gantype --norm --load --checkpoints $checkpoints
	   #--pretrain_generator \
