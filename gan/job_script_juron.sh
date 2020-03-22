#!/usr/bin/env bash
#BSUB -q normal
#BSUB -W 24:00
#BSUB -R "span[ptile=4]"
#BSUB -n 4
#BSUB -gpu "num=4"
#BSUB -J brightness_4
#BSUB -e "error-%J-er"
#BSUB -o "output-%J-ou"
#network_data='pretrain_60m_4GPU_learning_rate_40000'
#network_data='pretrain_60m_2GPU_learning_rate_80000'
#network_data='pretrain_60m_1GPU_learning_rate_160000'
#network_data='pretrain_60m_8GPU_learning_rate_20000'
#network_data='pretrain_60m_16GPU_learning_rate_6000'
#network_data='pretrain_20m_8GPU_learning_rate_16000'
#network_data='pretrain_20m_16GPU_learning_rate_6000'
#network_data='pretrain_20m_4GPU_learning_rate_32000'
#network_data='pretrain_20m_2GPU_learning_rate_64000'
#network_data='pretrain_20m_1GPU_learning_rate_128000'

network_data='pretrain_60m_4GPU_degradation'
generator_learning_rate=0.0004
discrimator_learning_rate=0.0002
checkpoints='network_data/run_20_pretrain_20m_bright_skip_l2a_learning_rate_4GPU/'
# patch 256 128 192 96 32 intepolation to the same.
batch_size=32
epoch=4000
patch_size=96
train_data='../../DSen2/data/train_60_20_10_32_new_degradation_divide_2000.tfrecords'
#train_data='../../DSen2/data/train_10_20_32_interp_divide_2000.tfrecords' 
#train_data='../../DSen2/data/train_10_20_32_interp_divide_2000_L2A.tfrecords'
valid_data='../../DSen2/data/valid_10_20.tfrecords'
#train_data='../../DSen2/data/train10_20_2000_channel_last_64_32.tfrecords'
#train_data='../../DSen2/data/train20_band_channel_last_128_interp_stdnorm.tfrecords'
patch_1=96
patch_2=96
patch_3=96
shuffel=2000
network='residual_dense_gan_tensorflow'
#network='wassersteinGAN'
adv_weight=0.0002
#gantype='wgan-gp'
#gantype='Vanila-gan'
gantype='hinge'
#gantype='relativistic'
contloss='L1'
# Load the required modules
module load gcc/8.1.0
module load python/3.6.1
module load tensorflow/1.12.0-gcc_5.4.0-cuda_10.0.130
module load keras/2.2.4
module load scikit-learn/0.20.2
module load h5py/2.9.0
module load matplotlib/2.1.0
module load pillow/5.0.0
module load opencv/3.4.5-gcc_5.4.0-cuda_10.0.130
module load horovod/0.15.2
module load gdal/2.4.1
module load scikit-image/0.14.2
module load numpy/1.16.0
module load cuda/10.0.130
# Load your virtual environment #
source /p/project/joaiml/remote_sensing/run_zhang/.env_juron/bin/activate
#--validation_data $valid_data --valid
# Run the program
#mpirun python train_distribute.py
#mpirun -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH \
#        -x PATH -mca pml ob1 -mca btl ^openib python -u train_tfrecords_new_juron.py
#
#       -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib 
#-bind-to none -map-by slot -mca pml ob1 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x NCCL_IB_CUDA_SUPPORT=1 -x NCCL_IB_DISABLE=0 -x PATH  
#nvprof --analysis-metrics --export-profile gpu_performance
# -x HOROVOD_TIMELINE=timeline.json -x HOROVOD_TIMELINE_MARK_CYCLES=1 
mpirun -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x HOROVOD_MPI_THREADS_DISABLE=1 -x PATH -mca pml ob1 \
	python $network"/"train.py \
       --patch_size $patch_size --contentloss $contloss --validation_data $valid_data \
       --epochs $epoch --batch_size $batch_size --shuffel $shuffel \
       --train_data $train_data --patch_1 $patch_1 --patch_2 $patch_2 \
	   --patch_3 $patch_3 --kd 1 --adv_weight $adv_weight \
       --generator_learning_rate $generator_learning_rate \
	   --discrimator_learning_rate $discrimator_learning_rate \
	   --network_data $network_data --gan_type $gantype --run_60 \
	   --pretrain_generator \
	   #--load --checkpoints $checkpoints  
