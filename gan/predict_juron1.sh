#!/usr/bin/env bash
#BSUB -q normal
#BSUB -W 01:00
#BSUB -n 1
#BSUB -R "span[ptile=4]"
#BSUB -gpu "num=1"
#BSUB -e "error-%J-er"
#BSUB -o "output-%J-ou"
#BSUB -J test_64

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

network='residual_dense_gan_tensorflow'
model_name='/p/project/joaiml/remote_sensing/run_zhang/sen2_pansharpening/gan/network_data/run_20_dense_dsen2_self_2skip_20_64_32_02/'
#test_dir='/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/test20_16_no_interp/'
test_dir='/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/test20_64_32/'
#test_dir='/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/test20_32_interp_bilinear/'
test_one='S2A_MSIL1C_20170616T223701_N0205_R072_T59GLL_20170616T223702.SAFE'  
prediction_file='self_skip_subpixel'
batch_size=128
python $network'/'predict-tensorflow.py \
              --predict_data $test_dir --model $model_name --patch_size 32 --batch_size $batch_size --save
