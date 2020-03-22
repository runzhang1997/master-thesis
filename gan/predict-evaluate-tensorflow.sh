#!/bin/bash
#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1 --partition=gpus
 
# Load the required modules
module use /usr/local/software/jureca/OtherStages
module load GCC/8.3.0  MVAPICH2/2.3.1-GDR

module load TensorFlow/1.13.1-GPU-Python-3.6.8
module load Horovod/0.16.2-GPU-Python-3.6.8
module load Keras/2.2.4-GPU-Python-3.6.8
module load h5py/2.9.0-Python-3.6.8
# Load your virtual environment 

source /p/project/joaiml/remote_sensing/run_zhang/.env_juwels/bin/activate

network='residual_dense_gan_tensorflow'
model_name='/p/project/joaiml/remote_sensing/run_zhang/sen2_pansharpening/gan/network_data/run_60_pretrain_60m_4GPU_degradation/pretrain-115654'
test_dir='/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/test60_96_interp/'
#test_dir='/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/test20_32_interp_bilinear_L2A/'
#test_dir='/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/test20_32_interp_bilinear/'
test_one='S2A_MSIL1C_20161230T074322_N0204_R092_T37NCE_20161230T075722.SAFE'  
prediction_file='pretrain_60m_new_degra'
batch_size=32 
python $network'/'predict-tensorflow.py \
              --predict_data $test_dir --model $model_name --patch_size 96 --batch_size $batch_size  --run_60 \
			   --real \
			   --save --prediction_file $prediction_file \
			  #--real \
			  #
