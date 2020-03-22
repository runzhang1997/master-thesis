#!/usr/bin/env bash
#BSUB -q normal
#BSUB -W 24:00
#BSUB -n 1
#BSUB -e "err_%J.er"
#BSUB -o "out_%J.out"
#BSUB -J sentinel-2

# Load the required modules
module --force purge
module load gcc/8.1.0
module load python/3.6.1
module load gdal/2.4.1
module load tensorflow/1.12.0-gcc_5.4.0-cuda_10.0.130
module load keras/2.2.4
module load python/3.6.1
module load scikit-learn/0.20.2
module load h5py/2.9.0
module load matplotlib/2.1.0
module load pillow/5.0.0
module load opencv/3.4.5-gcc_5.4.0-cuda_10.0.130
module load scikit-image/0.14.2
module load numpy/1.16.0
#nvidia-smi
# Load your virtual environment 
source /p/project/joaiml/remote_sensing/run_zhang/.env_juron/bin/activate

# Run the program
#for file in ../sentinel_data/training_data/unzip/*;do
#        #echo "$file"
#        python create_training_patches.py $file/ --save_prefix '../DSen2/data/'
#done

#python convert_to_tensor.py
for file in ../../sentinel_data/test_data/unzip/*; do
    #echo "$file"
    python create_training_patches.py $file/ --test_data --save_prefix '../../DSen2/data/'
done

  
