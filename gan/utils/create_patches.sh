#!/usr/bin/env bash
#BSUB -q normal
#BSUB -W 24:00
#BSUB -n 1
#BSUB -e "error.%J.er"
#BSUB -o "output_%J.out"
#BSUB -J generate_lanczos_test
module --force purge
# Load the required modules
module load gcc/8.1.0
module load python/3.6.1 
module load tensorflow/1.12.0-gcc_5.4.0-cuda_10.0.130
module load keras/2.2.4
module load scikit-learn/0.20.2 
module load h5py/2.9.0 
module load matplotlib/2.1.0 
module load pillow/5.0.0
module load horovod/0.15.2
module load gdal/2.4.1
module load scikit-image/0.14.2
module load numpy/1.16.0
module load opencv/3.4.5-gcc_5.4.0-cuda_10.0.130
module load opencv-python
source /p/project/joaiml/remote_sensing/run_zhang/.env_juron/bin/activate

# Run the program
#for file in ../../sentinel_data/test_data/unzip/*;do
#        #echo "$file"
#        python create_patches.py $file/ --test_data --run_60
#done

# after this one try use interp=false.
for file in ../../sentinel_data/test_data/unzip/*;do
        #echo "$file"
        python create_patches.py $file/ --test_data --run_60
done
