module --force purge;
module use /usr/local/software/jureca/OtherStages
module load GCC/8.3.0  
module --ignore-cache load MVAPICH2/2.3.2-GDR

module load TensorFlow/1.13.1-GPU-Python-3.6.8
module load Horovod/0.16.2-GPU-Python-3.6.8
module load Keras/2.2.4-GPU-Python-3.6.8
module load CUDA/10.1.105

# Use this only the first time
#python3 -m venv /p/project/joaiml/remote_sensing/run_zhang/.env_juwels 

# Use this every time for activate the environment
source /p/project/joaiml/remote_sensing/run_zhang/.env_deep/bin/activate
