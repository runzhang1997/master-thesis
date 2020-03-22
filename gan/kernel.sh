#!/bin/bash
# bsub -gpu "num=4" -Is -tty /bin/bash
module --force purge
module load gcc/8.1.0
module load python/3.6.1 
module load tensorflow/1.12.0-gcc_5.4.0-cuda_10.0.130
#module load tensorflow/1.7.0-gcc_5.4.0-cuda_9.1.85
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
module load cuda/10.0.130
source /p/project/joaiml/remote_sensing/run_zhang/.env_juron/bin/activate
#export PYTHONPATH=/p/project/joaiml/remote_sensing/run_zhang/.env_juron/lib/python3.6/site-packages:${PYTHONPATH}


