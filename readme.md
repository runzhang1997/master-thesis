# creat dataset
cd gan/utils
bsub < creat_patches.sh



# train
cd gan
bsub < job_script_juron.sh
or
sbatch < job_script_juwels.sh


# test

bsub < predict_juron_tensorflow.sh
or
sbatch < predict-evaluate-tensorflow.sh