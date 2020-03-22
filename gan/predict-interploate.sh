#!/usr/bin/env bash
#test_dir='Set14/' 
#scale=6
#interp='bicubic'
#interp='bilinear'
#interp='nearest'
#interp='lanczos4'

#test_dir='test20_32_interp_bilinear_L2A/'
test_one='S2A_MSIL1C_20170616T223701_N0205_R072_T59GLL_20170616T223702.SAFE'
network='residual_dense_gan_tensorflow'
test_dir='test60_192/' 
#for test_dir in 'test60_no_interp/'
#do
#echo $test_dir 6 $interp
#python predict-interploate.py --predict_data $test_dir \
#--run_60 --interp $interp
#done
#predication_file='bicubicInterp_real_degraded'
predication_file='bicubic_interp'
#for interp in 'bicubic' 
#do
#    python $network'/'predict-interploate.py --predict_data $test_dir --interp $interp --save --predication_file $predication_file
#done
#python $network'/'predict-interploate.py --predict_data $test_dir --interp 'bicubic'
python $network'/'predict-interploate-original.py --predict_data $test_dir --interp 'bicubic' --save --predication_file $predication_file --run_60