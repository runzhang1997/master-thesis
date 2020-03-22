# creat tenso file
# based on those tensor to do the shard
from __future__ import division
import numpy as np
from random import randrange
import os
import glob
import argparse
import tensorflow as tf
from normalize import normalize

def OpenDataFiles(path, run_60):
    if run_60:
        train_path = path + 'train60_96_new_degradation/'
    else:
        train_path = path + 'train20_32_interp_bilinear_L2A/'
    # Initialize in able to concatenate
    data20_gt = data60_gt = data10 = data20 = data60 = None
    #fileList = ['S2A_MSIL1C_20170616T223701_N0205_R072_T59GLL_20170616T223702.SAFE']
    fileList = [os.path.basename(x) for x in sorted(glob.glob(train_path + '*SAFE'))]
    for dset in fileList:
        data10_new = np.load(train_path + dset + '/data10.npy')
        data20_new = np.load(train_path + dset + '/data20.npy')
        data10 = np.concatenate((data10, data10_new)) if data10 is not None else data10_new
        data20 = np.concatenate((data20, data20_new)) if data20 is not None else data20_new
        if run_60:
            data60_gt_new = np.load(train_path + dset + '/data60_gt.npy')
            data60_new = np.load(train_path + dset + '/data60.npy')
            data60_gt = np.concatenate((data60_gt, data60_gt_new)) if data60_gt is not None else data60_gt_new
            data60 = np.concatenate((data60, data60_new)) if data60 is not None else data60_new
        else:
            data20_gt_new = np.load(train_path + dset + '/data20_gt.npy')
            data20_gt = np.concatenate((data20_gt, data20_gt_new)) if data20_gt is not None else data20_gt_new

    if run_60:
        return data10, data20, data60, data60_gt
    else:
        return data10, data20, data20_gt

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SupRes')
    parser.add_argument('--run_60', action='store_true', help='60 -> 10m / 20 -> 10m')
    args = parser.parse_args()
    path = '/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/'
    if args.run_60:
        train10, train20, train60, label = OpenDataFiles(path, True)
        train10 = np.moveaxis(train10,1,3)
        train20 = np.moveaxis(train20,1,3)
        train60 = np.moveaxis(train60,1,3)
        label = np.moveaxis(label,1,3) 
    else:
        train10, train20, label = OpenDataFiles(path, False)
        train10 = np.moveaxis(train10, 1, 3)
        train20 = np.moveaxis(train20, 1, 3)
        label = np.moveaxis(label, 1, 3)           
    
    
    print (np.mean(train10,axis=(0,1,2)))
    print (np.mean(train20,axis=(0,1,2)))
    print (np.mean(train60,axis=(0,1,2))) 
    
    print (np.std(train10,axis=(0,1,2)))
    print (np.std(train20,axis=(0,1,2)))
    print (np.std(train60,axis=(0,1,2)))
    
    print (np.mean(label, axis=(0,1,2)))
    print (np.std(label, axis=(0,1,2)))
      
    print(train10.shape)
    print(train20.shape)
    print(train60.shape)
    print(label.shape)
    
    #train10, train20, train60, label = normalize(patch_size=32, L2A=False, train10=train10, train20=train20, train60=train60, label=label, train=True, run_60=True) 
    train10 = train10/2000.0
    train20 = train20/2000.0
    train60 = train60/2000.0
    label = label/2000.0
    
    if args.run_60:
        #output_path = '/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/train60_band_channel_last_32_interp_stdnorm.tfrecords'
        output_path1 = '/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/train_60_20_10_32_new_degradation_divide_2000.tfrecords'
        output_path2 = '/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/train_60_32_new_degradation_divide_2000.tfrecords'
    else:
        output_path1 = '/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/train_10_20_32_interp_divide_2000_L2A.tfrecords'
        output_path2 = '/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/valid_20_32_interp_divide_2000_L2A.tfrecords'
        
    writer1=tf.python_io.TFRecordWriter(output_path1)
    writer2=tf.python_io.TFRecordWriter(output_path2)
    size = train10.shape[0]
    #size = 2
    idx = np.random.permutation(size)
    if args.run_60:
        train10, train20, train60, label = train10[idx], train20[idx], train60[idx], label[idx]
    else:
        train10, train20, label = train10[idx], train20[idx], label[idx]
    
    for i in range(0,size):
        if i % 10000 == 0:
            print (i)
        
        train_input1 = train10[i].tostring()
        train_input2 = train20[i].tostring()
        if args.run_60:
            train_input3 = train60[i].tostring()
        label_i = label[i].tostring()
        if args.run_60:
            example1 = tf.train.Example(features=tf.train.Features(feature={
                'train_input1': _bytes_feature(train_input1),
                'train_input2':_bytes_feature(train_input2),
                'train_input3':_bytes_feature(train_input3),
                'label':_bytes_feature(label_i)}))
            example2 = tf.train.Example(features=tf.train.Features(feature={
                'train_input3':_bytes_feature(train_input3),
                'label':_bytes_feature(label_i)}))
        else:
            example1 = tf.train.Example(features=tf.train.Features(feature={
                'train_input1': _bytes_feature(train_input1),
                'train_input2':_bytes_feature(train_input2),
                'label':_bytes_feature(label_i)}))
            
            example2 = tf.train.Example(features=tf.train.Features(feature={
                'train_input2':_bytes_feature(train_input2),
                'label':_bytes_feature(label_i)}))        
        
        '''
        train_input1 = train10[i].reshape(-1)
        train_input2 = train20[i].reshape(-1)
        if args.run_60:
            train_input3 = train60[i].reshape(-1)
        label_i = label[i].reshape(-1)        
        if i==1:
            print (label_i.shape)           
        if args.run_60:
            example = tf.train.Example(features=tf.train.Features(feature={
                'train_input1': _float_feature(train_input1),
                'train_input2':_float_feature(train_input2),
                'train_input3':_float_feature(train_input3),
                'label':_float_feature(label_i)}))
        else:
            example1 = tf.train.Example(features=tf.train.Features(feature={
                'train_input1': _float_feature(train_input1),
                'train_input2':_float_feature(train_input2),
                'label':_float_feature(label_i)}))
            
            example2 = tf.train.Example(features=tf.train.Features(feature={
                'train_input2':_float_feature(train_input2),
                'label':_float_feature(label_i)}))
        '''
        writer1.write(example1.SerializeToString())
        writer2.write(example2.SerializeToString())
        
    writer1.close()
    writer2.close()
    
