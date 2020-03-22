# creat tenso file
# based on those tensor to do the shard
from __future__ import division
import numpy as np
from random import randrange
import os
import glob
import argparse
import tensorflow as tf

def OpenDataFiles(path, run_60):
    if run_60:
        train_path = path + 'train60_128/'
    else:
        train_path = path + 'train20_128_L2A/'
    # Initialize in able to concatenate
    data20_gt = data60_gt = data10 = data20 = data60 = None
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
             
    '''
    # 32
    if args.run_60:
        train10[:,:,:,0]=(train10[:,:,:,0]-1302)/1491
        train10[:,:,:,1]=(train10[:,:,:,1]-1306)/1293
        train10[:,:,:,2]=(train10[:,:,:,2]-1400)/1348
        train10[:,:,:,3]=(train10[:,:,:,3]-2350)/1416

        train20[:,:,:,0]=(train20[:,:,:,0]-1505)/1441
        train20[:,:,:,1]=(train20[:,:,:,1]-2122)/1400
        train20[:,:,:,2]=(train20[:,:,:,2]-2428)/1447
        train20[:,:,:,3]=(train20[:,:,:,3]-2634)/1471
        train20[:,:,:,4]=(train20[:,:,:,4]-1985)/1199
        train20[:,:,:,5]=(train20[:,:,:,5]-1298)/1001
    
        train60[:,:,:,0]=(train20[:,:,:,0]-1627)/1297
        train60[:,:,:,1]=(train20[:,:,:,1]-821)/702

        label[:,:,:,0]=(label[:,:,:,0]-1627)/1297
        label[:,:,:,1]=(label[:,:,:,1]-821)/702
    else:
        train10[:,:,:,0]=(train10[:,:,:,0]-1311)/1512
        train10[:,:,:,1]=(train10[:,:,:,1]-1313)/1309
        train10[:,:,:,2]=(train10[:,:,:,2]-1407)/1363
        train10[:,:,:,3]=(train10[:,:,:,3]-2355)/1443

        train20[:,:,:,0]=(train20[:,:,:,0]-1515)/1486
        train20[:,:,:,1]=(train20[:,:,:,1]-2128)/1457
        train20[:,:,:,2]=(train20[:,:,:,2]-2434)/1514
        train20[:,:,:,3]=(train20[:,:,:,3]-2639)/1546
        train20[:,:,:,4]=(train20[:,:,:,4]-1991)/1260
        train20[:,:,:,5]=(train20[:,:,:,5]-1304)/1050

        label[:,:,:,0]=(label[:,:,:,0]-1515)/1486
        label[:,:,:,1]=(label[:,:,:,1]-2128)/1457
        label[:,:,:,2]=(label[:,:,:,2]-2434)/1514
        label[:,:,:,3]=(label[:,:,:,3]-2639)/1546
        label[:,:,:,4]=(label[:,:,:,4]-1991)/1260
        label[:,:,:,5]=(label[:,:,:,5]-1304)/1050
    '''
    '''
    # 128
    if args.run_60:
        train10[:,:,:,0]=(train10[:,:,:,0]-1306)/1493
        train10[:,:,:,1]=(train10[:,:,:,1]-1309)/1293
        train10[:,:,:,2]=(train10[:,:,:,2]-1403)/1349
        train10[:,:,:,3]=(train10[:,:,:,3]-2361)/1414

        train20[:,:,:,0]=(train20[:,:,:,0]-1511)/1443
        train20[:,:,:,1]=(train20[:,:,:,1]-2131)/1400
        train20[:,:,:,2]=(train20[:,:,:,2]-2439)/1446
        train20[:,:,:,3]=(train20[:,:,:,3]-2646)/1470
        train20[:,:,:,4]=(train20[:,:,:,4]-1994)/1200
        train20[:,:,:,5]=(train20[:,:,:,5]-1304)/1003
    
        train60[:,:,:,0]=(train20[:,:,:,0]-1629)/1300
        train60[:,:,:,1]=(train20[:,:,:,1]-825)/701

        label[:,:,:,0]=(label[:,:,:,0]-1629)/1300
        label[:,:,:,1]=(label[:,:,:,1]-825)/701
    else:
        # 128 patches 20m 27000
        train10[:,:,:,0]=(train10[:,:,:,0]-1306)/1501
        train10[:,:,:,1]=(train10[:,:,:,1]-1309)/1300
        train10[:,:,:,2]=(train10[:,:,:,2]-1403)/1354
        train10[:,:,:,3]=(train10[:,:,:,3]-2353)/1433

        train20[:,:,:,0]=(train20[:,:,:,0]-1509)/1462
        train20[:,:,:,1]=(train20[:,:,:,1]-2124)/1427
        train20[:,:,:,2]=(train20[:,:,:,2]-2430)/1480
        train20[:,:,:,3]=(train20[:,:,:,3]-2636)/1509
        train20[:,:,:,4]=(train20[:,:,:,4]-1991)/1236
        train20[:,:,:,5]=(train20[:,:,:,5]-1304)/1031
    
        label[:,:,:,0]=(label[:,:,:,0]-1509)/1462
        label[:,:,:,1]=(label[:,:,:,1]-2124)/1427
        label[:,:,:,2]=(label[:,:,:,2]-2430)/1480
        label[:,:,:,3]=(label[:,:,:,3]-2636)/1509
        label[:,:,:,4]=(label[:,:,:,4]-1991)/1236
        label[:,:,:,5]=(label[:,:,:,5]-1304)/1031
    '''
    # 128 L2A
    if args.run_60:
        train10[:,:,:,0]=(train10[:,:,:,0]-1306)/1501
    else:
        # 128 patches 20m 27000
        train10[:,:,:,0]=(train10[:,:,:,0]-1273)/1777
        train10[:,:,:,1]=(train10[:,:,:,1]-1171)/1677
        train10[:,:,:,2]=(train10[:,:,:,2]-911)/1672
        train10[:,:,:,3]=(train10[:,:,:,3]-2607)/1694

        train20[:,:,:,0]=(train20[:,:,:,0]-1615)/1749
        train20[:,:,:,1]=(train20[:,:,:,1]-2259)/1641
        train20[:,:,:,2]=(train20[:,:,:,2]-2505)/1633
        train20[:,:,:,3]=(train20[:,:,:,3]-2680)/1621
        train20[:,:,:,4]=(train20[:,:,:,4]-2192)/1376
        train20[:,:,:,5]=(train20[:,:,:,5]-1571)/1240
    
        label[:,:,:,0]=(label[:,:,:,0]-1615)/1749
        label[:,:,:,1]=(label[:,:,:,1]-2259)/1641
        label[:,:,:,2]=(label[:,:,:,2]-2505)/1633
        label[:,:,:,3]=(label[:,:,:,3]-2680)/1621
        label[:,:,:,4]=(label[:,:,:,4]-2192)/1376
        label[:,:,:,5]=(label[:,:,:,5]-1571)/1240
    '''    
    print (np.mean(train10,axis=(0,1,2)))
    print (np.mean(train20,axis=(0,1,2)))
    #print (np.mean(train60,axis=(0,1,2)))
    
    print (np.std(train10,axis=(0,1,2)))
    print (np.std(train20,axis=(0,1,2)))
    #print (np.std(train60,axis=(0,1,2)))
    
    print (np.mean(label, axis=(0,1,2)))
    print (np.std(label, axis=(0,1,2)))
      
    print(train10.shape)
    print(train20.shape)
    #print(train60.shape)
    print(label.shape)
    
    '''
    if args.run_60:
        output_path = '/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/train60_band_channel_last_128_interp_stdnorm.tfrecords'
    else:
        output_path = '/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/train20_L2A_band_channel_last_128_interp_stdnorm.tfrecords'
    writer=tf.python_io.TFRecordWriter(output_path)
    size = train10.shape[0]
    #size = 2
    idx = np.random.permutation(size)
    if args.run_60:
        train10, train20, train60, label = train10[idx], train20[idx], train60[idx], label[idx]
    else:
        train10, train20, label = train10[idx], train20[idx], label[idx]
    
    for i in range(0,size):
        if i % 100 == 0:
            print (i)
        train_input1 = train10[i].tostring()
        train_input2 = train20[i].tostring()
        if args.run_60:
            train_input3 = train60[i].tostring()
        label_i = label[i].tostring()
        if args.run_60:
            example = tf.train.Example(features=tf.train.Features(feature={
                'train_input1': _bytes_feature(train_input1),
                'train_input2':_bytes_feature(train_input2),
                'train_input3':_bytes_feature(train_input3),
                'label':_bytes_feature(label_i)}))
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
                'train_input1': _bytes_feature(train_input1),
                'train_input2':_bytes_feature(train_input2),
                'label':_bytes_feature(label_i)}))

        writer.write(example.SerializeToString())
    writer.close()