import tensorflow as tf
import numpy as np
import argparse
import network
import os
import sys
import time
import horovod.tensorflow as hvd
#from tensorflow.python.client import timeline


if __name__ == '__main__':

    def decode(serialized_example):
        """decode the serialized example"""   
        patch_1 = args.patch_1
        patch_2 = args.patch_2
        
        features = tf.parse_single_example(serialized_example,
                                       features={
                                           'train_input1': tf.FixedLenFeature([], tf.string),
                                           'train_input2': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.string)
                                       })
                                       
        train_input_10 = tf.decode_raw(features['train_input1'], tf.float32)
        train_input_10 = tf.reshape(train_input_10, [patch_1, patch_1, 4])
        train_input_20 = tf.decode_raw(features['train_input2'], tf.float32)
        train_input_20 = tf.reshape(train_input_20, [patch_2, patch_2, 6])
        label = tf.decode_raw(features['label'], tf.float32)
        label = tf.reshape(label, [patch_1, patch_1, 6])
        
        
        
    tf.reset_default_graph() 
      
    dataset = tf.data.TFRecordDataset(args.train_data)
    dataset = dataset.shard(hvd.size(), hvd.rank())
    dataset = dataset.repeat(args.epochs)  
    #dataset = dataset.cache()
    dataset = dataset.shuffle(args.shuffel) # shuffle
    dataset = dataset.map(decode)
    #dataset = dataset.apply(tf.contrib.data.map_and_batch(decode, batch_size, drop_remainder=True))
    dataset = dataset.batch(args.batch_size,drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=1)
    iterator = dataset.make_one_shot_iterator()
    train10, train20, label = iterator.get_next()
    
    
    print (train10.shape)
            #break
            prediction = 0
            interval = args.batch_size
            bound = np.ceil(train10.shape[0]/interval)
            print (bound)
            print (type(bound))
            #interval = 4/529
            for i in range(0,int(bound)):
                if (i+1)*interval > train10.shape[0]:
                    remain = train10.shape[0] - interval*i
                    print (remain)
                    if run_60:
                        prediction_i = sess.run(g_y_pred, feed_dict={input_1: train10[i*interval:train10.shape[0]],
                                                                     input_2: train20[i*interval:train10.shape[0]],
                                                                     input_3: train60[i*interval:train10.shape[0]],
                                                                     batch_length: remain
                                                                     })
                    else:
                        prediction_i = sess.run(g_y_pred, feed_dict={input_1: train10[i*interval:train10.shape[0]],
                                                                     input_2: train20[i*interval:train10.shape[0]],
                                                                     batch_length: remain
                                                                     })
                else:
                    if run_60:
                        prediction_i = sess.run(g_y_pred, feed_dict={input_1: train10[i*interval:(i+1)*interval],
                                                                     input_2: train20[i*interval:(i+1)*interval],
                                                                     input_3: train60[i*interval:(i+1)*interval],
                                                                     batch_length: interval})
                    else:
                        prediction_i = sess.run(g_y_pred, feed_dict={input_1: train10[i*interval:(i+1)*interval],
                                                                     input_2: train20[i*interval:(i+1)*interval],
                                                                     batch_length:interval})              
                
                if i ==0:
                    prediction = prediction_i
                else:
                    prediction = np.append(prediction,prediction_i, axis=0)

            prediction = np.moveaxis(prediction,3,1)
            
            print (prediction.shape)
            images = recompose_images(prediction, border=border, size=image_size)
            images = images * 2000.0
            #images = reverse_normalize(images, run_60=run_60, L2A=L2A, patch_size=32)
            #images = images * 1470 + 2000
            images[images<0] = 1
            end = time.time()
            print('Elapsed time: {}.'.format(end - start))
            
            gt = np.load(args.predict_data +dset+'/no_tiling/data20_gt.npy')
            print (np.mean(images)-np.mean(gt))
            
            RMSE_err.append(RMSE(images, gt))
            SRE_err.append(SRE(images, gt))
            ERGAS_err.append(ERGAS(images, gt, 2))
            SAM_err.append(SAM(images, gt))
            psnr.append(compare_psnr(gt,images, data_range=np.max(gt)))
            ssim.append(compare_ssim(gt,images, multichannel=True))            
            bpsnr.append(bPSNR(gt,images))