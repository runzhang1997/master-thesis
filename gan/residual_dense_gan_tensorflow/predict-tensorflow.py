import tensorflow as tf
import numpy as np
import math
import argparse
import network
import os
import sys
import glob
import json
import time
from skimage.measure import compare_psnr, compare_ssim
sys.path.append('../')
from evaluation import _uqi_single, uqi, ERGAS, SAM, SRE, RMSE, bPSNR
from utils import recompose_images, downPixelAggr, downPixelAggr60 
from normalize import reverse_normalize, normalize

def OpenDataFilesTest(path, run_60, real):
    if real: 
        data10 = np.load(path + '/data10_gt.npy')
        data20 = np.load(path + '/data20_gt_interp.npy')
    else: 
        data10 = np.load(path + '/data10.npy')
        data20 = np.load(path + '/data20_interp.npy')
        
    if run_60:
        if real:
            data60 = np.load(path + '/data60_gt_interp.npy')
        else:
            data60 = np.load(path + '/data60_interp.npy')
        train = [data10, data20, data60]
    else:
        train = [data10, data20]

    with open(path + '/roi.json') as data_file:
        data = json.load(data_file)
        
    if real:
        if run_60:
            image_size = [(data[2] - data[0]) * 6, (data[3] - data[1]) * 6]
        else:
            image_size = [(data[2] - data[0]) * 2, (data[3] - data[1]) * 2]
    else:
        image_size = [(data[2] - data[0]), (data[3] - data[1])]
        
    print("The image size is: {}".format(image_size))
    return train, image_size
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SupResS2.')
    #parser.add_argument('--predict', action='store', dest='predict_file', help='Predict.')
    #parser.add_argument('--resume', action='store', dest='resume_file', help='Resume training.')
    parser.add_argument('--patch_size', help='.', default = 32, type=int)
    parser.add_argument('--save', action='store_true', help='.')
    parser.add_argument('--evaluate', action='store_true', help='.')
    parser.add_argument('--real', action='store_true', help='.')
    parser.add_argument('--test_one', default='none', help='.')
    parser.add_argument('--run_60', action='store_true', help='Whether to run a 60->10m network. Default 20->10m.')
    parser.add_argument('--L2A', action='store_true', help='Whether to run a 60->10m network. Default 20->10m.')
    parser.add_argument('--batch_size', help='.',type=int)
    parser.add_argument('--predict_data', help='Path of train data.')
    parser.add_argument('--model', help='Path of train data.')
    parser.add_argument('--norm', action='store_true', help='instance normlization or not')
    parser.add_argument('--prediction_file', default='none', help='savename')
    parser.add_argument('--relu', action='store_true', help='instance normlization or not')
    args = parser.parse_args()
    
    input_1 = tf.placeholder(tf.float32, [None, None, None, 4], name='input_1')
    input_2 = tf.placeholder(tf.float32, [None, None, None, 6], name='input_2')
    input_3 = tf.placeholder(tf.float32, [None, None, None, 2], name='input_3')
    
    batch_length = tf.placeholder(tf.int32)
    discriminator = network.Discriminator(loss_type = 'wgan-gp', image_size=args.patch_size, batch_size=batch_length, norm=args.norm, run_60=args.run_60)
    generator = network.Generator(adversarial_loss ='wgan-gp', discriminator=discriminator,norm=False, 
    predict=True, batch_size=batch_length, relu=args.relu, run_60=args.run_60)
    
    # generator adversarial loss   
    run_60 = args.run_60
    L2A = args.L2A
    
    if run_60:
        g_y_pred = generator.forward(input_1, input_2, input_3)
        border = 12
    else:
        g_y_pred = generator.forward(input_1, input_2)
        if args.real:
            border = 8
        else:
            border = 4
    
    with tf.Session() as sess:
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        
        latest_checkpt = tf.train.latest_checkpoint(args.model)
        saver.restore(sess, latest_checkpt)
        #saver.restore(sess, args.model)
        if args.test_one == 'none':
            fileList = [os.path.basename(x) for x in sorted(glob.glob(args.predict_data + '*SAFE'))]
        else:
            fileList = [args.test_one]
            #fileList = ['S2B_MSIL1C_20171102T045929_N0206_R119_T45RTL_20171106T173725.SAFE','S2B_MSIL1C_20171022T023429_N0205_R103_T49JGM_20171022T023428.SAFE','S2B_MSIL1C_20171001T141959_N0205_R096_T23VNH_20171001T141956.SAFE','S2B_MSIL1C_20170928T155019_N0205_R054_T18TWL_20170928T155933.SAFE','S2B_MSIL1C_20170914T144719_N0205_R139_T19LDC_20170914T144721.SAFE']
        RMSE_err = []
        SRE_err = []
        SAM_err = []
        ERGAS_err = []
        psnr = []
        bpsnr = []
        ssim = []
        print (run_60)
        print (L2A)
        for dset in fileList:
            start = time.time()
            print("Predicting: {}.".format(dset))
            train, image_size = OpenDataFilesTest(args.predict_data+ dset, run_60, args.real)
            #print (train.shape)
            if run_60:
                train10 = np.moveaxis(train[0],1,3)
                train20 = np.moveaxis(train[1],1,3)
                train60 = np.moveaxis(train[2],1,3)
                train10 = train10/2000.0
                train20 = train20/2000.0
                train60 = train60/2000.0
                #train10, train20, train60 = normalize(train10=train10, train20=train20, train60=train60, run_60=run_60, L2A=L2A, patch_size=32)
            else:
                train10 = np.moveaxis(train[0],1,3)
                train20 = np.moveaxis(train[1],1,3)
                train10 = train10/2000.0
                train20 = train20/2000.0
                
                '''
                train10_mean = np.mean(train10,axis=(0,1,2))
                train20_mean = np.mean(train20,axis=(0,1,2))
                train10_std = np.std(train10,axis=(0,1,2))
                train20_std = np.std(train20,axis=(0,1,2))
                '''
                '''
                train10[:,:,:,0]=(train10[:,:,:,0]-train10_mean[0])/train10_std[0]
                train10[:,:,:,1]=(train10[:,:,:,1]-train10_mean[1])/train10_std[1]
                train10[:,:,:,2]=(train10[:,:,:,2]-train10_mean[2])/train10_std[2]
                train10[:,:,:,3]=(train10[:,:,:,3]-train10_mean[3])/train10_std[3]

                train20[:,:,:,0]=(train20[:,:,:,0]-train20_mean[0])/train20_std[0]
                train20[:,:,:,1]=(train20[:,:,:,1]-train20_mean[1])/train20_std[1]
                train20[:,:,:,2]=(train20[:,:,:,2]-train20_mean[2])/train20_std[2]
                train20[:,:,:,3]=(train20[:,:,:,3]-train20_mean[3])/train20_std[3]
                train20[:,:,:,4]=(train20[:,:,:,4]-train20_mean[4])/train20_std[4]
                train20[:,:,:,5]=(train20[:,:,:,5]-train20_mean[5])/train20_std[5]
                '''
                #train10, train20 = normalize(train10=train10, train20=train20, run_60=run_60, L2A=L2A, patch_size=32)

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
                                                                     batch_length: interval})              
                
                if i ==0:
                    prediction = prediction_i
                else:
                    prediction = np.append(prediction,prediction_i, axis=0)

            prediction = np.moveaxis(prediction,3,1)
            
            print (prediction.shape)
            image = recompose_images(prediction, border=border, size=image_size)
            image = image * 2000.0
            
            
            #image = reverse_normalize(image, run_60=run_60, L2A=L2A, patch_size=32)
            '''
            if run_60:
                image[:,:,0] = image[:,:,0] * train20_std[0] + train20_mean[0]
                image[:,:,1] = image[:,:,1] * train20_std[1] + train20_mean[1]
            else:
                image[:,:,0] = image[:,:,0] * train20_std[0] + train20_mean[0]
                image[:,:,1] = image[:,:,1] * train20_std[1] + train20_mean[1]
                image[:,:,2] = image[:,:,2] * train20_std[2] + train20_mean[2]
                image[:,:,3] = image[:,:,3] * train20_std[3] + train20_mean[3]
                image[:,:,4] = image[:,:,4] * train20_std[4] + train20_mean[4]
                image[:,:,5] = image[:,:,5] * train20_std[5] + train20_mean[5]
            '''
            #images = images * 1470 + 2000
            image[image<0] = 1
            end = time.time()
            print('Elapsed time: {}.'.format(end - start))
            
            if args.evaluate:
                if run_60:
                    gt = np.load(args.predict_data + dset + '/no_tiling/data60_gt.npy')
                else:
                    gt = np.load(args.predict_data + dset + '/no_tiling/data20_gt.npy')
                
                print (np.mean(image)-np.mean(gt))                     
            
                RMSE_err.append(RMSE(image, gt))
                SRE_err.append(SRE(image, gt))
                ERGAS_err.append(ERGAS(image, gt, 2))
                SAM_err.append(SAM(image, gt))
                psnr.append(compare_psnr(gt,image, data_range=np.max(gt)))
                ssim.append(compare_ssim(gt,image, multichannel=True))            
                bpsnr.append(bPSNR(gt,image))
            
            if args.save:
                print('Writing to file...')
                np.save(args.predict_data+ dset + '/' + args.prediction_file, image)
                '''
                if args.real:
                    if run_60:
                        degrade = downPixelAggr60(image, 6)
                    else:
                        degrade = downPixelAggr(image, 2)
                    np.save(args.predict_data+ dset + '/' + args.prediction_file + '_real_degraded', degrade) 
                '''
        print('RMSE Average: {:.4f}'.format(np.mean(RMSE_err)))
        print('SRE Average: {:.4f}'.format(np.mean(SRE_err)))
        print('ERGAS Average: {:.4f}'.format(np.mean(ERGAS_err)))
        print('SAM Average:{:.4f}'.format(np.mean(SAM_err)))
        print('psnr: {:.4f}'.format(np.mean(psnr)))
        print('ssim: {:.4f}'.format(np.mean(ssim)))       
        print('bpsnr: {:.4f}'.format(np.mean(bpsnr)))
        
