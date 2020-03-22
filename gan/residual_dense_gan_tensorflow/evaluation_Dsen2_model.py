import numpy as np
import argparse
from scipy import misc
from scipy.ndimage import uniform_filter
import math
import os
import glob
import time
import json
from skimage.measure import compare_psnr, compare_ssim
from evaluation import _uqi_single, uqi, ERGAS, SAM, SRE, RMSE, bPSNR
from utils import recompose_images, downPixelAggr, downPixelAggr60 
"""
model_name='s2_032_'
test_dir='/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/test20_128_interp_bilinear'
python evaluation_Dsen2_model.py --data_path $test_dir --model $model_name
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SupResS2.')
    parser.add_argument('--model', help='Path of train data.') 
    #parser.add_argument('--data_path', help='Path of train data.')
    args = parser.parse_args()  
    path = '/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/test20_32_interp_bilinear/'
    #model_name='pretrain_20m_real_relativistic' 
    model_name = args.model
    #model_name='pretrain_20m_real_wgan'
    #model_name='pretrain_20m_real_vanila'
    #model_name='pretrain_20m_real_hinge'
    filelist = [os.path.basename(x) for x in sorted(glob.glob(path+'*SAFE'))]
    
    RMSE_err = []
    SRE_err = []
    SAM_err = []
    ERGAS_err = []
    psnr = []
    ssim = []   
    bpsnr = []
    '''
    for band in range(6):
        print ("bands %d"%band)
        for dset in filelist:
            images = np.load(path+dset+'/%s.npy'%model_name)
            gt = np.load(path+dset+'/no_tiling/data20_gt.npy')
            print(dset)           
            gt = gt[:,:,band]
            images = images[:,:,band]
            images = images.reshape(images.shape[0], images.shape[1], 1)
            gt = gt.reshape(gt.shape[0], gt.shape[1], 1)
            
            RMSE_err.append(RMSE(images, gt))
            SRE_err.append(SRE(images, gt))           
            #ERGAS_err.append(ERGAS(images, gt, 2))
            #SAM_err.append(SAM(images, gt))
            #psnr.append(compare_psnr(gt,images, data_range=np.max(gt)))
            #ssim.append(compare_ssim(gt,images, multichannel=True))           
        print('RMSE Average: {:.4f}'.format(np.mean(RMSE_err)))
        print('SRE Average: {:.4f}'.format(np.mean(SRE_err)))
        #print('ERGAS Average: {:.4f}'.format(np.mean(ERGAS_err)))
        #print('SAM Average:{:.4f}'.format(np.mean(SAM_err)))
        #print('psnr: {:.4f}'.format(np.mean(psnr)))
        #print('ssim: {:.4f}'.format(np.mean(ssim)))
    '''
    
    for dset in filelist:
        images = np.load(path+dset+'/%s.npy'%model_name)
        images = downPixelAggr(images, 2)
        gt = np.load(path+dset+'/no_tiling/data20_gt.npy')
        print(dset)
       
        RMSE_err.append(RMSE(images, gt))        
        SRE_err.append(SRE(images, gt))           
        ERGAS_err.append(ERGAS(images, gt, 2))
        SAM_err.append(SAM(images, gt))
        psnr.append(compare_psnr(gt,images, data_range=np.max(gt)))
        ssim.append(compare_ssim(gt,images, multichannel=True))        
        bpsnr.append(bPSNR(gt,images))
        
    print('RMSE Average: {:.4f}'.format(np.mean(RMSE_err)))
    print('SRE Average: {:.4f}'.format(np.mean(SRE_err)))
    print('ERGAS Average: {:.4f}'.format(np.mean(ERGAS_err)))
    print('SAM Average:{:.4f}'.format(np.mean(SAM_err)))
    print('psnr: {:.4f}'.format(np.mean(psnr)))
    print('ssim: {:.4f}'.format(np.mean(ssim)))
    print('bpsnr: {:.4f}'.format(np.mean(bpsnr)))