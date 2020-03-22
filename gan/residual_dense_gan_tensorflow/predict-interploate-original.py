import numpy as np
import math
from scipy import misc
from scipy.ndimage import uniform_filter
import argparse
import network
import os
import sys
import cv2
import glob
from tqdm import tqdm
import time
from skimage.measure import compare_psnr, compare_ssim
sys.path.append('../')
from evaluation import _uqi_single, uqi, ERGAS, SAM, SRE, RMSE, bPSNR
from utils import recompose_images, downPixelAggr
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SupResS2.')
    parser.add_argument('--run_60', action='store_true', help='.')
    parser.add_argument('--save', action='store_true', help='.')
    parser.add_argument('--predict_data', help='Path of test data.')
    parser.add_argument('--interp', help='Path of train data.')
    parser.add_argument('--test_one', default = 'none', help='P.')
    parser.add_argument('--predication_file', default = 'none', help='P.')
    args = parser.parse_args()
    
    path = '/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/' + args.predict_data
    
    if args.test_one =='none':
        imageList = [os.path.basename(x) for x in sorted(glob.glob(path + '*SAFE'))]
    else:
        imageList = [args.test_one]
    RMSE_err = []
    SRE_err = []
    SAM_err = []
    ERGAS_err = []
    psnr = []
    ssim = []
    uiqi = []
    bpsnr = []
    if args.interp =='bicubic':
        inter = cv2.INTER_CUBIC      
    elif args.interp == 'nearest':
        inter =cv2.INTER_NEAREST       
    elif args.interp == 'bilinear':
        inter =cv2.INTER_LINEAR        
    else:
        inter =cv2.INTER_LANCZOS4
                    
    for dset in imageList:
        print("Predicting: {}.".format(dset))
        if args.run_60:
            gt = np.load(path + dset + '/no_tiling/data60_gt.npy')
            
            up_gt = cv2.resize(gt, (gt.shape[1] * 6, gt.shape[0] * 6),  interpolation = inter)
            print('Writing to file...')
            np.save(path+ dset + '/' + args.predication_file + '_60_real', up_gt)
            down_gt = downPixelAggr(up_gt, 6)
            print (down_gt.shape)
            print('Writing to file...')
            np.save(path+ dset + '/' + args.predication_file + '_60_real_degraded', down_gt)
            #up_gt = cv2.resize(down_gt, (gt.shape[1], gt.shape[0]),  interpolation = inter)
            #down_gt = np.moveaxis(down_gt,1,3)
            '''
            up_gt = np.zeros((down_gt.shape[0], down_gt.shape[1] * 6, down_gt.shape[2] * 6, down_gt.shape[3]))
            for i in range(0, down_gt.shape[0]):
                up_gt[i] = cv2.resize(down_gt[i], (down_gt.shape[1] * 6, down_gt.shape[2] * 6),  interpolation=inter)
            up_gt = np.moveaxis(up_gt,3,1)
            up_gt = recompose_images(up_gt, border=12, size=(1830, 1830, 2))   
            '''
        else:       
            gt = np.load(path + dset + '/no_tiling/data20_gt.npy')
            up_gt = cv2.resize(gt, (gt.shape[1] * 2, gt.shape[0] * 2),  interpolation = inter)
            
            print('Writing to file...')
            np.save(path+ dset + '/' + args.predication_file + '_real', up_gt)
            
            down_gt = downPixelAggr(up_gt, 2)
            print(down_gt.shape)
            
            print('Writing to file...')
            np.save(path+ dset + '/' + args.predication_file + '_real_degraded', down_gt)
            
            #up_gt = cv2.resize(down_gt, (gt.shape[1], gt.shape[0]),  interpolation = inter)
            #print (up_gt.shape)
            #up_gt = np.zeros((down_gt.shape[0], down_gt.shape[1] * 2, down_gt.shape[2] * 2, down_gt.shape[3]))
            #for i in range(0, down_gt.shape[0]):
            #    up_gt[i] = cv2.resize(down_gt[i], (down_gt.shape[1] * 2, down_gt.shape[2] * 2),  interpolation=inter)
            #up_gt = np.moveaxis(up_gt,3,1)
            #up_gt = recompose_images(up_gt, border=4, size=(5490, 5490, 6))    
        '''    
        ssim.append(compare_ssim(gt,up_gt,multichannel=True))
        uiqi.append(uqi(gt,up_gt))
        RMSE_err.append(RMSE(up_gt, gt))
        SRE_err.append(SRE(up_gt, gt))
        ERGAS_err.append(ERGAS(up_gt, gt, 2))
        SAM_err.append(SAM(up_gt, gt))
        psnr.append(compare_psnr(gt,up_gt,data_range=np.max(gt)))
        bpsnr.append(bPSNR(up_gt, gt))
        
        if args.save or args.test_one!='none':
            print('Writing to file...')
            np.save(path+ dset + '/' + args.predication_file, up_gt)
        '''
    print('psnr: {:.4f}'.format(np.mean(psnr)))
    print('bpsnr: {:.4f}'.format(np.mean(bpsnr)))
    print('RMSE Average: {:.4f}'.format(np.mean(RMSE_err)))
    print('UIQI Average: {:.4f}'.format(np.mean(uiqi)))    
    print('ssim: {:.4f}'.format(np.mean(ssim)))
    print('SRE Average: {:.4f}'.format(np.mean(SRE_err)))
    print('SAM Average:{:.4f}'.format(np.mean(SAM_err)))
    print('ERGAS Average: {:.4f}'.format(np.mean(ERGAS_err)))
    
    '''                  
    for dset in imageList:
        print("Predicting: {}.".format(dset))
        if args.run_60:
            gt = np.load(path + dset + '/no_tiling/data60_gt.npy')
            down_gt = np.load(path + dset + '/data60.npy')
            down_gt = np.moveaxis(down_gt,1,3)
            up_gt = np.zeros((down_gt.shape[0], down_gt.shape[1] * 6, down_gt.shape[2] * 6, down_gt.shape[3]))
            for i in range(0, down_gt.shape[0]):
                up_gt[i] = cv2.resize(down_gt[i], (down_gt.shape[1] * 6, down_gt.shape[2] * 6),  interpolation=inter)
            up_gt = np.moveaxis(up_gt,3,1)
            up_gt = recompose_images(up_gt, border=12, size=(1830, 1830, 2))    
        else:       
            gt = np.load(path + dset + '/no_tiling/data20_gt.npy')
            print (gt.shape)
            down_gt = np.load(path + dset + '/data20.npy')
            down_gt = np.moveaxis(down_gt,1,3)
            up_gt = np.zeros((down_gt.shape[0], down_gt.shape[1] * 2, down_gt.shape[2] * 2, down_gt.shape[3]))
            for i in range(0, down_gt.shape[0]):
                up_gt[i] = cv2.resize(down_gt[i], (down_gt.shape[1] * 2, down_gt.shape[2] * 2),  interpolation=inter)
            up_gt = np.moveaxis(up_gt,3,1)
            up_gt = recompose_images(up_gt, border=4, size=(5490, 5490, 6))    
        
        #psnr.append(compare_psnr(gt,up_gt))
        ssim.append(compare_ssim(gt,up_gt,multichannel=True))
        uiqi.append(uqi(gt,up_gt))
        RMSE_err.append(RMSE(up_gt, gt))
        SRE_err.append(SRE(up_gt, gt))
        ERGAS_err.append(ERGAS(up_gt, gt, 2))
        SAM_err.append(SAM(up_gt, gt))
        if args.save or args.test_one!='none':
            #print('Writing to file...')
            np.save(path+ dset + '/' + args.predication_file, up_gt)
        
    print('RMSE Average: {:.4f}'.format(np.mean(RMSE_err)))
    print('UIQI Average: {:.4f}'.format(np.mean(uiqi)))
    #print('psnr: {:.4f}'.format(np.mean(psnr)))
    print('ssim: {:.4f}'.format(np.mean(ssim)))
    print('SRE Average: {:.4f}'.format(np.mean(SRE_err)))
    print('SAM Average:{:.4f}'.format(np.mean(SAM_err)))
    print('ERGAS Average: {:.4f}'.format(np.mean(ERGAS_err)))
    '''