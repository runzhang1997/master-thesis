import numpy as np
import argparse
import math
import os
import glob
#I1: estimated image
#I2: reference image
# Optimal value = 0
def ERGAS(I1, I2, ratio):
    I1 = I1.astype(float)
    I2 = I2.astype(float)

    Err = I1 - I2
    ERGAS_index = 0
    for iLR in range(0, I1.shape[2]):
        ERGAS_index = ERGAS_index + np.mean(np.square(Err[:, :, iLR])) / np.square(np.mean(I1[:, :, iLR]))

    ERGAS_index = (100.0 / ratio) * math.sqrt((1.0 / Err.shape[2]) * ERGAS_index)
    print('ERGAS: {:.4f}'.format(ERGAS_index))
    return ERGAS_index

# The order does not matter
# Optimal value = 0
def SAM(I1, I2):
    (M, N, P) = I2.shape
    prod_scal = np.sum(I1 * I2, axis=-1)
    norm_orig = np.sum(I1 * I1, axis=-1)
    norm_fusa = np.sum(I2 * I2, axis=-1)
    prod_norm = np.sqrt(norm_orig * norm_fusa)
    prod_scal = np.reshape(prod_scal, (M * N, 1))
    prod_norm = np.reshape(prod_norm, (M * N, 1))
    z_idx = np.where(prod_norm == 0)
    np.delete(prod_scal, z_idx)
    np.delete(prod_norm, z_idx)
    angolo = np.nanmean(np.arccos(np.divide(prod_scal, prod_norm)))
    SAM_index = np.real(angolo) * 180.0 / np.pi
    print('SAM: {:.4f}'.format(SAM_index))
    return SAM_index

#X: estimated image
#Xhat: reference image
# Higher is the value, better it is
def SRE(X, Xhat):
    (M, N, P) = X.shape
    X = np.reshape(X, (M * N, P))
    Xhat = np.reshape(Xhat, (M * N, P))
    SRE_vec = 10 * np.log10(np.sum(np.power(Xhat, 2), axis=0) / np.sum(np.power(Xhat - X, 2), axis=0))
    SRE = np.mean(SRE_vec)
    print('SRE: {:.4f}'.format(SRE))
    return SRE
    #return {'SRE': SRE, 'SRE_vec': SRE_vec}

def RMSE(x1, x2):
    diff = x1.astype(np.float64)-x2.astype(np.float64)
    rms = np.sqrt(np.mean(np.power(diff, 2)))
    print('RMSE: {:.4f}'.format(rms))
    return rms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SupResS2.')
    parser.add_argument('--model', help='Path of train data.')
    parser.add_argument('--data_path', help='Path of train data.')
    args = parser.parse_args()  
    path = args.data_path
    filelist = [os.path.basename(x) for x in sorted(glob.glob(path+'*SAFE'))]
    RMSE_err = 0.0
    SRE_err = 0.0
    SAM_err = 0.0
    ERGAS_err = 0.0
    for dset in filelist:
        predict = np.load(path+dset+'/%s.npy'%args.model)
        gt = np.load(path+dset+'/no_tiling/data20_gt.npy')
        print(dset)
        RMSE_err +=RMSE(predict, gt)
        SRE_err += SRE(predict, gt)
        ERGAS_err += ERGAS(predict, gt, 2)
        SAM_err += SAM(predict, gt)  
    print('RMSE Average: {:.4f}'.format(RMSE_err/15.0))
    print('SRE Average: {:.4f}'.format(SRE_err/15.0))
    print('ERGAS Average: {:.4f}'.format(ERGAS_err/15.0))
    print('SAM Average:{:.4f}'.format(SAM_err/15.0))
#print (result.shape)
	#print (ground_truth.shape)
	#print (ground_truth.shape[2])

	#print (ERGAS(result,ground_truth,2))
	#print (SAM(result,ground_truth))
