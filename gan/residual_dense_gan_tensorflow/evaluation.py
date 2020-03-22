#import tensorflow as tf
import numpy as np
import math
from math import ceil
from scipy.ndimage import uniform_filter
from skimage.measure import compare_psnr, compare_ssim

#from FUNCTION import *

def bPSNR (images,gt):
    b = np.mean(gt-images)
    print ("brightness-difference:%f"%b)
    return compare_psnr(gt,images+b, data_range=np.max(gt))

def _uqi_single(GT,P,ws):
    N = ws**2
    window = np.ones((ws,ws))

    GT_sq = GT*GT
    P_sq = P*P
    GT_P = GT*P

    GT_sum = uniform_filter(GT, ws)    
    P_sum =  uniform_filter(P, ws)     
    GT_sq_sum = uniform_filter(GT_sq, ws)  
    P_sq_sum = uniform_filter(P_sq, ws)  
    GT_P_sum = uniform_filter(GT_P, ws)

    GT_P_sum_mul = GT_sum*P_sum
    GT_P_sum_sq_sum_mul = GT_sum*GT_sum + P_sum*P_sum
    numerator = 4*(N*GT_P_sum - GT_P_sum_mul)*GT_P_sum_mul
    denominator1 = N*(GT_sq_sum + P_sq_sum) - GT_P_sum_sq_sum_mul
    denominator = denominator1*GT_P_sum_sq_sum_mul

    q_map = np.ones(denominator.shape)
    index = np.logical_and((denominator1 == 0) , (GT_P_sum_sq_sum_mul != 0))
    q_map[index] = 2*GT_P_sum_mul[index]/GT_P_sum_sq_sum_mul[index]
    index = (denominator != 0)
    q_map[index] = numerator[index]/denominator[index]

    s = int(np.round(ws/2))
    return np.mean(q_map[s:-s,s:-s])

def uqi (GT,P,ws=8):
    """calculates universal image quality index (uqi).
    :param GT: first (original) input image.
    :param P: second (deformed) input image.
    :param ws: sliding window size (default = 8).
    """
    return np.mean([_uqi_single(GT[:,:,i],P[:,:,i],ws) for i in range(GT.shape[2])])
    
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
#Higher is the value, better it is
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


## next task, to realize the metrics cPSNR in the thesis.
'''
def psnr(input, label, shave):
    assert (label.dtype == input.dtype)
    if shave != 0:
        input = input[shave:-shave, shave:-shave, :]
        label = label[shave:-shave, shave:-shave, :]
    diff = np.int32(label) - np.int32(input)
    mse = np.mean(diff ** 2.)
    if mse == 0:
        return 1e6
    max_i = np.iinfo(label.dtype).max
    return 10*np.log10(max_i * max_i / mse)

def tf_psnr(input, label, shave):
    assert (label.dtype == input.dtype)
    if shave != 0:
        input = input[shave:-shave, shave:-shave, :]
        label = label[shave:-shave, shave:-shave, :]
    diff = tf.cast(label, tf.int32) - tf.cast(input, tf.int32)
    mse = tf.reduce_mean(tf.cast(diff, tf.float32) ** 2.)
    return tf.cond(tf.equal(mse, 0), lambda: tf.constant(1e6, dtype=mse.dtype), lambda: 20*np.log10(label.dtype.max)-10*tf_log10(mse))

def tf_psnr_float(input, label, shave, max_i, single_channel, is_circle, scale):
    assert (label.dtype == input.dtype)

    if shave != 0:
        if single_channel:
            if shave != 0 and len(input.get_shape()) == 2:
                input = input[shave:-shave, shave:-shave]
                label = label[shave:-shave, shave:-shave]
            elif shave != 0 and len(input.get_shape()) == 3:
                input = input[:, shave:-shave, shave:-shave]
                label = label[:, shave:-shave, shave:-shave]
            else:
                assert False, 'DIMENSION ERROR'
        else:
            if len(input.get_shape()) == 2:
                input = input[shave:-shave, shave:-shave]
                label = label[shave:-shave, shave:-shave]
            elif len(input.get_shape()) == 3:
                input = input[shave:-shave, shave:-shave, :]
                label = label[shave:-shave, shave:-shave, :]
            elif len(input.get_shape()) == 4:
                input = input[:, shave:-shave, shave:-shave, :]
                label = label[:, shave:-shave, shave:-shave, :]
            else:
                assert False, 'DIMENSION ERROR'

    diff = tf.sub(label, input)
    if is_circle:
        h = diff
        abs_dh = tf.abs(h)
        middle = 0.5 * max_i
        mask_1 = tf.cast( abs_dh > middle, tf.float32)
        mask_2 = tf.cast((mask_1 * h) < 0, tf.float32)
        mask_3 = tf.cast((mask_1 * h) > 0, tf.float32)
        diff = (h + (middle * mask_2) - (middle * mask_3)) * 2

    mse = tf.reduce_mean(tf.square(diff)) * scale + 1e-10
    return 20*np.log10(max_i)-10*tf_log10(mse)

def ssim(img1, img2, shave, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    assert(img1.dtype == img2.dtype)
    if shave != 0:
        img1 = img1[shave:-shave, shave:-shave, :]
        img2 = img2[shave:-shave, shave:-shave, :]
    size = (size - 1) // 2
    window = fspecial_gauss(size, sigma)
    L = np.iinfo(img1.dtype).max
    K1 = 0.01
    K2 = 0.03
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mu1 = ndimage.convolve(img1, window)
    mu2 = ndimage.convolve(img2, window)
    mu1 = mu1[size:-size, size:-size, :]
    mu2 = mu2[size:-size, size:-size, :]
    mu1_sq  = mu1*mu1
    mu2_sq  = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = ndimage.convolve(img1*img1, window)
    sigma2_sq = ndimage.convolve(img2*img2, window)
    sigma12   = ndimage.convolve(img1*img2, window)
    sigma1_sq = sigma1_sq[size:-size, size:-size, :] - mu1_sq
    sigma2_sq = sigma2_sq[size:-size, size:-size, :] - mu2_sq
    sigma12   = sigma12  [size:-size, size:-size, :] - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value =  ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = np.mean(value)
    return value

def tf_fspecial_gauss(size, sigma):
    x_data, y_data = np.mgrid[-size:size + 1, -size:size + 1]

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x*x + y*y)/(2.0*sigma*sigma)))
    g = tf.expand_dims(g, -1)
    g = tf.expand_dims(g, -1)
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, shave, L, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    assert(img1.dtype == img2.dtype)
    img1 = tf.transpose(img1, [2, 0, 1])
    img2 = tf.transpose(img2, [2, 0, 1])
    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)
    if shave != 0:
        img1 = img1[:, shave:-shave, shave:-shave, :]
        img2 = img2[:, shave:-shave, shave:-shave, :]
    window = tf_fspecial_gauss((size-1)//2, sigma)
    K1 = 0.01
    K2 = 0.03
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    img1 = tf.cast(img1, tf.float32)
    img2 = tf.cast(img2, tf.float32)
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='VALID')
    mu1_sq  = mu1*mu1
    mu2_sq  = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1], padding='VALID') - mu2_sq
    sigma12   = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1], padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value =  ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value
'''