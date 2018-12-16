import numpy as np
import tensorflow as tf
import keras.backend as K

from scipy.misc import imresize
from skimage.color import rgb2ycbcr
from scipy.fftpack import dct, idct
from skimage.measure import compare_ssim, compare_psnr

def modcrop(img, factor):
    imgsize = img.shape
    if len(imgsize) == 2:
        return img[:imgsize[0] - imgsize[0]%factor, :imgsize[1] - imgsize[1]%factor]
    else :
        return img[:imgsize[0] - imgsize[0]%factor, :imgsize[1] - imgsize[1]%factor, :]
        
    
def grad_prox_SR(xfk, xlk_old, LR, scaleFactor, L, lambda1, alpha):
    grad = imresize((imresize(xfk, 1.0/scaleFactor, 'bicubic', 'F') - LR), scaleFactor, 'bicubic', 'F') + alpha * (xfk - xlk_old)
    Y = xfk - 1.0/L*grad

    WY = dct(Y, norm='ortho')
    WY = solve_Lp(WY, lambda1/L, 0.8)

    xgk = idct(K.eval(WY), norm='ortho')
    return xgk

def solve_Lp(y, lambda1, p):
    J = 2
    tau = (2*lambda1*(1-p))**(1.0/(2-p)) + p*lambda1*(2*(1-p)*lambda1) ** ((p-1)/(2-p))
    x = tf.zeros(tf.shape(y))

    i0 = K.greater(K.abs(y), tau)

    t = K.abs(y)
    for j in range(J):
        t = tf.abs(y) - p*lambda1*t**(p-1)
    result = tf.where(i0, tf.sign(y)*t, x)

    return result
   
def shave(im, border=0):
    return im[border:-border, border:-border]

def compute_psnr(im1, im2, shave_border):
    if len(im1.shape) == 3:
        im1 = rgb2ycbcr(im1)[:,:,0]
    if len(im2.shape) == 3:
        im2 = rgb2ycbcr(im2)[:,:,0]

    im1 = shave(im1, shave_border)
    im2 = shave(im2, shave_border)
    im1 = np.clip(im1.astype(np.float32), 0.0, 1.0)
    im2 = np.clip(im2.astype(np.float32), 0.0, 1.0)
    psnr = compare_psnr(im1, im2)
    ssim = compare_ssim(im1, im2)

    return psnr, ssim
