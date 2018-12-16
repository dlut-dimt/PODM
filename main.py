import os
import copy
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt

from numpy.linalg import norm
from scipy.misc import imresize
from skimage import img_as_float
from skimage.io import imread, imshow, imsave
from skimage.color import rgb2ycbcr, ycbcr2rgb

from denoise_net import denoise_model
from util import modcrop, grad_prox_SR, compute_psnr


if __name__ == "__main__":

    scaleFactor = 4.0
    L = 1
    lambda1 = 1e-4
    net_idx = [24,23,21,19,18,16,15,14,13,12,11,10,9,8,8,7,7,6,6,5,5,4,4,4,4,3,3,3,3,2]
    maxIter = len(net_idx)
    inIter = 5
    C = 0.08
    alpha = 1e-3
    mu = 1e-3

    K.clear_session()
    HR = img_as_float(imread("./baby_GT.bmp")).astype(np.float32)
    HR = modcrop(HR, int(scaleFactor))

    if len(HR.shape) == 2:
        HR = np.tile(HR[...,np.newaxis], (1,1,3))

    HR_ycc = rgb2ycbcr(HR) / 255.0
    label = HR_ycc[:,:,0]

    LR = imresize(label, 1.0/scaleFactor, 'bicubic','F')
    HR_bic = imresize(LR, scaleFactor, 'bicubic', 'F')

    HR_ycc1 = copy.deepcopy(HR_ycc)
    HR_ycc1[:,:,0] = HR_bic
    imsave('./lr.png', np.clip(ycbcr2rgb(HR_ycc1*255),0.0,1.0))

    xlk = copy.deepcopy(HR_bic)
    xlk_old = copy.deepcopy(xlk)

    for i in range(maxIter):

        # step 1
        model = denoise_model()
        model.load_weights('./model/net' + str(int(net_idx[i])) + '.hdf5')
        xlk_input =xlk[np.newaxis,...,np.newaxis]
        residual = model.predict(xlk_input)
        residual = np.squeeze(residual)
        xpk = xlk - residual

        # step 2
        xfk = xpk
        for k in range(inIter):
            grad_1 = imresize((imresize(xfk, 1.0/scaleFactor, 'bicubic', 'F') - LR), scaleFactor, 'bicubic', 'F') + mu * (xfk - xpk)
            xfk = xfk - 2 * grad_1

        # step 3
        xgk = grad_prox_SR(xfk, xlk_old, LR, scaleFactor, L, lambda1, alpha)

        grad_xfk = imresize((imresize(xfk, 1.0/scaleFactor, 'bicubic', 'F') - LR), scaleFactor, 'bicubic', 'F')
        grad_xgk = imresize((imresize(xgk, 1.0/scaleFactor, 'bicubic', 'F') - LR), scaleFactor, 'bicubic', 'F')

        error_x1 = (alpha - 1.0 / L) * (xgk - xfk) 
        error_x2 = grad_xfk - grad_xgk
        error1 = norm(error_x1 - error_x2, 'fro')
        error2 = norm(xgk - xlk_old, 'fro')

        if error1 <= C * error2:
            xlk = xgk
        else :
            xlk = grad_prox_SR(xlk_old, xlk_old, LR, scaleFactor, L, lambda1, 0)

        error = norm(xlk - xlk_old, 'fro') / norm(xlk_old, 'fro')
        if error < 1e-4:
            break
        
        xgk_color = copy.deepcopy(HR_ycc)
        xgk_color[:,:,0] = xgk
        xgk_color = ycbcr2rgb(xgk_color*255)
        
        psnr, ssim = compute_psnr(label, xgk, int(scaleFactor))

        print("iter:%d psnr:%f ssim:%f"%(i, psnr, ssim))

    imsave('./output.png', np.clip(xgk_color, 0.0, 1.0))

