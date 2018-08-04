import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('opencv.png',0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

oriFFT = np.log(np.abs(f))
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(141),plt.imshow(img, cmap = 'gray')
plt.title('Input'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(magnitude_spectrum, 'gray')
plt.title('Center_shift'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(oriFFT)
plt.title('Origin FFT'), plt.xticks([]), plt.yticks([])

# 逆变换
f1shift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f1shift)
#img_back = np.fft.ifft2(f)
#出来的是复数，无法显示
img_back2 = np.abs(img_back)
plt.subplot(144),plt.imshow(img_back2,'gray'),plt.title('img back')
plt.title('Reverse'), plt.xticks([]), plt.yticks([])

plt.show()

'''
#知识点
1.fft
一维fft：np.fft.fft，结果为实数+虚数
https://blog.csdn.net/ouening/article/details/71079535

二维fft：np.fft.fft2
https://blog.csdn.net/on2way/article/details/46981825  对图像傅里叶详细，可继续学习。
fftshift：移频，将0频分量移到中间
#取绝对值：将复数变化成实数
#取对数的目的为了将数据变化到较小的范围（比如0-255）

s1 = np.abs(fshift) #取振幅
s1_angle = np.angle(fshift) #取相位

'''
