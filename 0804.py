import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('dave.png',0)
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)

img1 = np.float64(img)#转化为浮点型的
sobelxy = cv.Sobel(img1,-1,1,1)

canny = cv.Canny(img,100,200,5)

plt.subplot(2,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(sobelxy,cmap = 'gray')
plt.title('Sobel XY'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(canny,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.show()



'''
#知识点
1.CV filter

cv2.Laplacian(src,ddepth)
ddepth:目标图像要求的深度
laplacian = cv2.Laplacian(img , cv2.CV_64F)

cv2.Sobel(src,ddepth,dx,dy[,ksize])
作用：计算Sobel算子
ddpeth:输出图像的深度，比如CV_8U,CV_64F等
dx:x的导数，dy:y方向的导数
ksize:核的代销，必须是1,3,5或7
如果ksize=-1,会使用3*3的Scharr滤波器，它的效果要比3*3的Sobel滤波器好，3*3的Scharr滤波器卷积核如下:
        -3  0 3
x方向   -10 0 10
        -3 0 3
        -3 -10 -3
y方向    0  0  0
         3  10 3

滤波函数第二个参数，如果原始图像是uint8型的，那么在经过算子计算以后，得到的图像可能会有负值，
如果与原图像数据类型一致，那么负值就会被截断变成0或者255，使得结果错误，那么针对这种问题有两种方式改变：
一种就是改变输出图像的数据类型（第二个参数cv2.CV_64F），
另一种就是改变原始图像的数据类型（np.float64,此时第二个参数可以为-1，与原始图像一致）。

https://blog.csdn.net/qingyuanluofeng/article/details/51594506
http://www.bubuko.com/infodetail-966506.html

2.plt
plt.xticks([]), plt.yticks([])去掉x轴和y轴坐标
'''