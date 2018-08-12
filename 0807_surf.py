import numpy as np
import cv2 as cv
img = cv.imread('./img/butterfly.jpg',0)
#img = cv.imread('./img/home.jpg')

surf = cv.xfeatures2d.SURF_create(400)

#kp, des = surf.detectAndCompute(img,None)
surf.setHessianThreshold(50000)

kp, des = surf.detectAndCompute(img, None)
#kp = surf.detect(img, None)
print (des[0])

#img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
img2 = cv.drawKeypoints(img,kp,None,flags=4)
cv.imshow('surf',img2)

cv.waitKey(0)
cv.destroyAllWindows()


'''
知识点
Hessian矩阵是用来判断该点是不是极值点的，具体的就是把多元函数的2阶偏导数拼成一个矩阵
surf 特征64维
'''

'''
import cv2 as cv
import numpy as np

img = cv.imread('./img/home.jpg')
surf = cv.xfeatures2d.SURF_create(400)

kp,des = surf.detectAndCompute(img, None)

img2 = img.copy()
img2 = cv.drawKeypoints(img, kp, img2, (255,0,0), 4)

cv.imshow('surf',img2)
cv.waitKey(0)
cv.destroyAllWindows()
'''

