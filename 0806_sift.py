import numpy as np
import cv2 as cv
img = cv.imread('./img/home.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()

#kp = sift.detect(gray,None)
(kp, features) = sift.detectAndCompute(gray, None)
print ('kp',kp)
print ('features',features[0])

#kppt = np.array([k.pt for k in kp]) #np.float32([k.pt for k in kp])
#print (kppt)
'''
kppt=[]
for k in kp:
    kppt.append(k.pt)
kpptA = np.array(kppt) #np.float32(kppt)
print ('kppt',kpptA)
'''
img = cv.drawKeypoints(gray,kp,img)
#cv.drawKeypoints(gray,kp,img)
# cv2.drawKeypoints(I,kps,I,(0,255,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow("SIFT", img)
cv.imwrite('sift_keypoints.jpg',img)
cv.waitKey(0)
cv.destroyAllWindows()


'''
知识点
1. 特征点检测
https://blog.csdn.net/ei1990/article/details/78289898
https://blog.csdn.net/amusi1994/article/details/79591205

function = cv2.Function_Name_create()
keypoints = function.detect(img, None)

keypoint 数据结构
https://www.cnblogs.com/cj695/p/4041399.html
keypoint包含以下信息：
angle：角度，表示关键点的方向
class_id：用于图片分类，需自己设置
octave：代表是从金字塔哪一层提取的得到的数据
pt：关键点点的坐标
response：响应程度，代表着该关键点how good，更确切的说，是该点角点的程度
size：该点直径的大小

features
sift的特征，128维

# 注意显示之前要先将img2初始化
img2 = img.copy()
img2 = cv2.drawKeyPoints(img, keypoints, color=(0,255,0))

第一个参数image：原始图像，可以使三通道或单通道图像；
第二个参数keypoints：特征点向量，向量内每一个元素是一个KeyPoint对象，包含了特征点的各种属性信息；
第三个参数outImage：特征点绘制的画布图像，可以是原图像；
第四个参数color：绘制的特征点的颜色信息，默认绘制的是随机彩色；
第五个参数flags：特征点的绘制模式，其实就是设置特征点的那些信息需要绘制，那些不需要绘制，有以下几种模式可选：
　　DEFAULT：只绘制特征点的坐标点,显示在图像上就是一个个小圆点,每个小圆点的圆心坐标都是特征点的坐标。 
　　DRAW_OVER_OUTIMG：函数不创建输出的图像,而是直接在输出图像变量空间绘制,要求本身输出图像变量就 是一个初始化好了的,size与type都是已经初始化好的变量 
　　NOT_DRAW_SINGLE_POINTS：单点的特征点不被绘制 
　　DRAW_RICH_KEYPOINTS：绘制特征点的时候绘制的是一个个带有方向的圆,这种方法同时显示图像的坐 标,size，和方向,是最能显示特征的一种绘制方式。
'''

'''
import cv2 as cv
#import numpy as np

img = cv.imread('./img/home.jpg')

sift = cv.xfeatures2d.SIFT_create()

kp = sift.detect(img, None)

img2 = img.copy()

img2 = cv.drawKeypoints(img,kp,img)

cv.imshow('lala',img2)

cv.waitKey(0)
'''