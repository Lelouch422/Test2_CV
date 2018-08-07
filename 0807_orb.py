
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('./img//box.png',0)          # queryImage
img2 = cv.imread('./img/box_in_scene.png',0) # trainImage
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
print (matches)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:20],None, flags=2)
plt.imshow(img3),plt.show()



'''
知识点
1.BFMatcher和FlannBasedMatcher
BFMatcher总是尝试所有可能的匹配，从而使得它总能够找到最佳匹配，这也是Brute Force（暴力法）的原始含义。
而FlannBasedMatcher中FLANN的含义是Fast Library forApproximate Nearest Neighbors，它是一种近似法，算法更快但是找到的是最近邻近似匹配

使用特征提取过程得到的特征描述符（descriptor）数据类型有的是float类型的，比如说SurfDescriptorExtractor，SiftDescriptorExtractor，有的是uchar类型的，比如说有ORB，BriefDescriptorExtractor。
对应float类型的匹配方式有：FlannBasedMatcher，BruteForce<L2<float>>，BruteForce<SL2<float>>，BruteForce<L1<float>>。对应uchar类型的匹配方式有：BruteForce<Hammin>，BruteForce<HammingLUT>。
所以ORB和BRIEF特征描述子只能使用BruteForce匹配法。

2.类型转换
http://www.cnblogs.com/hhh5460/p/5129032.html
numpy中的数据类型转换，不能直接改原数据的dtype!  只能用函数astype()
b.dtype = 'int'
b = b.astype(int)
NumPy中的基本数据类型
https://blog.csdn.net/paulsweet123/article/details/52461933
'''

'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img1 = cv.imread('./img//box.png',0)
img2 = cv.imread('./img/box_in_scene.png',0)


#surf特征不是一维的，且类型不是
surf = cv.xfeatures2d.SURF_create()
kp1,des1 = surf.detectAndCompute(img1, None)
kp2,des2 = surf.detectAndCompute(img2, None)

# FLANN parameters 
FLANN_INDEX_KDTREE = 0 
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) 
search_params = dict(checks=50)  # or pass empty dictionary 
flann = cv.FlannBasedMatcher(index_params,search_params) 
matches = flann.knnMatch(des1,des2,k=2)

print (matches)
#des1.dtype = 'uint8'
#des2.dtype = 'uint8'
#des1 = des1.astype(np.uint8)
#des2 = des2.astype(np.uint8)
#print(des1)
#bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
#matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)

img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:20],None, flags=2)

plt.imshow(img3)
plt.show()
'''

'''
#SIFT match
https://www.jb51.net/article/85845.htm

#coding=utf-8 
import cv2 
import scipy as sp 
 
img1 = cv2.imread('./img//box.png',0) # queryImage
img2 = cv2.imread('./img/box_in_scene.png',0) # trainImage 
 
# Initiate SIFT detector 
sift = cv2.xfeatures2d.SIFT_create() 
 
# find the keypoints and descriptors with SIFT 
kp1, des1 = sift.detectAndCompute(img1,None) 
kp2, des2 = sift.detectAndCompute(img2,None) 
 
# FLANN parameters 
FLANN_INDEX_KDTREE = 0 
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) 
search_params = dict(checks=50)  # or pass empty dictionary 
flann = cv2.FlannBasedMatcher(index_params,search_params) 
matches = flann.knnMatch(des1,des2,k=2) 

print (matches)
print ('matches...',len(matches)) 
# Apply ratio test 
good = [] 
for m,n in matches: 
  if m.distance < 0.35*n.distance: 
    good.append(m) 
print ('good',len(good)) 
# ##################################### 
# visualization 
h1, w1 = img1.shape[:2] 
h2, w2 = img2.shape[:2] 
view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8) 
view[:h1, :w1, 0] = img1 
view[:h2, w1:, 0] = img2 
view[:, :, 1] = view[:, :, 0] 
view[:, :, 2] = view[:, :, 0] 
 
for m in good: 
  # draw the keypoints 
  # print m.queryIdx, m.trainIdx, m.distance 
  color = tuple([sp.random.randint(0, 255) for _ in range(3)]) 
  #print 'kp1,kp2',kp1,kp2 
  cv2.line(view, (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])) , (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1])), color) 
 
cv2.imshow("view", view) 

#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20][1],None, flags=2)
#cv2.imshow("img3", img3)

cv2.waitKey(0)
'''