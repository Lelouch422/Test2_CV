import cv2
import matplotlib.pyplot as plt

img = cv2.imread('timg.jpg',0) #直接读为灰度图像
res = cv2.equalizeHist(img)

clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(20,20))
cl1 = clahe.apply(img)

plt.subplot(131),plt.imshow(img,'gray')
plt.subplot(132),plt.imshow(res,'gray')
plt.subplot(133),plt.imshow(cl1,'gray')

plt.show()


'''
知识点
直方图均衡化:拉伸直方图
1、统计直方图每个灰度值出现的次数
2、计算每个灰度值出现的概率，并按灰度值从小到大计算累计概率
3、用累计概率*255得到新的像素值

https://blog.csdn.net/zhangfuliang123/article/details/74170894
https://blog.csdn.net/sunny2038/article/details/9403059

自适应直方图均衡AHE,CLAHE
https://blog.csdn.net/piaoxuezhong/article/details/78271785
AHE算法与经典算法的不同点在于它通过计算图像多个局部区域的直方图，并重新分布亮度，以此改变图像对比度。
所以，该算法更适合于提高图像的局部对比度和细节部分。不过呢，AHE存在过度放大图像中相对均匀区域的噪音的问题。

CLAHE与AHE的不同主要在于其对于对比度的限幅，在CLAHE中，对于每个小区域都必须使用对比度限幅，用来克服AHE的过度放大噪音的问题。 

opencv实现CLAHE的主要步骤，可以分为如下几步:
step1. 扩展图像边界，使其能够刚好切分为若干子块，假设每个子块面积为tileSizeTotal，子块系数 lutScale = 255.0 / tileSizeTotal，对预设的limit做处理：limit = MAX(1, limit * tileSizeTotal / 256)；
step2. 对每个子块，计算直方图；
step3. 对每个子块直方图的每个灰度级，使用预设的limit值做限定，同时统计整个直方图超过limit的像素数；
step4. 计算每个子块的lut累积直方图tileLut，tileLut[i] = sum[i] * lutScale，sum[i] 是累积直方图，lutScale确保tileLut取值在[0, 255]；
step5. 遍历原始图像每个点，考虑该点所在子块及右、下、右下一共4个子块的tileLut，以原始灰度值为索引得到4个值，然后做双线性插值得该点变换后的灰度值；

'''

