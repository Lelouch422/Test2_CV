from Stitcher_0812 import Stitcher
import cv2

# 读取拼接图片
#imageA = cv2.imread("./img/saber_sl.jpg")
#imageB = cv2.imread("./img/saber_sr.jpg")
imageA = cv2.imread("./img/left_01.png")
imageB = cv2.imread("./img/right_01.png")
#imageB = cv2.imread("./img/right_02.png")

# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
#(result, vis) = stitcher.stitch([imageB, imageA], showMatches=True)


# 显示所有图片
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
知识点
特征匹配
https://www.cnblogs.com/wangguchangqing/p/4333873.html
DescriptorMatcher是匹配特征向量的抽象类，在OpenCV2中的特征匹配方法都继承自该类（例如：BFmatcher，FlannBasedMatcher）
DMatcher 是用来保存匹配结果的，主要有以下几个属性distance，trainIdx，queryIdx

KNNMatch，可设置K = 2 ，即对每个匹配返回两个最近邻描述符，仅当第一个匹配与第二个匹配之间的距离足够小时，才认为这是一个匹配。

cv2.findHomography 计算视角变换矩阵,返回(H, status)，H是3x3视角变换矩阵

cv2.warpPerspective 投影变换/透视变换
https://blog.csdn.net/qingyuanluofeng/article/details/51582142
https://blog.csdn.net/on2way/article/details/46801063

'''