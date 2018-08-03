import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('opencv.png',0) #直接读为灰度图像
for i in range(2000): #添加点噪声
    temp_x = np.random.randint(0,img.shape[0])
    temp_y = np.random.randint(0,img.shape[1])
    img[temp_x][temp_y] = 255

blur_1 = cv2.GaussianBlur(img,(5,5),0)

blur_2 = cv2.medianBlur(img,5)

plt.subplot(1,3,1),plt.imshow(img,'gray')#默认彩色，另一种彩色bgr
plt.subplot(1,3,2),plt.imshow(blur_1,'gray')
plt.subplot(1,3,3),plt.imshow(blur_2,'gray')
plt.show()



'''
#知识点
1.Jupyter
https://blog.csdn.net/to_baidu/article/details/52609115
修改jupyter notebook工作路径，注意：不能从win快捷栏启动

2.numpy
Numpy创建随机数组np.random
均匀分布
np.random.rand(10, 10)创建指定形状(示例为10行10列)的数组(范围在0至1之间)
np.random.uniform(0, 100)创建指定范围内的一个数
np.random.randint(0, 100) 创建指定范围内的一个整数

3.cv - img
图像宽 img.shape[0]
图像长 img.shape[1]
修改图像像素值：img.[x][y] = M
cv2.imread
使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255
imread('xxx',0)直接读为灰度图像

https://blog.csdn.net/renelian1572/article/details/78761278
Python 中各种imread函数的区别与联系

4.matplotlib
pylab将pyplot与numpy合并为一个命名空间。这对于交互式工作很方便，但是对于编程来说，建议将名称空间分开
import numpy as np
import matplotlib.pyplot as plt
https://blog.csdn.net/Notzuonotdied/article/details/77876080
Python Matplotlib简易教程
https://blog.csdn.net/sinat_34022298/article/details/76348969
Matplotlib.pyplot 常用方法（一）

5.Others
机器学习三剑客之Numpy https://www.jianshu.com/p/83c8ef18a1e8
机器学习三剑客之Pandas https://www.jianshu.com/p/7414364992e4
机器学习三剑客之Matplotlib https://www.jianshu.com/p/f2782e741a75

《Sklearn 与 TensorFlow 机器学习实用指南》
https://github.com/apachecn/hands_on_Ml_with_Sklearn_and_TF
https://www.jianshu.com/u/b508a6aa98eb
'''