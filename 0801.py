# %matplotlib inline
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
#img = cv.imread('.\\opencv.png')
img = cv.imread('.\\saber.jpg')
kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)
dst2 = cv.medianBlur(img,5)
dst3 = cv.GaussianBlur(img,(5,5),0)
plt.subplot(141),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(dst),plt.title('Avg')
plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(dst2),plt.title('Median')
plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(dst3),plt.title('Gauss')
plt.xticks([]), plt.yticks([])
plt.show()

'''
#知识点
1.Jupyter
https://blog.csdn.net/red_stone1/article/details/72858962
Shift+Enter 运行当前cell
Cell -> Run All 运行所有cell
加标题：
Insert->Insert cell above，修改cell type为Markdown
# : First level title
## : Second level title

https://blog.csdn.net/DataCastle/article/details/78890469
常用的Markdown用法
魔法函数
%timeit 测试单行语句的执行时间

2.iPython
https://blog.csdn.net/ztf312/article/details/78677093
ipython支持tab补全
ipython有很多magic函数
ipython有很多快捷键: Ctrl+L 清屏

3. numpy
https://www.jianshu.com/p/83c8ef18a1e8 (可继续学习）

# 将列表转换为数组
a = [1, 2, 3, 4]
b = np.array(a)
数组元素个数
b.size
数组形状
b.shape
数组维度
b.ndim
数组元素类型
b.dtype
快速创建N维1/0数组
array_one = np.ones([10, 10])
array_zero = np.zeros([10, 10])

4. opencv卷积
https://blog.csdn.net/on2way/article/details/46828567 (连载，可继续学习）
cv2.filter2D() 滤波函数，参数（src，ddpeth，kernel），当ddepth输入值为-1时，目标图像和原图像深度保持一致。
cv2.blur() 均值滤波,参数(img,(5,5))
cv2.GaussianBlur()高斯滤波，参数(img,(5,5),0)
cv2.medianBlur() 中值滤波，参数(img,5)

5. 红色石头
https://blog.csdn.net/red_stone1
http://redstonewill.com/

'''