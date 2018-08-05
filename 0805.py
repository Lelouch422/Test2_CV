import numpy as np
import cv2 as cv
filename = './img/chessboard.png'
img = cv.imread(filename)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img_gray = cv.imread(filename,0)
print(gray)
print(img_gray)

gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv.imshow('dst',img)

dst = np.abs(dst)
cv.imshow('dst0',dst)

if cv.waitKey(0)& 0xff == 27:
    cv.destroyAllWindows()
    
    
    
'''
知识点：
1.形态学操作
https://blog.csdn.net/sunny2038/article/details/9137759
eroded = cv2.erode(img,kernel) 腐蚀 
dilated = cv2.dilate(img,kernel) 膨胀

结构元素kernel定义2种方法：
使用其自带的getStructuringElement函数：kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
椭圆（MORPH_ELLIPSE），十字形结构（MORPH_CROSS），矩形（MORPH_RECT）
使用NumPy的ndarray来定义一个结构元素：NpKernel = np.uint8(np.ones((3,3)))

closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) 闭运算：先膨胀后腐蚀，用来连接被误分为许多小块的对象
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 开运算：先腐蚀后膨胀，用于移除由图像噪音形成的斑点

2.Harris角点
https://www.cnblogs.com/DOMLX/p/8763369.html
• img - 数据类型为 float32 的输入图像。
• blockSize - 角点检测中要考虑的领域大小。
• ksize - Sobel 求导中使用的窗口大小
• k - Harris 角点检测方程中的自由参数,取值参数为 [0,04,0.06].

3.其他
cv2.cvtColor(input_image , flag) 颜色空间转换, flag是转换类型：cv2.COLOR_BGR2GRAY,cv2.COLOR_BGR2HSV
cv2.waitKey()：键盘绑定函数，共一个参数，表示等待毫秒数，将等待特定的几毫秒，看键盘是否有输入，返回值为ASCII值。如果其参数为0，则表示无限期的等待键盘输入。
cv2.destroyAllWindows()：删除建立的全部窗口
'''