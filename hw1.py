# 使用opencv-python
import numpy as np
import cv2

# 在指定路径读取图片，不可以像Windows中一样使用'\',需要使用'\\'
img = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\miku.jpg")

# 创建窗口，第一个参数为窗口名，第二个参数表示窗口大小可自由改变
cv2.namedWindow("Image_jpg", cv2.WINDOW_NORMAL)

# 在窗口中显示图片
cv2.imshow("Image_jpg", img)

# 参数表示窗口自动关闭等待的时间，0为不自动关闭
cv2.waitKey(0)

# 同样的方式测试不同类型的图片
img1 = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\kaisa.png")
cv2.namedWindow("Image_png", cv2.WINDOW_NORMAL)
cv2.imshow("Image_png", img1)
cv2.waitKey(0)
img2 = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\kaisa.tif")
cv2.namedWindow("Image_tif", cv2.WINDOW_NORMAL)
cv2.imshow("Image_tif", img2)
cv2.waitKey(0)

# 将窗口释放
cv2.destroyAllWindows()

# 截取图像中的任意一部分并显示
img3 = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\RaidenShogun.jpg")
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
# 与之前不同的地方在下面这一行，图像名称括号内第一个参数为y，第二个参数为x，给出指定坐标即可得到指定范围内的截图并显示
cv2.imshow("Image", img3[100:900,200:1050])
cv2.waitKey(0)
cv2.destroyAllWindows()



