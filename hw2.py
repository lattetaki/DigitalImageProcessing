import numpy as np
import cv2

# # 单通道图简单的算术运算测试
# # 生成纯黑纯白的两张图
# size = (160, 160)
# black = np.zeros(size)
# cv2.imwrite('black.jpg', black)
# white = black
# white[:] = 255
# cv2.imwrite('white.jpg', white)
#
#
# # 两张图经过算术运算可得到灰色的图
# gray = (white+black)/4
# cv2.imwrite('gray.jpg', gray)
#
#
#
# 两幅图片叠加效果
# 准备了两张大小相同的图片
img1 = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\limes.jpg")
img2 = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\flowers.jpg")

# 使用了简单的算术运算对图片进行了初步处理 看了一下矩阵好像也没有问题，但是打印出图片之后是纯白，原因暂时未知
# 故使用opencv方法对两张图进行加权相加
#img3 = 0.5*img1 + 0.5*img2
#print(img3)

# img3 = cv2.addWeighted(img1,0.4,img2,0.6,0)
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
#
# # 两张图片相减以检测运动
# # 导入两张有微小变化的图片
# img4 = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\malaoshi.png")
# img5 = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\malaoshi1.png")
#
# # 使用‘-’时，默认的方法是当结果小于0，取%255的值作为替代，效果一般
# img6 = img4-img5
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", img6)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 使用opencv的方法对图像进行相减，结果小于0时会直接取0，效果较好
# img7 = cv2.subtract(img4,img5)
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", img7)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# # 图像乘法
# # 导入一张单通道人物图与一张黑白图
img8 = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\grayman.jpg")
img9 = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\black&white.jpg")
#
# # 人物图与黑白背景相乘时，黑背景被完全遮盖，白背景的位置相乘之后取了负色
# img10 = img8*img9
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", img10)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 使用opencv方法进行相乘,同样取了负色,看上去效果没有'*'好
# img11 = cv2.multiply(img8,img9)
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", img11)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 通过非运算进行图像取反
# 看不懂自己创建的矩阵出了什么问题，自己写的函数一直会报错，暂时用opencv进行非运算
# img12 = cv2.bitwise_not(img9)
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", img12)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

















