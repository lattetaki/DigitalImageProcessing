import numpy as np
import cv2

# 图像点运算处理

# ①图像求反
# 导入简单的黑白图进行观察
# img = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\black&white.jpg")
#
# # 对每一个点进行取反操作
# for i in range(5486):
#     for j in range(3658):
#         img[i][j][0] = 0-img[i][j][0]+255
#         img[i][j][1] = 0 - img[i][j][1] + 255
#         img[i][j][2] = 0 - img[i][j][2] + 255
#
# # 将图片输出
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ②增强对比度
# 导入经典camera图
# img1 = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\camera.jpg")

# 线性变换函数
# 对数变换 伽马校正 动态范围压缩等方法与此类似 将下方函数修改即可
# def enhencing_contrast_ratio(pixel):
#     if pixel<80:
#         return pixel/2
#     if pixel >= 80 and pixel <200:
#         return (pixel-80)*(187/120)+40
#     if pixel >=200:
#         return pixel*0.5+128
#
# # 对图像进行处理
# height = len(img1)
# width = len(img1[0])
# for i in range(height):
#     for j in range(width):
#         img1[i][j][0] = enhencing_contrast_ratio(img1[i][j][0])
#         img1[i][j][1] = enhencing_contrast_ratio(img1[i][j][1])
#         img1[i][j][2] = enhencing_contrast_ratio(img1[i][j][2])
#
# # 显示修改后的图片
# # 可能是参数设置的问题 增强效果一般
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # ③伪彩色处理
# # 导入经典camera图
# img2 = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\camera.jpg")
#
# # 三个通道有不同的变换函数
# def trans_red(channel):
#     if channel < 128:
#         return 0
#     if channel >= 128 and channel < 192:
#         return (channel-128)*4
#     if channel >=192:
#         return 255
#
# def trans_green(channel):
#     if channel < 64:
#         return channel*4
#     if channel >= 64 and channel < 192:
#         return 255
#     if channel >= 192:
#         return -4*channel+1023
#
# def trans_blue(channel):
#     if channel <= 64:
#         return 255
#     if channel >64 and channel <= 128:
#         return -4*channel+511
#     if channel > 128:
#         return 0
#
# # 对图像进行处理
# height = len(img2)
# width = len(img2[0])
#
# # opencv的储存顺序为BGR
# for i in range(height):
#     for j in range(width):
#         img2[i][j][0] = trans_blue(img2[i][j][0])
#         img2[i][j][1] = trans_green(img2[i][j][1])
#         img2[i][j][2] = trans_red(img2[i][j][2])
#
# # 显示修改后的图片
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 图像空域的平滑滤波技术

# 加权平均滤波器
# 均值滤波器和高斯滤波器体现在滤波器的不同上，实现方法没有大的差别
# 导入要处理的图片
# img3 = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\QQtest.jpg")
#
# # 手写一个3*3加权平均滤波器
# Filter = [[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]]
#
# # 开始卷积运算,步长为1
# height = len(img3)
# width = len(img3[0])
# for i in range(5):
#     for i in range(1,height-1):
#         for j in range(1,width-1):
#             img3[i][j][0] = img3[i-1][j-1][0] * Filter[2][2] + img3[i-1][j][0] * Filter[2][1] + img3[i-1][j+1][0] * Filter[2][0] + img3[i][j-1][0] * Filter[1][2] + img3[i][j][0] * Filter[1][1] + img3[i][j+1][0] * Filter[1][0] + img3[i+1][j-1][0] * Filter[0][2] + img3[i+1][j][0] * Filter[0][1] + img3[i+1][j+1][0] * Filter[0][0]
# # 处理的是单通道的图片，储存时RGB值保持一致，所以每个通道取值保持一致即可，处理三通道图片时仿照上式分别处理
#             img3[i][j][1] = img3[i][j][0]
#             img3[i][j][2] = img3[i][j][0]
#
# # 显示修改后的图片
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 中值滤波器
# img4 = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\camera.jpg")
#
# # 连续采样五次，取中值
# def median_filter(num1,num2,num3,num4,num5):
#     L = [num1,num2,num3,num4,num5]
#     temp = 0
#     for i in range(5):
#         for j in range(5-i-1):
#             if L[j]>L[j+1]:
#                 temp = L[j]
#                 L[j] = L[j+1]
#                 L[j+1] = temp
#     return L[2]
#
# height = len(img4)
# width = len(img4[0])
# for i in range(0,height):
#     for j in range(2,width-2):
#         img4[i][j][0] = median_filter(img4[i][j-2][0],img4[i][j-1][0],img4[i][j][0],img4[i][j+1][0],img4[i][j+2][0])
# # 处理的是单通道的图片，算出一个值后RGB保持一致即可，如果处理三通道值，每个通道都算一遍即可
#         img4[i][j][1] = img4[i][j][0]
#         img4[i][j][2] = img4[i][j][0]
#
# # 显示修改后的图片
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", img4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 图像空域的锐化滤波技术
img5 = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\camera.jpg")

# # Robert算子
# Robert_w1 = [[-1, 0], [0,1]]
# Robert_w2 = [[0,-1], [1, 0]]
#
# # Robert算子锐化
# height = len(img5)
# width = len(img5[0])
# for i in range(0, height-1):
#     for j in range(0, width-1):
# # 为了简化计算，算子中为0的计算就不写出来了
#         img5[i][j][0] = np.abs(img5[i][j][0]*Robert_w1[1][1]+img5[i+1][j+1][0]*Robert_w1[0][0])+\
#                         np.abs(img5[i+1][j][0]*Robert_w1[0][1]+img5[i][j+1][0]*Robert_w1[1][0])
# # 处理的是单通道的图片，储存时RGB值保持一致，所以每个通道取值保持一致即可，处理三通道图片时仿照上式分别处理
#         img5[i][j][1] = img5[i][j][0]
#         img5[i][j][2] = img5[i][j][0]
#
# # 显示修改后的图片
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", img5)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Sobel算子
# Sobel_w1 = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
# Sobel_w2 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
#
# #Sobel算子锐化
# height = len(img5)
# width = len(img5[0])
# for i in range(1, height-1):
#     for j in range(1, width-1):
#         img5[i][j][0] = np.sqrt(np.abs(img5[i-1][j-1][0] * Sobel_w1[2][2] + img5[i-1][j][0] * Sobel_w1[2][1] + img5[i-1][j+1][0] * Sobel_w1[2][0] +\
#                         img5[i][j-1][0] * Sobel_w1[1][2] + img5[i][j][0] * Sobel_w1[1][1] + img5[i][j+1][0] * Sobel_w1[1][0] +\
#                         img5[i+1][j-1][0] * Sobel_w1[0][2] + img5[i+1][j][0] * Sobel_w1[0][1] + img5[i+1][j+1][0] * Sobel_w1[0][0]) +\
#                         np.abs(img5[i-1][j-1][0] * Sobel_w2[2][2] + img5[i-1][j][0] * Sobel_w2[2][1] + img5[i-1][j+1][0] * Sobel_w2[2][0] +\
#                         img5[i][j-1][0] * Sobel_w2[1][2] + img5[i][j][0] * Sobel_w2[1][1] + img5[i][j+1][0] * Sobel_w2[1][0] +\
#                         img5[i+1][j-1][0] * Sobel_w2[0][2] + img5[i+1][j][0] * Sobel_w2[0][1] + img5[i+1][j+1][0] * Sobel_w2[0][0]))
# # 处理的是单通道的图片，储存时RGB值保持一致，所以每个通道取值保持一致即可，处理三通道图片时仿照上式分别处理
#         img5[i][j][1] = img5[i][j][0]
#         img5[i][j][2] = img5[i][j][0]
#
# # 显示修改后的图片
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", img5)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Laplace算子
Laplace_w1 = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
Laplace_w2 = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]

# Laplace算子锐化
height = len(img5)
width = len(img5[0])
for i in range(1, height-1):
    for j in range(1, width-1):
        img5[i][j][0] = np.sqrt(np.abs(img5[i-1][j-1][0] * Laplace_w1[2][2] + img5[i-1][j][0] * Laplace_w1[2][1] + img5[i-1][j+1][0] * Laplace_w1[2][0] +\
                        img5[i][j-1][0] * Laplace_w1[1][2] + img5[i][j][0] * Laplace_w1[1][1] + img5[i][j+1][0] * Laplace_w1[1][0] +\
                        img5[i+1][j-1][0] * Laplace_w1[0][2] + img5[i+1][j][0] * Laplace_w1[0][1] + img5[i+1][j+1][0] * Laplace_w1[0][0]) +\
                        np.abs(img5[i-1][j-1][0] * Laplace_w2[2][2] + img5[i-1][j][0] * Laplace_w2[2][1] + img5[i-1][j+1][0] * Laplace_w2[2][0] +\
                        img5[i][j-1][0] * Laplace_w2[1][2] + img5[i][j][0] * Laplace_w2[1][1] + img5[i][j+1][0] * Laplace_w2[1][0] +\
                        img5[i+1][j-1][0] * Laplace_w2[0][2] + img5[i+1][j][0] * Laplace_w2[0][1] + img5[i+1][j+1][0] * Laplace_w2[0][0]))
# 处理的是单通道的图片，储存时RGB值保持一致，所以每个通道取值保持一致即可，处理三通道图片时仿照上式分别处理
        img5[i][j][1] = img5[i][j][0]
        img5[i][j][2] = img5[i][j][0]

# 显示修改后的图片
cv2.namedWindow("Image_jpg", 0)
cv2.imshow("Image_jpg", img5)
cv2.waitKey(0)
cv2.destroyAllWindows()






















