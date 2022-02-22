import numpy as np
import cv2

# 伪彩色增强——利用变换函数
img1 = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\camera.jpg")

# numpy储存黑白图像时，同样使用了三个通道，只不过三个值相等，所以直接对每个值修改即可
# def gray2colored(pixel):
#     new_pixel = pixel        # 创建像素点副本
# # 调整蓝色通道
#     if new_pixel[0] <= 30:
#         new_pixel[0] = 255
#     else:
#         new_pixel[0] = 255-new_pixel[0]
# # 调整绿色通道
#     if new_pixel[1] >= 127:
#         new_pixel[1] = 255-new_pixel[1]
# # 调整红色通道
#     if new_pixel[2] <= 30:
#         new_pixel[2] = 0
#     return new_pixel
#
# # 对每个像素逐一进行空域处理
# height = len(img1)
# width = len(img1[0])
# for i in range(height):
#     for j in range(width):
#         img1[i][j] = gray2colored(img1[i][j])
#
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# RGB到HSI的转换处理
def RGB2HSI(pixel):
# 此处可对RGB值进行归一化处理，然后θ用弧度制，不进行此处理,在HSI转化回RGB时注意使用cos时换到弧度制
#     r = pixel[2]/(pixel[0]+pixel[1]+pixel[2])
#     g = pixel[1]/(pixel[0]+pixel[1]+pixel[2])
#     b = 1-r-g
    r = int(pixel[2])
    g = int(pixel[1])
    b = int(pixel[0])
# 求H所用参数θ,为了防止出现分母为0的情况，在分母后都加上了0.000001，
# 显然带来了额外开销，但直接ignore这种情况会有warning，且结果是错误的
    theta = np.arccos(((r-g)+(r-b))/2/(np.sqrt((r-g)**2+(r-b)*(g-b))+0.000001))*180/np.pi
# 计算HSI值
    if g >= b:
        H = theta
    else:
        H = 360 - theta
    S = 1-3*min(r, g, b)/(r + g + b + 0.000001)
    I = (r + g + b)/3

    HSI_pixel = [H,S,I]
    return HSI_pixel

def HSI2RGB(pixel):
    H = pixel[0]
    S = pixel[1]
    I = pixel[2]
    if H<0 or H>360:
        print("there's something wrong with HSI")
        return
    if H<120:
        B = I*(1-S)
        R = I*(1+(S*np.cos(H*np.pi/180)/np.cos((60-H)*np.pi/180)))
        G = 3*I-(R+B)
    if H<240:
        H = H - 120
        R = I*(1-S)
        G = I*(1+S*np.cos(H*np.pi/180)/np.cos((60-H)*np.pi/180))
        B = 3*I-(R+G)
    else:
        H = H-240
        G = I*(1-S)
        B = I*(1+S*np.cos(H)/np.cos((60-H)*np.pi/180))
        R = 3*I-(G+B)
    new_RGB = [B, G, R]
    return new_RGB

#  对图像进行处理
img2 = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\cyberpunk-women.jpg")
# height = len(img2)
# width = len(img2[0])
# for i in range(height):
#     for j in range(width):
#         img2[i][j] = RGB2HSI(img2[i][j])
# # 对I进行简单压缩
#         img2[i][j][2] = 0.5 * img2[i][j][2]
#         img2[i][j] = HSI2RGB(img2[i][j])
HSI_IMG = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
# height = len(HSI_IMG)
# width = len(HSI_IMG[0])
# for i in range(height):
#     for j in range(width):
#         HSI_IMG[i][j][2] = 0.5*HSI_IMG[i][j][2]
# new_img2 = cv2.cvtColor(HSI_IMG,cv2.COLOR_HSV2BGR)
# new_img2 = cv2.imread("flowers.jpg")
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", new_img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print(HSI_IMG)






