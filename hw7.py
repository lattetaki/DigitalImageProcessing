import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage


# # 阈值分割法--通过直方图获取阈值
# def threshold_seg(pixel,T):
#     if pixel < T:
#         new_pixel = 0
#     else:
#         new_pixel = 255
#     return new_pixel
#
# img = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\camera.jpg",0)
# # 绘制直方图,找到峰值与谷底
# plt.hist(img.ravel(), 256)
# plt.show()
# #目测得到谷底在80处，选择80作为阈值进行处理
#
# # 创建一个副本
# new_img = img.copy()
# height = len(new_img)
# width = len(new_img[0])
# for i in range(height):
#     for j in range(width):
#         new_img[i][j] = threshold_seg(new_img[i][j],80)
#
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 直方图取阈值改进--选择波峰中间值
# new_img = img*1
# height = len(new_img)
# width = len(new_img[0])
# for i in range(height):
#     for j in range(width):
#         new_img[i][j] = threshold_seg(new_img[i][j],(28+205)/2)
#
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #局部阈值二值化--将图像四等分,原始图像为512*512
# img_part = []
# img_part.append(img[0:256, 0:256].copy())
# img_part.append(img[0:256, 256:512].copy())
# img_part.append(img[256:512, 0:256].copy())
# img_part.append(img[256:512, 256:512].copy())
# #分别获取直方图
# for i in range(4):
#     plt.hist(img_part[i].ravel(), 256,[0,256])
#     plt.show()
# #由直方图求得阈值约为 117 180 90 100
# # 根据阈值分别进行处理
# for i in range(256):
#     for j in range(256):
#         img_part[0][i][j] = threshold_seg(img_part[0][i][j],117)
# for i in range(256):
#     for j in range(256):
#         img_part[1][i][j] = threshold_seg(img_part[1][i][j],180)
# for i in range(256):
#     for j in range(256):
#         img_part[2][i][j] = threshold_seg(img_part[2][i][j],90)
# for i in range(256):
#     for j in range(256):
#         img_part[3][i][j] = threshold_seg(img_part[3][i][j],100)
# img[0:256, 0:256] = img_part[0]
# img[0:256, 256:512] = img_part[1]
# img[256:512, 0:256] = img_part[2]
# img[256:512, 256:512] = img_part[3]
#
# cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg",img )
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 基于种子点生长的分割算法
# # 初始种子选择
def originalSeed(gray, th):
# th为筛选种子点的阈值，高于阈值的点设为255，其余为0，生成种子图ret
    ret, thresh = cv2.cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)  # 二值图，种子区域(不同划分可获得不同种子)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 3×3结构元
    thresh_copy = thresh.copy()  # 复制thresh_A到thresh_copy
    thresh_B = np.zeros(gray.shape, np.uint8)  # thresh_B大小与A相同，像素值为0
    seeds = []  # 为了记录种子坐标
    # 循环，直到thresh_copy中的像素值全部为0
    while thresh_copy.any():
        Xa_copy, Ya_copy = np.where(thresh_copy > 0)  # thresh_A_copy中值为255的像素的坐标
        thresh_B[Xa_copy[0], Ya_copy[0]] = 255  # 选取第一个点，并将thresh_B中对应像素值改为255
        # 连通分量算法，先对thresh_B进行膨胀，再和thresh执行and操作（取交集）
        for i in range(200):
            dilation_B = cv2.dilate(thresh_B, kernel, iterations=1)
            thresh_B = cv2.bitwise_and(thresh, dilation_B)
        # 取thresh_B值为255的像素坐标，并将thresh_copy中对应坐标像素值变为0
        Xb, Yb = np.where(thresh_B > 0)
        thresh_copy[Xb, Yb] = 0
        # 循环，在thresh_B中只有一个像素点时停止
        while str(thresh_B.tolist()).count("255") > 1:
            thresh_B = cv2.erode(thresh_B, kernel, iterations=1)  # 腐蚀操作
        X_seed, Y_seed = np.where(thresh_B > 0)  # 取处种子坐标
        if X_seed.size > 0 and Y_seed.size > 0:
            seeds.append((X_seed[0], Y_seed[0]))  # 将种子坐标写入seeds
        thresh_B[Xb, Yb] = 0  # 将thresh_B像素值置零
    return seeds
#
# # 区域生长
def regionGrow(gray, seeds, thresh, p):
    seedMark = np.zeros(gray.shape)
    # 八邻域
    if p == 8:
        connection = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    elif p == 4:
        connection = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    # seeds内无元素时候生长停止
    while len(seeds) != 0:
        # 栈顶元素出栈
        pt = seeds.pop(0)
        for i in range(p):
            tmpX = pt[0] + connection[i][0]
            tmpY = pt[1] + connection[i][1]
            # 检测边界点
            if tmpX < 0 or tmpY < 0 or tmpX >= gray.shape[0] or tmpY >= gray.shape[1]:
                continue
            if abs(int(gray[tmpX, tmpY]) - int(gray[pt])) < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = 255
                seeds.append((tmpX, tmpY))
    return seedMark
#
# img = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\camera.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # hist = cv2.calcHist([gray], [0], None, [256], [0,256])#直方图
#
# #设置阈值得到种子
# seeds = originalSeed(gray, th=210)
# #设置是否生长的阈值（thresh）以及扩散的方式（p）
# seedMark = regionGrow(gray, seeds, thresh=10, p=8)
# # plt.plot(hist)
# # plt.xlim([0, 256])
# # plt.show()
# cv2.imshow("seedMark", seedMark)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 基于分裂合并的分割算法

def judge(w0, h0, w, h):
# 首先判断方框是否需要再次拆分为四个
    a = img[h0: h0 + h, w0: w0 + w]
    ave = np.mean(a)
    std = np.std(a, ddof=1)  #标准差
    count = 0
    for i in range(w0, w0 + w):
        for j in range(h0, h0 + h):
        #注意！我输入的图片数灰度图，所以直接用的img[j,i]，RGB图像的话每个img像素是一个三维向量，不能直接与avg进行比较大小。
            if abs(img[j, i] - ave) < 1 * std:
                count += 1
    if (count / (w * h)) < 0.95:#合适的点还是比较少，接着拆
        return True
    else:
        return False

##将图像将根据阈值二值化处理，在此默认125
def draw(w0, h0, w, h):
    for i in range(w0, w0 + w):
        for j in range(h0, h0 + h):
            if img[j, i] > 125:
                img[j, i] = 255
            else:
                img[j, i] = 0


def function(w0, h0, w, h):
    if judge(w0, h0, w, h) and (min(w, h) > 5):
        function(w0, h0, int(w / 2), int(h / 2))
        function(w0 + int(w / 2), h0, int(w / 2), int(h / 2))
        function(w0, h0 + int(h / 2), int(w / 2), int(h / 2))
        function(w0 + int(w / 2), h0 + int(h / 2), int(w / 2), int(h / 2))
    else:
        draw(w0, h0, w, h)

img = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\camera.jpg",0)
height, width = img.shape
function(0, 0, width, height)
cv2.imshow('Image',img)
cv2.waitKey()
cv2.destroyAllWindows()



