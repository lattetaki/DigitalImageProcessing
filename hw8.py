import numpy as np
import cv2


# 膨胀
# img为原图，operator为结构元素，mark为结构元素原点
# 根据mark所在位置的不同，原图需要扩展的行数或列数不确定，索性不在函数中扩展原图，需要处理时在过程中扩展
def Dilation(img, operator, mark):
    new_img = np.zeros((img.shape[0], img.shape[1]))
    new_img[::] = 255
    for i in range(img.shape[0] - operator.shape[0]):
        for j in range(img.shape[1] - operator.shape[1]):
            flag = 0  # 标志是否有公共元素
            for x in range(operator.shape[0]):
                for y in range(operator.shape[1]):
                    if operator[x, y] == 0 and img[x + i, y + j] == 0:
                        flag += 1
                        break
                if flag:
                    break
            if flag:
                new_img[i + mark[0], j + mark[1]] = 0
    return new_img


# 腐蚀
def Erosoin(img, operator, mark):
    new_img = np.zeros((img.shape[0], img.shape[1]))
    new_img[::] = 255
    for i in range(img.shape[0] - operator.shape[0]):
        for j in range(img.shape[1] - operator.shape[1]):
            flag = 0  # 标志是否有非公共元素
            for x in range(operator.shape[0]):
                for y in range(operator.shape[1]):
                    if operator[x, y] == 0 and img[x + i, y + j] == 255:
                        flag += 1
                        break
                if flag:
                    break
            if flag == 0:
                new_img[i + mark[0], j + mark[1]] = 0
    return new_img


def opening(img, operator, mark):
    new_img = Erosoin(img, operator, mark)
    opening_img = Dilation(new_img, operator, mark)
    return opening_img


def closing(img, operator, mark):
    new_img = Dilation(img, operator, mark)
    closing_img = Erosoin(img, operator, mark)
    return closing_img


def Tophat(img, operator, mark):
    new_img = opening(img, operator, mark).astype(int)  # cv2.subtract一直报错 遂自己写
    Tophat_img = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x = img[i, j] - new_img[i, j]
            if x < 0:
                x = 0
            Tophat_img[i, j] = x
    return Tophat_img


def Blackhat(img, operator, mark):
    new_img = closing(img, operator, mark).astype(int)  # opening的结果是浮点数组，若不转化为整形数，算术运算会报错
    Blackhat_img = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x = img[i, j] - new_img[i, j]
            if x < 0:
                x = 0
            Blackhat_img[i, j] = x
    return Blackhat_img

def Hit(img, operator, mark):
    new_img = np.zeros((img.shape[0], img.shape[1]))
    new_img[::] = 255
    for i in range(img.shape[0] - operator.shape[0]):
        for j in range(img.shape[1] - operator.shape[1]):
            flag = 0  # 标志是否有不同元素
            for x in range(operator.shape[0]):
                for y in range(operator.shape[1]):
                    if operator[x, y] != img[x + i, y + j]:
                        flag += 1
                        break
                if flag:
                    break
            if flag == 0:
                new_img[i + mark[0], j + mark[1]] = 0
    return new_img

# 边缘提取
def Edgeextraction(img, operator, mark):
    new_img = Erosoin(img, operator, mark).astype(int)
    edge_img = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x = img[i, j] - new_img[i, j]
            if x < 0:
                x = 0
            edge_img[i, j] = x
    return edge_img

# 图像细化
def Thining(img, operator, mark, time = 0):
    if time == 1:   #递归次数默认为4次，即结构元素四个方向都用来细化一遍
        return
    new_img = Hit(img, operator, mark).astype(int)
    thining_img = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x = img[i, j] - new_img[i, j]
            if x < 0:
                x = 0
            thining_img[i, j] = x
    Thining(thining_img,np.rot90(operator,time+1),mark,time+1)   #将结构元素翻转90度，递归计算
    return thining_img

# 连通分量提取
# label是一个list，初始元素为选择用于连通分量提取的点
def Connectedextraction(img, operator, mark, label):
    pass

img = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\camera.jpg", 0)

# 将灰度图转化为二值图
ret, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

# 选择结构元素
# operator = np.array([(0,255),(0,0)])     # L型
operator = np.array([(255, 0, 255), (0, 0, 0), (255, 0, 255)])  # 十字型
# operator = np.array([(255,255,255),(0,0,0),(255,255,255)])     # 一字型
# operator = np.array([(1,1,1),(1,1,1),(1,1,1)])    #8邻接


result = Tophat(binary_img, operator, [0, 0])
cv2.namedWindow("Image_jpg", 0)
# cv2.imshow("Image_jpg", binary_img)
cv2.imshow("Image_jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
