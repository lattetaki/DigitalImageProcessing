import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

# 图像频域增强
# 频域锐化与频域平滑实现方式基本相同，差距在滤波器上，故4、5题要求一并实现

# 导入我们经典的camera图
img = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\camera.jpg",0)

# 进行频域变换，最后一步得到的img_dft_new是低频在图像中心的频域图,img_fre是使其能够显示为图像
img_32f = np.float32(img)
img_dft = cv2.dft(img_32f,flags = cv2.DFT_COMPLEX_OUTPUT)
img_center = np.fft.fftshift(img_dft)
img_fre = 20 * np.log(cv2.magnitude(img_center[:, :, 0], img_center[:, :, 1]))

# 制作了滤波器可供选择
# 滤波器的范围暂时只支持在代码中修改，形状为了方便取了正方形而不是圆形
def low_filter(img):
    crow, ccol = int(img.shape[0] / 2), int(img.shape[1] / 2)
    mask = np.zeros((img.shape[0], img.shape[1], 2))
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1
    return mask

def high_filter(img):
    crow, ccol = int(img.shape[0] / 2), int(img.shape[1] / 2)
    mask = np.ones((img.shape[0], img.shape[1], 2))
    mask[crow - 60:crow + 60, ccol - 60:ccol + 60] = 0
    return mask

# 滤波器的值大概在0.11-0.20波动，下面直接取相等会被取0，修改方法①让滤波器能接受这个精度②值先乘个100存进去，用的时候再除
# 已解决 首先要转换mask的dtype，其次是排查出之前版本的小错误，修正后滤波器可用
def butterworth_low(img):
    crow, ccol = int(img.shape[0] / 2), int(img.shape[1] / 2)
    mask = np.zeros((img.shape[0], img.shape[1], 2))
    mask = mask.astype(np.float64)
    for i in range (120):
        for j in range(120):
            mask[crow-60+i][ccol-60+j] = 1/(1+(np.sqrt(i**2+j**2)/60)**2)
    return mask

def butterworth_high(img):
    crow, ccol = int(img.shape[0] / 2), int(img.shape[1] / 2)
    mask = np.zeros((img.shape[0], img.shape[1], 2))
    mask = mask.astype(np.float64)
    for i in range (120):
        for j in range(120):
            mask[crow-60+i][ccol-60+j] = 1-1/(1+np.sqrt(i**2+j**2))
    return mask

# 选择要使用的滤波器
mask = butterworth_low(img)

# 滤波
mask_img = img_center * mask

# 使用np.fft.ifftshift将低频移动到原来的位置
img_idf = np.fft.ifftshift(mask_img)

# 使用cv2.idft进行傅里叶的反变化
img_idf = cv2.idft(img_idf)

# 使用cv2.magnitude转化为空间域内
img_new = cv2.magnitude(img_idf[:, :, 0], img_idf[:, :, 1])

# 将原图和新图一并画出
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.imshow(img_new, cmap='gray')
plt.show()




