import numpy as np
import cv2
from skimage import io , data
# chosen = 2
# if chosen == 1:
#
#     # the image need to cut
#     img = cv2.imread("C:\\Users\\12620\\Desktop\\Digital Image Processing\\pythonProject\\images\\")
#     img1 = img[0:227,0:286]
#
#     cv2.namedWindow("Image_jpg", 0)
#     cv2.imshow("Image_jpg", img1)
#     cv2.waitKey(0)
#
#     cv2.destroyAllWindows()
#
#     # create the new image
#     #cv2.imwrite('malaoshi1.png',img1)
#
# if chosen == 2:
#     size = (5487, 3658)
#     black = np.zeros(size)
#     black[:2500] = 255
#     cv2.imwrite('black&white.jpg', black)

img = data.chelsea()
# io.imsave("chelsea.jpg",img)
img1 = data.camera()
io.imsave("camera.jpg",img1)