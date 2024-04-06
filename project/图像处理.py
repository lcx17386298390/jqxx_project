# 图像去噪灰度处理
import cv2
import numpy as np
from matplotlib import pyplot as plt

def gray_guss(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

# 显示图片
def cv_show(name, img):
    # cv2.imshow(name, img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    pass

# 图像预处理
def image_preprocess(image,filename):
    gray_image = gray_guss(image)
    ret, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
    cv_show(filename,image)
    # 计算二值图像黑白点的个数，处理绿牌照问题，让车牌号码始终为白色
    area_white = 0
    area_black = 0
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            if image[i, j] == 255:
                area_white += 1
            else:
                area_black += 1
    if area_white > area_black:
        ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        cv_show('image',image)
    return image