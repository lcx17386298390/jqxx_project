# import tkinter as tk

# # 工具提示类，鼠标悬停在小部件上时显示工具提示
# class ToolTip:
#     def __init__(self, widget, text):
#         self.widget = widget
#         self.text = text
#         self.tooltip = None
#         self.widget.bind("<Enter>", self.show_tooltip)
#         self.widget.bind("<Leave>", self.hide_tooltip)

#     def show_tooltip(self, event=None):
#         x = y = 0
#         x, y, _, _ = self.widget.bbox("insert")
#         x += self.widget.winfo_rootx() + 25
#         y += self.widget.winfo_rooty() + 20
#         self.tooltip = tk.Toplevel(self.widget)
#         self.tooltip.wm_overrideredirect(True)
#         self.tooltip.wm_geometry(f"+{x}+{y}")
#         tk.Label(self.tooltip, text=self.text, background="#ffffe0", relief="solid", borderwidth=1).pack()

#     def hide_tooltip(self, event=None):
#         if self.tooltip:
#             self.tooltip.destroy()
#             self.tooltip = None
# root = tk.Tk()
# button = tk.Button(root, text="Hover over me!")
# button.pack()
# ToolTip(button, "This is a tooltip!")
# root.mainloop()

# import cv2
# import numpy as np

# img_ = cv2.imread(r"D:\My_Code_Project\Image_Dateset\car_number\val\02845703125-88_271-144&502_426&604-425&584_145&604_144&515_426&502-0_0_3_24_33_26_33_27-124-18.jpg")  # 读取图片
# img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)  # 转换了灰度化

# # 图像阈值化操作——获得二值化图
# ret, img_thre = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 形态学处理:定义矩形结构
# closed = cv2.dilate(img_thre, kernel, iterations=1)  # 闭运算：迭代5次

# height, width = closed.shape[:2]
# # 储存每一列的黑色像素数
# v = [0] * width
# # 储存每一行的黑色像素数
# z = [0] * height
# hfg = [[0 for col in range(2)] for row in range(height)]
# lfg = [[0 for col in range(2)] for row in range(width)]
# box = [0,0,0,0]

# ######水平投影  #统计每一行的黑点数，行分割#######
# a = 0
# emptyImage1 = np.zeros((height, width, 3), np.uint8)
# for y in range(0, height):
#     for x in range(0, width):
#         if closed[y, x] == 0:
#             a = a + 1
#         else:
#             continue
#     z[y] = a
#     a = 0

# # 绘制水平投影图
# l = len(z)
# for y in range(0, height):
#     for x in range(0, z[y]):
#         b = (255, 255, 255)
#         emptyImage1[y, x] = b

# #根据水平投影值选定行分割点
# inline = 1
# start = 0
# j = 0
# # print(height,width)
# # print(z)
# for i in range(0,height):
#     # inline 为起始位置标识，0.95 * width可自行调节，为判断字符位置的条件
#     if inline == 1 and z[i] < 0.95 * width:  #从空白区进入文字区
#         start = i  #记录起始行分割点
#         #print i
#         inline = 0
#     # i - start > 3字符分割长度不小于3，inline为分割终止位置标识，0.95 * width可自行调节，为判断字符位置的条件
#     elif (i - start > 3) and z[i] >= 0.95 * width and inline == 0 :  #从文字区进入空白区
#         inline = 1
#         hfg[j][0] = start - 2  #保存行分割位置
#         hfg[j][1] = i + 2
#         j = j + 1
# ####################### 至此完成行的分割 #################

# #####对每一行垂直投影、分割#####
# a = 0
# for p in range(0, j):
#     # 垂直投影  #统计每一列的黑点数
#     for x in range(0, width):
#         for y in range(hfg[p][0], hfg[p][1]):
#             cp1 = closed[y,x]
#             if cp1 == 0:
#                 a = a + 1
#             else :
#                 continue
#         v[x] = a  #保存每一列像素值
#         a = 0
#     print(v)
#     # 创建空白图片，绘制垂直投影图
#     l = len(v)
#     emptyImage = np.zeros((height, width, 3), np.uint8)
#     for x in range(0, width):
#         for y in range(0, v[x]):
#             b = (255, 255, 255)
#             emptyImage[y, x] = b
#     #垂直分割点
#     incol = 1
#     start1 = 0
#     j1 = 0
#     z1 = hfg[p][0]
#     z2 = hfg[p][1]
#     word = []
#     for i1 in range(0,width):
#         if incol == 1 and v[i1] <= 34 :  #从空白区进入文字区
#             start1 = i1  #记录起始列分割点
#             incol = 0
#         elif (i1 - start1 > 3) and v[i1] > 34 and incol == 0 :  #从文字区进入空白区
#             incol = 1
#             lfg[j1][0] = start1 - 2   #保存列分割位置
#             lfg[j1][1] = i1 + 2
#             l1 = start1 - 2
#             l2 = i1 + 2
#             j1 = j1 + 1
#             cv2.rectangle(img_, (l1, z1), (l2, z2), (255,0,0), 2)
# cv2.imshow('original_img', img_)
# cv2.imshow('erode', closed)
# cv2.imshow('chuizhi', emptyImage)
# cv2.imshow('shuipin', emptyImage1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread(r"D:\My_Code_Project\Image_Dateset\car_number\train\0275-90_269-246&438_534&535-534&528_246&535_247&441_533&438-0_0_3_24_31_29_26_24-187-33.jpg")

# new_img[15:85, 5:55]
# new_img[15:85, 50:100]
# new_img[15:85, 120:170]
# new_img[15:85, 165:215]
# new_img[15:85, 210:260]
# new_img[15:85, 255:305]
# new_img[15:85, 300:350]
# new_img[15:85, 345:395]
# 切割图像
# img[y:y+h, x:x+w]
cropped_img1 = img[10:90, 10:50]
cropped_img2 = img[10:90, 55:95]
cropped_img3 = img[10:90, 125:165]
cropped_img4 = img[10:90, 170:210]
cropped_img5 = img[10:90, 215:255]
cropped_img6 = img[10:90, 260:300]
cropped_img7 = img[10:90, 305:345]
cropped_img8 = img[10:90, 350:390]

# 显示切割后的图像
plt.figure()
plt.subplot(1, 8, 1)
plt.imshow(cropped_img1)
plt.title('cropped_img1')
plt.subplot(1, 8, 2)
plt.imshow(cropped_img2)
plt.title('cropped_img2')
plt.subplot(1, 8, 3)
plt.imshow(cropped_img3)
plt.title('cropped_img3')
plt.subplot(1, 8, 4)
plt.imshow(cropped_img4)
plt.title('cropped_img4')
plt.subplot(1, 8, 5)
plt.imshow(cropped_img5)
plt.title('cropped_img5')
plt.subplot(1, 8, 6)
plt.imshow(cropped_img6)
plt.title('cropped_img6')
plt.subplot(1, 8, 7)
plt.imshow(cropped_img7)
plt.title('cropped_img7')
plt.subplot(1, 8, 8)
plt.imshow(cropped_img8)
plt.title('cropped_img8')
plt.show()
