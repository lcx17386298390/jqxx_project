##################################################################################
# 数据清理
# 作者: LCX (https://lifescript.top)
# 文件名称：车牌图像提取
# 本程序功能:
# 1. 对CCPD2020数据集的车牌图像根据名称含义进行提取
# 2. 提取车牌图像并保存
###################################################################################

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

class TuXiangTiQu:
    def __init__(self, folder_path):
        self.folder_path = folder_path      # folder_path = 'D:/My_Code_Project/三下机器学习课设/解压数据包/车牌/CCPD2020/test'
        # 图片信息字典(在此处我设置字典，给类实例化后数据都会存储在这里，所以不必留接口返回信息，看类的字典即可)
        self.image_info_dict = {}   # 结构：{图片名称：{图片绝对路径：file_path,面积比：mianjibi, 倾斜度：qinxiedu, 边界坐标：bianjie_zuobiao, 顶点坐标：dingdian_list, 车牌号：car_number}}
        

    # 获取图像信息
    def get_image_info(self, image_name):
        # 图像名称中的字段
        mianjibi = None
        qinxiedu = None
        bianjie_zuobiao = None
        dingdian_list = None
        car_number = None

        # 省份、字母和广告码
        provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
        alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
        ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

        # 使用 '-' 分割图像名称(面积比、倾斜度 2边界框坐标、四个顶点位置、车牌号)
        splits = image_name.split('-')

        # 提取面积比(车牌面积与整个画面面积的面积比。)
        mianjibi = float(splits[0][0]+'.'+splits[0][1:])
        if splits[0][0] != '0':
            mianjibi = float(splits[0][1]+'.'+splits[0][2:])

        # 提取倾斜度(水平倾斜度和垂直倾斜度。)
        qinxiedu = list(map(float, splits[1].split('_')))   # [水平倾斜度, 垂直倾斜度]

        # 提取边界框坐标(左上顶点和右下顶点的坐标。)
        bianjie_zuobiao = [[splits[2].split('_')[0].split('&')[0],splits[2].split('_')[0].split('&')[1]], [splits[2].split('_')[1].split('&')[0],splits[2].split('_')[1].split('&')[1]]] 
        bianjie_zuobiao = [list(map(int, sublist)) for sublist in bianjie_zuobiao] # [[左上顶点x, 左上顶点y], [右下顶点x, 右下顶点y]]

        # 提取四个顶点位置(整个图像中 LP 的四个顶点的精确 （x， y） 坐标。这些坐标从右下角的顶点开始(顺时针))
        dingdian_list = [splits[3].split('_')[0].split('&'), splits[3].split('_')[1].split('&'), splits[3].split('_')[2].split('&'), splits[3].split('_')[3].split('&')]
        dingdian_list = [list(map(int, sublist)) for sublist in dingdian_list] # [[右下角x, 右下角y], [左下角x, 左下角y], [左上角x, 左上角y], [右上角x, 右上角y]]
        # 提取车牌号
        car_number = list(map(int, splits[4].split('_')))
        car_number = provinces[car_number[0]] + \
                            alphabets[car_number[1]] + \
                            ''.join([ads[i] for i in car_number[2:]])
        self.image_info_dict[image_name] = {'file_path': str(self.folder_path+'/'+image_name),'mianjibi': mianjibi, 'qinxiedu': qinxiedu, 'bianjie_zuobiao': bianjie_zuobiao, 'dingdian_list': dingdian_list, 'car_number': car_number}
        return mianjibi, qinxiedu, bianjie_zuobiao, dingdian_list, car_number # 面积比、倾斜度、边界坐标、顶点坐标、车牌号

    # 提取车牌图片
    def get_car_number_image(self, image_name):
        # 车牌的绝对路径
        image_file = os.path.join(self.folder_path, image_name)

        # 获得车牌的各种信息
        mianjibi = self.image_info_dict[image_name]['mianjibi']
        qinxiedu = self.image_info_dict[image_name]['qinxiedu']
        bianjie_zuobiao = self.image_info_dict[image_name]['bianjie_zuobiao']
        dingdian_list = self.image_info_dict[image_name]['dingdian_list']
        car_number = self.image_info_dict[image_name]['car_number']
        
        # 提取车牌图片
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 定义提取框的左上角和右下角坐标
        bbox_top_left = (bianjie_zuobiao[0][0], bianjie_zuobiao[0][1])
        bbox_bottom_right = (bianjie_zuobiao[1][0], bianjie_zuobiao[1][1])

        # 定义车牌的四个顶点的坐标
        plate_points = np.array(dingdian_list, np.int32)

        # 在原图上画出提取框
        image_with_bbox = cv2.rectangle(image.copy(), bbox_top_left, bbox_bottom_right, (0, 255, 0), 2)

        # 在原图上画出车牌的四边形
        image_with_plate = cv2.polylines(image.copy(), [plate_points], isClosed=True, color=(255, 0, 0), thickness=2)

        # 计算车牌的宽和高
        plate_width = np.sqrt((dingdian_list[0][0] - dingdian_list[1][0])**2 + (dingdian_list[0][1] - dingdian_list[1][1])**2)
        plate_height = np.sqrt((dingdian_list[1][0] - dingdian_list[2][0])**2 + (dingdian_list[1][1] - dingdian_list[2][1])**2)
        # 定义转换车牌的四个顶点的坐标
        dst_points = np.float32([[plate_width, plate_height], [0, plate_height], [0, 0], [plate_width, 0]])
        # 提取车牌
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(np.float32(dingdian_list), dst_points)
        # 对图像进行透视变换，得到拉正后的车牌图像
        plate_image = cv2.warpPerspective(image, M, (int(plate_width), int(plate_height)))
        
        # # 测试显示    # 提取图像，暂不显示
        # # 创建一个新的figure
        # plt.figure()

        # # 创建第一个subplot并显示原图
        # plt.subplot(1, 4, 1)
        # plt.imshow(image)
        # plt.title('Original Image')

        # # 创建第二个subplot并显示带有提取框的图
        # plt.subplot(1, 4, 2)
        # plt.imshow(image_with_bbox)
        # plt.title('Image with Bounding Box')

        # # 创建第三个subplot并显示带有车牌的图
        # plt.subplot(1, 4, 3)
        # plt.imshow(image_with_plate)
        # plt.title('Image with Plate')

        # # 车牌
        # plt.subplot(1, 4, 4)
        # plt.imshow(plate_image)
        # plt.title('Plate Image')

        # # 显示所有的subplot
        # plt.show()

        # 返回车牌图像(rgb格式)
        return plate_image
    
    
    def start(self,image_name):
        # 获取图像信息
        self.get_image_info(image_name)
        # 提取车牌图片
        car_number_image =  self.get_car_number_image(image_name)
        self.save_car_number_image(image_name,car_number_image,str(self.folder_path+'/car_number'))

    # 保存车牌图片
    def save_car_number_image(self,image_name, image_file,save_path):
        # 如果目录不存在，则创建它
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = save_path+'/'+image_name
        cv2.imwrite(save_path, cv2.cvtColor(image_file, cv2.COLOR_RGB2BGR))

# # 需要提取图片的时候，实例化类，然后调用start方法即可
# path = "D:\My_Code_Project\Image_Dateset\car_number\CCPD2020\\val"
# files = os.listdir(path)
# txtq = TuXiangTiQu(path)
# for file in files:
#     txtq.start(file)

