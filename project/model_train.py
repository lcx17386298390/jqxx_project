# 模型训练类
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Add, Lambda, Reshape, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import keras
import 图像提取
import 图像处理
import json

# 定义模型训练类
class Model_train:
    def __init__(self, master_App, master_Train, SetPage, model_type='CNN'):
        # 设置顶层（切换训练状态）
        self.master_App = master_App
        # 设置上层(主要用来更改进度条状态)
        self.master_Train = master_Train
        # 提取设置界面（方便取得设置的值）
        self.SetPage = SetPage
        # 设置模型结构类别
        self.model_type = model_type
        # 车牌所有的字符类别
        self.num_classes = 65
        # 图片的高和宽
        self.img_height, self.img_width = 20, 20
        # 设置训练是否人为中止标志
        self.train_is_stop = False
        # 这里可以设置一些训练参数（比如Epoch总个数，tatch_size个数）->后续可以在界面上设置
        self.epochs = 0
        self.batch_size = 0
        self.test_size = 0
        self.random_state = 0
        # 当前epoch和batch数
        self.current_epoch = 0
        self.current_batche = 0
        # 训练图片总数量（用于进度条，训练数量= 图片总数*（1-testsize））
        self.train_image_nums = int()
        # 识别模型
        self.shibie_model = None
        # 省份、字母和广告码
        self.provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
        self.alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
        self.ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
        # 预训练车牌对应的标签字典
        self.provinces_after = ['53', '25', '28', '59', '27', '48', '36', '33', '30', '24', '47', '64', '29', '37', '19', '34', '58', '16', '55', '60', '22', '43', '13', '21', '61', '63', '46', '20', '42', '39', '56']
        self.alphabets_after = ['10', '11', '12', '14', '15', '17', '18', '23', '26', '31', '32', '35', '38', '40', '41', '44', '45', '49', '50', '51', '52', '54', '57', '62']
        self.ads_after = ['10', '11', '12', '14', '15', '17', '18', '23', '26', '31', '32', '35', '38', '40', '41', '44', '45', '49', '50', '51', '52', '54', '57', '62', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        # 创建模型
        self.model = self.create_model(model_type=model_type)

        # 加载 JSON 文件
        with open('D:\My_Code_Project\三下机器学习课设\解压数据包\单数字\VehicleLicense\dataset_info.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 获取标签字典
        self.label_dict = data['label_dict']


    # 训练模型(默认训练单字符识别模型, 传入参数为'single_number'则训练单字符识别模型，否则训练多字符识别模型)
    def fit_model(self, train_type='single_number', txt_file_path=None, img_dir_path=None):
        self.epochs = int(self.master_App.frames[self.SetPage].epoch_input_entry.get())
        self.batch_size = int(self.master_App.frames[self.SetPage].batch_size_input_entry.get())
        self.test_size = float(self.master_App.frames[self.SetPage].test_size_input_entry.get())
        self.random_state = int(self.master_App.frames[self.SetPage].random_state_input_entry.get())
        self.images, self.labels = [], []
        my_callback = self.MyCallback(self)
        if train_type == 'single_number':
            if txt_file_path is None:
                raise ValueError('单字符训练txt文件路径为空')
            else:
                self.images, self.labels = self.load_single_num_data(txt_file_path)
        elif train_type == 'car_number':
            if img_dir_path is None:
                raise ValueError('车牌训练文件夹路径为空')
            else:
                self.images, self.labels = self.load_data(img_dir_path)
                print("labels结构",self.labels)
                self.master_App.log_frame.add_log('labels结构：{}'.format(self.labels), 'info')
        self.train_image_nums = int(len(self.images)*(1-float(self.master_App.frames[self.SetPage].test_size_input_entry.get())))
        pre_images = self.images
        self.images = []
        # 判断是否开启预训练模式
        if self.master_App.on_pre_train:
            if self.model_type == 'CNN':
                for img in pre_images:
                    img = cv2.resize(img, (20, 20))
                    self.images.append(img)
            elif self.model_type == 'LeNet-5':
                for img in pre_images:
                    img = cv2.resize(img, (32, 32))
                    self.images.append(img)
            elif self.model_type == 'AlexNet':
                for img in pre_images:
                    img = cv2.resize(img, (224, 224))
                    img = np.expand_dims(img, axis=-1)
                    self.images.append(img)
            elif self.model_type == 'VGGNet-16':
                for img in pre_images:
                    img = cv2.resize(img, (224, 224))
                    # 将灰度图像复制三次，形成一个RGB图像
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    # img = np.repeat(img, 3, axis=-1)  # 现在x_rgb的形状是(224, 224, 3)
                    self.images.append(img)
            elif self.model_type == 'ResNet-50':
                for img in pre_images:
                    img = cv2.resize(img, (32, 32))
                    self.images.append(img)
            elif self.model_type == 'InceptionV3':
                for img in pre_images:
                    img = cv2.resize(img, (75, 75))
                    self.images.append(img)
        else:   # 非预训练模式只开启CNN模型
            if self.model_type == 'CNN':
                self.images = pre_images
                
        self.images = np.array(self.images, dtype='float32')
        self.labels = np.array(self.labels)
        self.images /= 255
        self.labels = to_categorical(self.labels, self.num_classes) # 将类别标签转换为one-hot编码
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.images, self.labels, test_size=self.test_size, random_state=self.random_state)
        history = self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_data=(self.x_test, self.y_test), callbacks=[my_callback])
        self.shibie_model = self.model
        return history
    
    # 定义中止训练的回调函数
    class MyCallback(keras.callbacks.Callback):
        def __init__(self, master):
            self.master_ModelTrain = master
        def on_epoch_end(self, epoch, logs=None):
            self.master_ModelTrain.current_epoch = epoch+1
            if self.master_ModelTrain.master_App.train_status.is_set():
                self.master_ModelTrain.model.stop_training = True
                self.master_ModelTrain.train_is_stop = True
                self.master_ModelTrain.master_App.train_is_stop = True
        def on_train_batch_end(self, batch, logs=None):
            self.master_ModelTrain.current_batche = (self.master_ModelTrain.train_image_nums/self.master_ModelTrain.batch_size)*self.master_ModelTrain.current_epoch + batch
            # 进度条参数更改
            self.master_ModelTrain.master_Train.progress_nums = self.master_ModelTrain.current_batche
            self.master_ModelTrain.master_Train.progress_nums_all = (self.master_ModelTrain.train_image_nums/self.master_ModelTrain.batch_size)*self.master_ModelTrain.epochs
            if self.master_ModelTrain.master_App.train_status.is_set():
                self.master_ModelTrain.model.stop_training = True
                self.master_ModelTrain.train_is_stop = True
                self.master_ModelTrain.master_App.train_is_stop = True

    # 加载车牌数据  # 此处是返回的是车牌的标签和图片
    def load_data(self, img_dir):
        images = []
        labels = []
        self.master_App.log_frame.add_log('数据提取中，请稍等······', 'info')
        # 开启预训练模式
        if self.master_App.on_pre_train:
            # 也就是字符训练，只取车牌中的字符，切割车牌图片
            for filename in os.listdir(img_dir):
                if self.master_App.train_is_stop:
                    print('训练中止')
                    self.master_App.log_frame.add_log('训练中止', 'info')
                    break
                img = cv2.imread(os.path.join(img_dir, filename))
                new_img = 图像处理.image_preprocess(img,filename)
                images.append(new_img[10:90, 10:50])
                images.append(new_img[10:90, 55:95])
                images.append(new_img[10:90, 125:165])
                images.append(new_img[10:90, 170:210])
                images.append(new_img[10:90, 215:255])
                images.append(new_img[10:90, 260:300])
                images.append(new_img[10:90, 305:345])
                images.append(new_img[10:90, 350:390])
                label = list(map(int,filename.split('-')[4].split('_')))
                labels.append(int(self.provinces_after[label[0]]))
                labels.append(int(self.alphabets_after[label[1]]))
                for i in label[2:]:
                    labels.append(self.ads_after[i])

        else:   # 未开启预训练，整个车牌图片作为训练集
            for filename in os.listdir(img_dir):
                if self.master_App.train_is_stop:
                    print('训练中止')
                    self.master_App.log_frame.add_log('训练中止', 'info')
                    break
                img = cv2.imread(os.path.join(img_dir, filename))
                img = 图像处理.image_preprocess(img,filename)
                img = cv2.resize(img, (400, 100))   # 400为宽，100为高
                img = np.reshape(img, (100, 400, 1)) # 100为高，400为宽
                label = list(map(int,filename.split('-')[4].split('_')))
                if img is not None:
                    images.append(img)
                    labels.append(label)  # 假设文件名的格式为"label_index.jpg"
        self.master_App.log_frame.add_log('数据提取完成', 'info')
        print('len_label:',len(labels))
        print('len_images:',len(images))
        self.master_App.log_frame.add_log('len_label:{};  len_images:{}'.format(len(labels),len(images)), 'info')
        return images, labels
    

    # 加载单字符数据（传入txt文件路径）:预训练，可以保存训练好的模型
    def load_single_num_data(self, txt_path):
        images = []
        labels = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                img = cv2.imread(line[0], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(int(line[1]))
        return images, labels


    # 创建模型
    def create_model(self,model_type):
        model = Sequential()
        # 判断是否开启预训练模式
        if self.master_App.on_pre_train:
            if model_type == 'CNN':
                model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(20, 20, 1)))
                model.add(Conv2D(64, (3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))
                model.add(Flatten())
                model.add(Dense(128, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(self.num_classes, activation='softmax'))
            # LeNet-5结构
            elif model_type == 'LeNet-5':
                model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(32,32,1), padding="same"))
                model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
                model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
                model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
                model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
                model.add(Flatten())
                model.add(Dense(84, activation='tanh'))
                model.add(Dense(self.num_classes, activation='softmax'))
            # AlexNet结构
            elif model_type == 'AlexNet':
                # 添加第一层卷积层，有 96 个 11x11 的卷积核，步长为 4，激活函数为 ReLU
                model.add(Conv2D(filters=96, input_shape=(224,224,1), kernel_size=(11,11), strides=(4,4), padding='valid'))
                model.add(Activation('relu'))
                # 添加最大池化层，池化窗口大小为 2x2
                model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
                # 添加第二层卷积层，有 256 个 11x11 的卷积核，步长为 1，激活函数为 ReLU
                model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
                model.add(Activation('relu'))
                # 添加最大池化层，池化窗口大小为 2x2
                model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
                # 添加第三层卷积层，有 384 个 3x3 的卷积核，步长为 1，激活函数为 ReLU
                model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
                model.add(Activation('relu'))
                # 添加第四层卷积层，有 384 个 3x3 的卷积核，步长为 1，激活函数为 ReLU
                model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
                model.add(Activation('relu'))
                # 添加第五层卷积层，有 256 个 3x3 的卷积核，步长为 1，激活函数为 ReLU
                model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
                model.add(Activation('relu'))
                # 添加最大池化层，池化窗口大小为 2x2
                model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
                # 将特征图展平为一维的向量
                model.add(Flatten())
                # 添加第一个全连接层，有 4096 个神经元，激活函数为 ReLU
                model.add(Dense(4096, input_shape=(224*224*3,)))
                model.add(Activation('relu'))
                # 添加 Dropout 层，防止过拟合
                model.add(Dropout(0.4))
                # 添加第二个全连接层，有 4096 个神经元，激活函数为 ReLU
                model.add(Dense(4096))
                model.add(Activation('relu'))
                # 添加 Dropout 层，防止过拟合
                model.add(Dropout(0.4))
                # 添加第三个全连接层，有 1000 个神经元，激活函数为 ReLU
                model.add(Dense(1000))
                model.add(Activation('relu'))
                # 添加 Dropout 层，防止过拟合
                model.add(Dropout(0.4))
                # 添加输出层，有 17 个神经元（对应于 17 个类别），激活函数为 softmax
                # model.add(Dense(17))
                model.add(Dense(self.num_classes))
                model.add(Activation('softmax'))
            # VGGNet-16
            elif model_type == 'VGGNet-16':
                # model.add(Conv2D(64, (3, 3), input_shape=(224, 224, 1), padding='same', activation='relu'))
                # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
                # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                # model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
                # model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
                # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                # model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
                # model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
                # model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
                # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
                # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
                # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
                # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
                # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
                # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
                # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                # model.add(Flatten())
                # model.add(Dense(4096, activation='relu'))
                # model.add(Dropout(0.5))
                # model.add(Dense(4096, activation='relu'))
                # model.add(Dropout(0.5))
                # model.add(Dense(self.num_classes, activation='softmax'))
                model = VGG16(include_top=False, weights="imagenet", input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=self.num_classes)
            # ResNet-50结构
            elif model_type == 'ResNet-50':
                # model.add(ZeroPadding2D((3, 3), input_shape=(self.img_height, self.img_width, 1)))
                # model.add(Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer='glorot_uniform'))
                # model.add(BatchNormalization(axis=3, name='bn_conv1'))
                # model.add(Activation('relu'))
                # model.add(MaxPooling2D((3, 3), strides=(2, 2)))
                # model.add(Conv2D(64, (1, 1), strides=(1, 1), kernel_initializer='glorot_uniform', name='res2a_branch2a'))
                # model.add(BatchNormalization(axis=3, name='bn2a_branch2a'))
                # model.add(Activation('relu'))
                # model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', padding='same', name='res2a_branch2b'))
                # model.add(BatchNormalization(axis=3, name='bn2a_branch2b'))
                # model.add(Activation('relu'))
                # model.add(Conv2D(256, (1, 1), strides=(1, 1), kernel_initializer='glorot_uniform', name='res2a_branch2c'))
                # model.add(Conv2D(256, (1, 1), strides=(1, 1), kernel_initializer='glorot_uniform', name='res2a_branch1'))
                # model.add(BatchNormalization(axis=3, name='bn2a_branch2c'))
                # model.add(BatchNormalization(axis=3, name='bn2a_branch1'))
                # model.add(Add()([model.layers[-1].output, model.layers[-2].output]))
                # # model.add(Lambda(lambda x: x[0] + x[1])([model.layers[-1].output, model.layers[-2].output]))
                # model.add(Activation('relu'))
                # model.add(Conv2D(64, (1, 1), strides=(1, 1), kernel_initializer='glorot_uniform', name='res2b_branch2a'))
                # model.add(BatchNormalization(axis=3, name='bn2b_branch2a'))
                # model.add(Activation('relu'))
                # model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', padding='same', name='res2b'))
                model = ResNet50(include_top=False, weights="imagenet", input_tensor=None, input_shape=(32, 32, 1), pooling=None, classes=self.num_classes)
            # InceptionV3结构
            elif model_type=='InceptionV3':
                from keras.applications.inception_v3 import InceptionV3
                model = InceptionV3(include_top=False, weights="imagenet", input_tensor=None, input_shape=(75, 75, 1), pooling=None, classes=self.num_classes)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:   # 非预训练模式只开启CNN模型
            if model_type == 'CNN':
                # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 400, 1)))
                # model.add(Conv2D(64, (3, 3), activation='relu'))
                # model.add(MaxPooling2D(pool_size=(2, 2)))
                # model.add(Dropout(0.25))
                # model.add(Flatten())
                # model.add(Dense(128, activation='relu'))
                # model.add(Dropout(0.5))
                # model.add(Reshape((8,-1)))
                # model.add(TimeDistributed(Dense(self.num_classes, activation='sigmoid')))
                # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # 使用binary_crossentropy损失

                model = Sequential()
                model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 400, 1)))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Conv2D(64, (3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Flatten())
                model.add(Dense(128, activation='relu'))
                model.add(Dense(8 * self.num_classes, activation='softmax'))  # 输出层有8*num_classes个神经元
                model.add(Reshape((8, self.num_classes)))  # 将输出重塑为8个字符，每个字符num_classes个类别
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    # 预测函数
    def predict(self, img, model):
        # 判断是否开启预训练模式
        img = 图像处理.image_preprocess(img, 'test_image')
        cv2.imshow('test_image', img)
        cv2.waitKey(0)
        if self.master_App.on_pre_train:
            if self.model_type == 'CNN':
                img = cv2.resize(img, (20, 20))
                cv2.imshow('test_image', img)
                cv2.waitKey(0)
                img = np.reshape(img, (1, 20, 20, 1))
            elif self.model_type == 'LeNet-5':
                img = cv2.resize(img, (32, 32))
                img = np.reshape(img, (1, 32, 32, 1))
            elif self.model_type == 'AlexNet':
                img = cv2.resize(img, (224, 224))
                img = np.expand_dims(img, axis=-1)
                img = np.repeat(img, 3, axis=-1)
            elif self.model_type == 'VGGNet-16':
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = np.reshape(img, (1, 224, 224, 3))
            elif self.model_type == 'ResNet-50':
                img = cv2.resize(img, (32, 32))
                img = np.expand_dims(img, axis=-1)
                img = np.reshape(img, (1, 32, 32, 1))
            elif self.model_type == 'InceptionV3':
                img = cv2.resize(img, (75, 75))
                img = np.reshape(img, (1, 75, 75, 1))
        else:   # 非预训练模式只开启CNN模型
            if self.model_type == 'CNN':
                img = cv2.resize(img, (400, 100))   # 400为宽，100为高
                img = np.reshape(img, (1, 100, 400, 1))     # 100为高，400为宽
        img = img.astype('float32')
        img /= 255
        result = model.predict(img)
        # 输出预测结果
        result = np.argmax(result, axis=-1)
        return result
    
    
    # 保存模型
    def save_model(self, model_path):
        model_path = os.path.join(model_path, self.model_type)  # 加上模型结构类型
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # 获取目录下的所有子目录
        dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        # 计算子目录的数量
        num_dirs = len(dirs)+1
        self.save_train_file_floder = os.path.join(model_path, 'train'+str(num_dirs))
        if not os.path.exists(self.save_train_file_floder):
            os.makedirs(self.save_train_file_floder)
        self.save_file_path = os.path.join(self.save_train_file_floder, 'model.h5')
        self.model.save(self.save_file_path)
        print('模型保存成功，保存路径为:', os.path.abspath(self.save_file_path))
        self.master_App.log_frame.add_log('模型保存成功，保存路径为:{}'.format(os.path.abspath(self.save_file_path)), 'info')
    
    # 学习效果查看
    def train_result_view(self,history):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        self.master_App.log_frame.add_log('Test loss:{};  Test accuracy:{}'.format(score[0],score[1]), 'info')
        # 绘制训练 & 验证的准确率值
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        # 绘制训练 & 验证的损失值
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        save_path = os.path.join(self.save_train_file_floder, '训练效果.png')
        plt.savefig(save_path)
        plt.show()
    
    # 加载模型(仅加载模型，不进行训练)
    def load_model(self, model_path):
        # 判断是否有模型文件
        if os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
            except Exception as e:
                print('模型加载失败，将重新预训练模型')
                self.master_App.log_frame.add_log('模型加载失败，将重新预训练模型', 'info')
                return None
            else:
                print('模型加载成功')
                self.master_App.log_frame.add_log('模型加载成功', 'info')
                return self.model
        else:
            print('模型文件不存在，将重新预训练模型')
            self.master_App.log_frame.add_log('模型文件不存在，将重新预训练模型', 'info')
            return None

    # 开始训练
    def test(self, test_image_name, train_type=None, txt_file_path=None, img_dir_path=None, load_model_path=None, save_model_path='./models'):
        load_model = None
        self.history = None
        # 加载模型
        if load_model_path is None:
            print('没有模型文件，开始训练')
            self.master_App.log_frame.add_log('没有模型文件，开始训练', 'info')
        else:
            print('正在加载模型文件:', os.path.abspath(load_model_path))
            self.master_App.log_frame.add_log('正在加载模型文件:{}'.format(os.path.abspath(load_model_path)), 'info')
            load_model = self.load_model(load_model_path)
            self.model = load_model if load_model is not None else self.model
        # 模型加载不成功（错误/路径为空），开始训练
        if load_model is None:
            # 单字符训练
            if train_type == 'single_number':
                self.history = self.fit_model(train_type=train_type, txt_file_path=txt_file_path)
                if self.master_App.train_is_stop:
                    print('训练中止')
                    self.master_App.log_frame.add_log('训练中止', 'info')
                    return
                print('模型训练完成')
                self.master_App.log_frame.add_log('模型训练完成', 'info')
                # 保存模型
                self.save_model(save_model_path)
                # 可视化学习效果
                self.train_result_view(self.history)
            # 车牌训练
            elif train_type == 'car_number':
                self.history = self.fit_model(train_type=train_type, img_dir_path=img_dir_path)
                self.save_model(save_model_path)
                # 可视化学习效果
                self.train_result_view(self.history)
        else:
            # 单字符训练
            if train_type == 'single_number':
                print("单字符已预训练，无需再次训练")
                self.master_App.log_frame.add_log('单字符已预训练，无需再次训练', 'info')
            elif train_type == 'car_number':
                self.history = self.fit_model(train_type=train_type, img_dir_path=img_dir_path)
                self.save_model(save_model_path)
                # 可视化学习效果
                self.train_result_view(self.history)
        # 测试显示
        new_img = cv2.imread(test_image_name)
        # 判断是否是预训练模式
        if self.master_App.on_pre_train:
            if train_type == 'single_number':
                # new_img = cv2.imread(test_image_name, cv2.IMREAD_GRAYSCALE)
                result = self.predict(new_img,self.model)
                print('预测结果',self.label_dict[str(result[0])])
                self.master_App.log_frame.add_log('预测结果:{}'.format(self.label_dict[str(result[0])]), 'info')
            elif train_type == 'car_number':
                img_list = []
                img_list.append(new_img[10:90, 10:50])
                img_list.append(new_img[10:90, 55:95])
                img_list.append(new_img[10:90, 125:165])
                img_list.append(new_img[10:90, 170:210])
                img_list.append(new_img[10:90, 215:255])
                img_list.append(new_img[10:90, 260:300])
                img_list.append(new_img[10:90, 305:345])
                img_list.append(new_img[10:90, 350:390])
                plt.figure(figsize=(12, 6))
                for i in range(8):
                    plt.subplot(2, 4, i+1)
                    plt.imshow(img_list[i], cmap='gray')
                plt.show()
                for img in img_list:
                    result = self.predict(img,self.model)
                    print('预测结果',self.label_dict[str(result[0])])
                    self.master_App.log_frame.add_log('预测结果:{}'.format(self.label_dict[str(result[0])]), 'info')
        else:
            result = self.predict(new_img,self.model)
            car_number = self.provinces[result[0][0]] + \
                    self.alphabets[result[0][1]] + \
                    ''.join([self.ads[i] for i in result[0][2:]])
            print('预测结果',car_number)
            self.master_App.log_frame.add_log('预测结果:{}'.format(car_number), 'info')
    
    # 预测接口
    def shibie_test(self, img_path, load_model_path=None):
        load_model = None
        self.history = None
        if self.shibie_model is None:
            # 加载模型
            if load_model_path is None:
                print('没有模型文件，无法识别')
                raise ValueError('没有模型文件，无法识别')
            else:
                print('正在加载模型文件:', os.path.abspath(load_model_path))
                self.master_App.log_frame.add_log('正在加载模型文件:{}'.format(os.path.abspath(load_model_path)), 'info')
                load_model = self.load_model(load_model_path)
                self.shibie_model = load_model if load_model is not None else self.model
            # 模型加载不成功（错误/路径为空），无法识别
            if load_model is None:
                print('模型加载失败，无法识别')
                self.master_App.log_frame.add_log('模型加载失败，无法识别', 'info')
                raise ValueError('模型加载失败，无法识别')
        # 测试显示
        new_img = cv2.imread(img_path)
        # 判断new_img的大小
        if new_img.shape[0] == 20 and new_img.shape[1] == 20:
            result = self.predict(new_img,self.shibie_model)
            print('单字符预测结果',self.label_dict[str(result[0])])
            self.master_App.log_frame.add_log('单字符预测结果:{}'.format(self.label_dict[str(result[0])]), 'info')
            return self.label_dict[str(result[0])]
        # 判断是否是预训练模式
        if self.master_App.on_pre_train:
            # new_img = 图像处理.image_preprocess(new_img, 'shibie_image')
            img_list = []
            img_list.append(new_img[10:90, 10:50])
            img_list.append(new_img[10:90, 55:95])
            img_list.append(new_img[10:90, 125:165])
            img_list.append(new_img[10:90, 170:210])
            img_list.append(new_img[10:90, 215:255])
            img_list.append(new_img[10:90, 260:300])
            img_list.append(new_img[10:90, 305:345])
            img_list.append(new_img[10:90, 350:390])
            plt.figure(figsize=(12, 6))
            for i in range(8):
                plt.subplot(2, 4, i+1)
                plt.imshow(img_list[i], cmap='gray')
            plt.show()
            car_number = ''
            for img in img_list:
                result = self.predict(img,self.shibie_model)
                print('预测结果',self.label_dict[str(result[0])])
                self.master_App.log_frame.add_log('预测结果:{}'.format(self.label_dict[str(result[0])]), 'info')
                car_number += self.label_dict[str(result[0])]
            return car_number
        else:
            # new_img = 图像处理.image_preprocess(new_img, 'shibie_image')
            result = self.predict(new_img,self.shibie_model)
            car_number = self.provinces[result[0][0]] + \
                    self.alphabets[result[0][1]] + \
                    ''.join([self.ads[i] for i in result[0][2:]])
            print('预测结果',car_number)
            self.master_App.log_frame.add_log('预测结果:{}'.format(car_number), 'info')
            return car_number
        

# # 实例化模型训练类
# model_train = Model_train()
# # 开始训练
# model_train.test('D://My_Code_Project//Image_Dateset//single_number//VehicleLicense//Data//xin//xin_0001.jpg', train_type='single_number', txt_file_path='D://My_Code_Project//Image_Dateset//single_number//VehicleLicense//trainval.txt')