# 模型训练类
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

# 定义模型训练类
class Model_train:
    def __init__(self):
        # 车牌所有的字符类别
        self.num_classes = 65
        # 图片的高和宽
        self.img_height, self.img_width = 20, 20
        # 创建模型
        self.model = self.create_model()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    # 训练模型(默认训练单字符识别模型, 传入参数为'single_number'则训练单字符识别模型，否则训练多字符识别模型)
    def fit_model(self, train_type='single_number', txt_file_path=None, img_dir_path=None):
        self.images, self.labels = [], []
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
        self.images = np.array(self.images, dtype='float32')
        self.labels = np.array(self.labels)
        self.images /= 255
        self.labels = to_categorical(self.labels, self.num_classes) # 将类别标签转换为one-hot编码
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.images, self.labels, test_size=0.2, random_state=42)
        history = self.model.fit(self.x_train, self.y_train, batch_size=128, epochs=10, verbose=1, validation_data=(self.x_test, self.y_test))
        return history
    

    # 加载车牌数据
    def load_data(self):
        images = []
        labels = []
        for filename in os.listdir(self.img_dir):
            img = cv2.imread(os.path.join(self.img_dir, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(int(filename.split('_')[0]))  # 假设文件名的格式为"label_index.jpg"
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
    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model

    # 预测函数
    def predict(self, img):
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = np.reshape(img, (1, self.img_height, self.img_width, 1))
        img = img.astype('float32')
        img /= 255
        result = self.model.predict(img)
        # 输出预测结果
        result = np.argmax(result)
        return result
    
    # 保存模型
    def save_model(self, model_path, train_type='single_number'):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # 获取目录下的所有子目录
        dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        # 计算子目录的数量
        num_dirs = len(dirs)+1
        self.save_train_file_floder = None
        if train_type == 'single_number':
            self.save_train_file_floder = os.path.join(model_path, 'train'+str('_'+str(num_dirs)+'/single_number'))
        elif train_type == 'car_number':
            self.save_train_file_floder = os.path.join(model_path, 'train'+str('_'+str(num_dirs)+'/car_number'))
        if not os.path.exists(self.save_train_file_floder):
            os.makedirs(self.save_train_file_floder)
        self.save_file_path = os.path.join(self.save_train_file_floder, 'model.h5')
        self.model.save(self.save_file_path)
    
    # 学习效果查看
    def train_result_view(self,history):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
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
            self.model.load_weights(model_path)
            print('模型加载成功')
            return self.model
        print('模型加载失败，重新训练')
        return None

    # 开始训练
    def test(self, image_name, train_type='single_number', txt_file_path=None, img_dir_path=None):
        # 加载模型
        load_model = self.load_model('./models/train_1/single_number/model.h5')
        self.model = load_model if load_model is not None else self.model
        self.history = None
        if load_model is None:
            # 没有模型文件，开始训练
            if train_type == 'single_number':
                self.history = self.fit_model(train_type=train_type, txt_file_path=txt_file_path)
            elif train_type == 'car_number':
                self.history = self.fit_model(train_type=train_type, img_dir_path=img_dir_path)
        self.save_model('./models')
        # 测试显示
        new_img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        self.train_result_view(self.history)
        print('预测结果',self.predict(new_img))

# 实例化模型训练类
model_train = Model_train()
# 开始训练
model_train.test('D://My_Code_Project//Image_Dateset//single_number//VehicleLicense//Data//xin//xin_0001.jpg', train_type='single_number', txt_file_path='D://My_Code_Project//Image_Dateset//single_number//VehicleLicense//trainval.txt')