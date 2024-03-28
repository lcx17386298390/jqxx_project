# 模型训练类
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 图像尺寸
img_height, img_width = 20, 20

# 类别数量（假设我们有10个数字和26个英文字母）
num_classes = 36

def load_data(img_dir):
    images = []
    labels = []
    for filename in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (img_width, img_height))
            images.append(img)
            labels.append(int(filename.split('_')[0]))  # 假设文件名的格式为"label_index.jpg"
    return images, labels

images, labels = load_data('/path/to/your/image/dir')
images = np.array(images, dtype='float32')
labels = np.array(labels)

# 归一化
images /= 255

# 将标签转换为one-hot向量
labels = to_categorical(labels, num_classes)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_height, img_width, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

model = create_model()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(x_test, y_test))

def predict(model, img):
    img = cv2.resize(img, (img_width, img_height))
    img = np.reshape(img, (1, img_height, img_width, 1))
    img = img.astype('float32')
    img /= 255
    return np.argmax(model.predict(img), axis=-1)

# 假设我们有一个新的字符图像
new_img = cv2.imread('/path/to/your/new/image.jpg', cv2.IMREAD_GRAYSCALE)
print(predict(model, new_img))
