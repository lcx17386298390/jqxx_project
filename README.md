# 未做操作
1训练完后应该还有保存训练模型的操作，然后可以手动选择历史保存下来的训练模型

# 项目做法
1、使用yolo模型，用什么深度学习的模型就换，比如LPRNet，这个听说挺好，有兴趣再说
2、自己写训练过程，逻辑上要难一点，但是代码总归有点简陋
    框架：keras，TensorFlow，PyTorch，Caffe，MXNet等
        keras框架比较简单，可以不去调用硬件设备运算，但是比较慢
    模型可以在各个框架中选择
3、模型应该有一个预训练，将每个字母先训练(可以用模型训练了保存下来，每次选择模型的时候可以加载预模型，在进行二次训练)

# 注意
1、单字符训练几个txt
    train：训练
    val：验证
    test：做测试用
    trainval是训练和验证的合集（就是这里设计的真是有问题，tmd耽误我半天，看不懂）
2、现在训练模型已经做完，保存模型的逻辑还没做
3、模型目录
    --models
        --train_1
            --single_number
                --model.h5
            --car_number
                --model.h5
        --train_2

### 2024-3-30
#### 复盘
1·现在已经做好gui两个训练接口
2·训练模型类已经各功能模块完成，用test方法封装，可以直接用test方法来连接口，但目前模型加载文件路径怎么选择有点麻烦
3·图像提取类：给名字即可（需要数据集规范命名）
4·现在接训练接口和gui

### 2024-4-1
修改了预训练bug，同时更新了进度条（进度条还有一点小毛病，不太准，还有上面的标签更改不规范）

# 模型
1、循环神经网络（RNN）
2、长短期记忆网络（LSTM）
3、门控循环单元（GRU）
4、Transformer
5、 多层感知器（MLP）：多层感知器是一种全连接的神经网络，适合处理规模较小、相对简单的图像。由于它没有卷积层和池化层，因此对于大规模的复杂图像，MLP 可能不如 CNN 效果好。

卷积递归神经网络（Convolutional Recurrent Neural Network，CRNN）：CRNN 结合了 CNN 和 RNN，能够有效处理含有序列信息的图像，如文字行识别。

空间变换网络（Spatial Transformer Network，STN）：STN 是一种可以学习进行空间变换的网络，例如旋转、缩放和平移，对于处理变形和扭曲的字符图像很有用。

胶囊网络（Capsule Network）：胶囊网络是一种试图使用向量而非标量来表示特征信息的网络，它能够捕捉图像中的空间层次关系，对于字符识别任务可能有帮助。

深度强化学习（Deep Reinforcement Learning，DRL）：对于一些需要决策过程的字符识别任务，如阅读理解和手写识别，可以考虑使用深度强化学习。

对于复杂的图像识别任务，目前 CNN 仍是最常用和效果最好

### CNN结构
LeNet-5: 这是Yann LeCun在1998年提出的，是最早的卷积神经网络之一。其结构简单，适合于入门学习，同时在手写数字识别任务上表现良好。

AlexNet: 由Alex Krizhevsky在2012年提出，是深度学习领域的里程碑之一。它在ImageNet大规模视觉识别挑战（ILSVRC）上取得了突破性的成果。其结构相比LeNet更深，使用了ReLU激活函数和Dropout等技术。

VGGNet (VGG-16/VGG-19): VGGNet由牛津大学的视觉几何组（Visual Geometry Group）在2014年提出。VGGNet的特点是其结构非常规整，全部使用了3x3的卷积核和2x2的池化核，验证了深度（卷积层数）对网络性能的影响。

GoogLeNet (Inception v1): GoogLeNet在2014年的ILSVRC比赛中取得了冠军。其引入了Inception模块，通过不同大小的卷积核并行处理信息，然后再将结果拼接起来，有效地增加了网络的宽度。

ResNet (Residual Network): ResNet由微软研究院在2015年提出，通过引入残差连接（Residual Connection）解决了深度神经网络难以训练的问题，使得神经网络的层数可以达到之前无法想象的深度（如152层），在各种任务上都取得了非常好的结果。

以上五种网络结构在一些基准测试中都取得了很好的结果。然而，需要注意的是，这些模型在复杂的图像数据集（如ImageNet）上表现良好，并不意味着它们在简单任务（如MNIST手写数字识别）上都能取得最好的结果。对于简单任务，复杂的模型可能会导致过拟合。因此，选择哪个模型取决于你的具体任务和数据。在实际使用中，你可能需要调整这些模型的结构以适应你的任务，例如改变卷积层、全连接层或者dropout层的数量，或者调整学习率等超参数。

### 2024-4-1
加上了模型选择，设置界面设置好，接口已连接好
明天首要任务：选择结构后加载训练好的模型

### 2024-4-2
--models
    --CNN
        --train1
        --train2
    --LeNet-5
        --train1
            --model.h5
            --image
        --train2
修改了更为合理的结构

加载已训练模型差不多，接口没有接
测试api思路：
    使用训练接口，加载设置面选择的模型


YOLO (You Only Look Once)：YOLO是一个非常快速的目标检测算法，它可以在一次前向传播中检测出图像中的所有目标。YOLO有多个版本，包括YOLOv1, YOLOv2 (YOLO9000), YOLOv3, YOLOv4, 和YOLOv5。

Faster R-CNN：Faster R-CNN是一种精度很高的目标检测算法，它使用了区域提议网络 (RPN) 来提取目标候选区，然后使用RoI池化和全连接层来分类和回归边界框。

SSD (Single Shot MultiBox Detector)：SSD是一种在速度和精度之间取得平衡的目标检测算法，它在多个尺度上进行预测，使得它能够检测出不同大小的目标。

RetinaNet：RetinaNet使用了一种名为Focal Loss的新颖损失函数，可以解决目标检测中的类别不平衡问题。这使得它在检测小目标（如远处的车牌）时表现得很好。              

### 2024-4-4
直接使用图片原图车牌部分进行训练，然后进行字符切割
个别图片数据集命名有问题，考虑如何检测并排出