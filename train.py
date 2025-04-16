from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, Xception, ResNet50V2, MobileNetV2, InceptionV3
import tensorflow as tf
import numpy as np
np.object = object
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import cv2
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import utils
import cv2
from tensorflow.keras import backend as K

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


# K.set_image_dim_ordering('tf')
def generate_arrays_from_file(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # print(name)
            # 从文件中读取图像
            img = cv2.imread("/media/lyz/8TDisk/LYZ_Project/YJY/YYQ_Breast/code/VGG16/data/image" + '/' + name)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # if img.shape[2] == 3:
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # else:
            #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = img/255.0
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i+1) % n
        # 处理图像
        X_train = utils.resize_image(X_train,(224,224))
        X_train = X_train.reshape(-1,224,224,3)
        Y_train = to_categorical(np.array(Y_train),num_classes = 2)
        yield (X_train, Y_train)


if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # 模型保存的位置
    log_dir = "./logs/"

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = tf.Session(config=config)

    # 打开数据集的txt
    with open(r"/media/lyz/8TDisk/LYZ_Project/YJY/YYQ_Breast/code/VGG16/data/train.txt", "r", encoding='UTF-8') as f:
        lines = f.readlines()


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


# K.set_image_dim_ordering('tf')
def generate_arrays_from_file(lines, batch_size, img_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # get batch size
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # print(name)
            # find image
            img = cv2.imread("/media/lyz/8TDisk/LYZ_Project/YJY/YYQ_Breast/code/VGG16/data/second_image" + '/' + name)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # if img.shape[2] == 3:
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # else:
            #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = img/255.0
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i+1) % n
        # 处理图像
        X_train = utils.resize_image(X_train,(img_size,img_size))
        X_train = X_train.reshape(-1,img_size,img_size,3)
        Y_train = to_categorical(np.array(Y_train),num_classes = 2)
        yield (X_train, Y_train)


if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # save model weight
    log_dir = "./logs/"

    # select model cls
    backbone = "/InceptionV3/"
    img_size = 299
    # 一次的训练集大小
    batch_size = 128
    k = 2

    # open dataset with txt file
    with open(r"/media/lyz/8TDisk/LYZ_Project/YJY/YYQ_Breast/code/VGG16/data/train.txt", "r", encoding='UTF-8') as f:
        lines = f.readlines()

    # random dataset
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # replace other pre-trained model
    model_base = InceptionV3(weights="imagenet", include_top=False)
    model_base.trainable = False

    model = Sequential()
    model.add(model_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))

    num_val = int(len(lines) / 3)  # 验证集的数量index
    num_train = len(lines) - num_val  # 训练集的数量index

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    k_1_train_list = lines[:num_train]
    k_1_test_list = lines[num_train:]

    k_2_train_list = lines[num_val:]
    k_2_test_list = lines[:num_val]

    k_3_train_list = lines[:num_val] + lines[num_train:]
    k_3_test_list = lines[num_val:num_train]

    k_trian_lines = [k_1_train_list, k_2_train_list, k_3_train_list]
    k_test_lines = [k_1_test_list, k_2_test_list, k_3_test_list]

    # 保存的方式，3世代保存一次
    checkpoint_period1 = ModelCheckpoint(
                                    log_dir + backbone + '{}/'.format(k+1) + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='accuracy',
                                    save_weights_only=False,
                                    save_best_only=True,
                                    period=3
                                )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='val_accuracy',
                            factor=0.5,
                            patience=3,
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss',
                            min_delta=0,
                            patience=10,
                            verbose=1
                        )

    # 交叉熵
    model.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(lr=1e-3),
            metrics = ['accuracy'])

    # 80%用于训练，20%用于估计。
    num_val = int(len(lines)/3) # 验证集的数量index
    num_train = len(lines) - num_val # 训练集的数量index

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    k_1_train_list = lines[:num_train]
    k_1_test_list  = lines[num_train:]

    k_2_train_list = lines[num_val:]
    k_2_test_list = lines[:num_val]

    k_3_train_list = lines[:num_val] + lines[num_train:]
    k_3_test_list = lines[num_val:num_train]

    k_trian_lines = [k_1_train_list, k_2_train_list, k_3_train_list]
    k_test_lines = [k_1_test_list, k_2_test_list, k_3_test_list]

    # 开始训练
    model.fit_generator(generate_arrays_from_file(k_trian_lines[k], batch_size, img_size),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=generate_arrays_from_file(k_test_lines[k], batch_size, img_size),
                validation_steps=max(1, num_val//batch_size),
                epochs=100,
                initial_epoch=0,
                callbacks=[checkpoint_period1, reduce_lr])

        # model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
        #         steps_per_epoch=max(1, num_train//batch_size),
        #         validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
        #         validation_steps=max(1, num_val//batch_size),
        #         epochs=100,
        #         initial_epoch=0,
        #         callbacks=[checkpoint_period1, reduce_lr])

    # model.save_weights(log_dir+'last1.h5')
    model.save(log_dir+'K_{}_last111.h5'.format(k+1))
    print('K_{}_finish!!--'.format(k+1))
