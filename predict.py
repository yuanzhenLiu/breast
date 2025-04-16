import numpy as np
import utils
import cv2
from keras import backend as K
from model.VGG16 import VGG16
import os

K.set_image_dim_ordering('tf')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    model = VGG16(6)
    model.load_weights("C://Users//Administrator//Desktop//keras_1//logs//bestweight.h5")
    img = cv2.imread(r"C://Users//Administrator//Desktop//zhichangai//colon_cancer//c_0000.jpg")
    # cv2.imshow("img", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255
    img = np.expand_dims(img, axis=0)
    img = utils.resize_image(img, (224, 224))
    #utils.print_answer(np.argmax(model.predict(img)))
    #print(utils.print_answer(np.argmax(model.predict(img))))
    print(model.predict(img))

    res_label = ["cancer", "norm", "ployph"]
    print(res_label[model.predict(img).argmax()])


import cv2
from keras import backend as K
from model.VGG16 import VGG16
import os

K.set_image_dim_ordering('tf')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    model = VGG16(6)
    model.load_weights("C://Users//Administrator//Desktop//keras_1//logs//bestweight.h5")
    img = cv2.imread(r"C://Users//Administrator//Desktop//zhichangai//colon_cancer//c_0000.jpg")
    # cv2.imshow("img", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255
    img = np.expand_dims(img, axis=0)
    img = utils.resize_image(img, (224, 224))
    #utils.print_answer(np.argmax(model.predict(img)))
    #print(utils.print_answer(np.argmax(model.predict(img))))
    print(model.predict(img))

    res_label = ["cancer", "norm", "ployph"]
    print(res_label[model.predict(img).argmax()])
