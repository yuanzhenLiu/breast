import os
from PIL import Image, ImageEnhance
import numpy as np
import random
import math
from tqdm import tqdm


def resize_img(img, target_size):
    img = img.resize((target_size, target_size), Image.BILINEAR)

    return img


def center_crop_img(img, target_size):
    w, h = img.size
    tw, th = target_size, target_size
    assert (w >= target_size) and (h >= target_size), \
            "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    # crop()四个参数分别是：(左上角点的x坐标，左上角点的y坐标，右下角点的x坐标，右下角点的y坐标)
    img = img.crop((x1, y1, x1 + tw, y1 + th))

    return img


def random_crop_img(img, target_size, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.size[0]) / img.size[1]) / (w**2),
                (float(img.size[1]) / img.size[0]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.size[0] * img.size[1] * np.random.uniform(scale_min,
                                                                scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = np.random.randint(0, img.size[0] - w + 1)
    j = np.random.randint(0, img.size[1] - h + 1)

    img = img.crop((i, j, i + w, j + h))
    img = img.resize((int(target_size), int(target_size)), Image.BILINEAR)

    return img


def rotate_image(img):
    # 将图片随机旋转-14到15之间的某一个角度
    angle = np.random.randint(-14, 15)
    img = img.rotate(angle)

    return img


def flip_image(img):
    # 将图片随机左右翻转， 根据需要也可以设置随机上下翻转
    v = random.random()
    if v < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img


def bright_image(img):
    # 随机调整亮度（调亮或暗）
    v = random.random()
    if v < 0.5:
        brightness_delta = 0.225
        delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
        # delta值为0表示黑色图片，值为1表示原始图片
        img = ImageEnhance.Brightness(img).enhance(delta)

    return img


def contrast_image(img):
    # 随机调整对比度
    v = random.random()
    if v < 0.5:
        contrast_delta = 0.5
        delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
        # delta值为0表示灰度图片，值为1表示原始图片
        img = ImageEnhance.Contrast(img).enhance(delta)

    return img


def saturation_image(img):
    # 随机调整颜色饱和度
    v = random.random()
    if v < 0.5:
        saturation_delta = 0.5
        delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
        # delta值为0表示黑白图片，值为1表示原始图片
        img = ImageEnhance.Color(img).enhance(delta)

    return img


def hue_image(img):
    # 随机调整颜色色度
    v = random.random()
    if v < 0.5:
        hue_delta = 18
        delta = np.random.uniform(-hue_delta, hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')

    return img


def distort_image(img):
    # 随机数据增强
    v = random.random()
    # 顺序可以自己随意调整
    if v < 0.35:
        img = bright_image(img)
        img = contrast_image(img)
        img = saturation_image(img)
        img = hue_image(img)
    elif v < 0.7:
        img = bright_image(img)
        img = saturation_image(img)
        img = hue_image(img)
        img = contrast_image(img)

    return img


if __name__ == "__main__":
    org_img_base_path = r"/media/lyz/8TDisk/LYZ_Project/YJY/YYQ_Breast/code/VGG16/data/second_image/PTs"
    for img_name in tqdm(os.listdir(org_img_base_path)):
        img_first_name = img_name.split('.')[0]
        img_path = os.path.join(org_img_base_path, img_name)
        org_img = Image.open(img_path)

        img1 = distort_image(org_img)
        img1_save_name = img_first_name + "_distort." + img_name.split('.')[-1]
        img1_save_path = os.path.join(org_img_base_path, img1_save_name)
        img1.save(img1_save_path)

        img2 = flip_image(org_img)
        img2_save_name = img_first_name + "_flip." + img_name.split('.')[-1]
        img2_save_path = os.path.join(org_img_base_path, img2_save_name)
        img2.save(img2_save_path)

        img3 = rotate_image(org_img)
        img3_save_name = img_first_name + "_rotate." + img_name.split('.')[-1]
        img3_save_path = os.path.join(org_img_base_path, img3_save_name)
        img3.save(img3_save_path)

        img4 = bright_image(org_img)
        img4_save_name = img_first_name + "_bright." + img_name.split('.')[-1]
        img4_save_path = os.path.join(org_img_base_path, img4_save_name)
        img4.save(img4_save_path)

        img5 = hue_image(org_img)
        img5_save_name = img_first_name + "_hue." + img_name.split('.')[-1]
        img5_save_path = os.path.join(org_img_base_path, img5_save_name)
        img5.save(img5_save_path)

        # img6 = contrast_image(org_img)
        # img6_save_name = img_first_name + "_contrast." + img_name.split('.')[-1]
        # img6_save_path = os.path.join(org_img_base_path, img6_save_name)
        # img6.save(img6_save_path)
        #
        # img7 = saturation_image(org_img)
        # img7_save_name = img_first_name + "_saturation." + img_name.split('.')[-1]
        # img7_save_path = os.path.join(org_img_base_path, img7_save_name)
        # img7.save(img7_save_path)






