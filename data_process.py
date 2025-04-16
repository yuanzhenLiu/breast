import os
from sklearn.model_selection import train_test_split


class dataProcess(object):
    @staticmethod
    def split_test_set(org_img_bas_pth, save_img_bas_pth):
        patient_id_list =[]
        for cls_name in os.listdir(org_img_bas_pth):
            cls_pth = os.path.join(org_img_bas_pth, cls_name)
            for img_name in os.listdir(cls_pth):
                img_name_1 = img_name.split('.')
                img_name_2 = img_name_1[0].split('_')
                patient_id = img_name_2[0]
                patient_id_list.append(patient_id)
            print(len(patient_id_list))
            print(len(set(patient_id_list)))
            patient_id_list = []

    @staticmethod
    def data_check(org_data_bas_pth):
        patient_id_list = []
        for img_name in os.listdir(org_data_bas_pth):
            img_pth = os.path.join(org_data_bas_pth, img_name)
            img_name_1 = img_name.split('.')[0]
            res_img_name = img_name_1.split('_')[0]
            patient_id_list.append(res_img_name)

        patient_id_set = set(patient_id_list)
        for i in patient_id_set:
            print(i)



if __name__ == "__main__":
    # org_img_bas_pth = r"/media/lyz/8TDisk/LYZ_Project/YJY/YYQ_Breast/code/VGG16/data/second_image"
    # dataProcess.split_test_set(org_img_bas_pth, None)

    # org_data_bas_pth = r"/media/lyz/8TDisk/LYZ_Project/YJY/YYQ_Breast/data/second_data/org_data/fib/PTs"
    # dataProcess.data_check(org_data_bas_pth)
    test_list = [x for x in range(100)]
    print(test_list[:10] + test_list[90:])
