# -*-coding = utf-8 -*-

import os
import cv2

if __name__ == "__main__":
    path_read = "D:\\work\\data\\"
    path_write = "D:\\work\\data1\\"
    target_size = [224, 224]

    for i in range(1,19):
        p_r = path_read + str(i) + "\\"
        p_w = path_write + str(i) + "\\"

        image_list = [x for x in os.listdir(p_r)]
        for num, img in enumerate(image_list):
            print(num, img)
            image = cv2.imread(p_r+img, cv2.IMREAD_COLOR)
            # print(path_read+"/"+img)
            new_image = cv2.resize(image, (target_size[0], target_size[1]), interpolation=cv2.INTER_CUBIC)
            print(1)
            image_dir = p_w+str(num).zfill(4)+'.png'
            print(2)
            cv2.imwrite(image_dir, new_image)
            print(3)