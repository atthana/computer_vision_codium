# https://medium.com/@basit.javed.awan/resizing-multiple-images-and-saving-them-using-opencv-518f385c28d3


import glob
import os

import cv2
import imutils


def resize_keep_aspect_ratio():
    input_folder = 'codium_raw_photos'
    output_folder = 'resize_raw_photos'

    try:
        os.mkdir(output_folder)
    except OSError as err:
        print('---------- err -----------')
        print(err)

    for img in glob.glob(input_folder + '/*.JPG'):
        print('---- img ---')
        filename = img.split('/')[-1].split('.')[0]
        print(img)
        print(filename)
        image = cv2.imread(img)
        img_resized = imutils.resize(image, width=600)  # resize without distortion
        cv2.imwrite('resize_raw_photos/{}.jpg'.format(filename), img_resized)
        cv2.imshow('image', img_resized)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


resize_keep_aspect_ratio()
