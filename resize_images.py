# https://medium.com/@basit.javed.awan/resizing-multiple-images-and-saving-them-using-opencv-518f385c28d3


import cv2
import glob
import os

inputFolder = 'Cars'
os.mkdir('Resized')
i = 0

for img in glob.glob(inputFolder + '/*.jpg'):
    image = cv2.imread(img)
    imgResized = cv2.resize(image, (150, 150))
    cv2.imwrite('Resized/image%041.jpg' %i, imgResized)
    i += 1
    cv2.imshow('image', imgResized)
    cv2.waitKey(30)
cv2.destroyAllWindows()
