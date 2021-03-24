# https://github.com/ageitgey/face_recognition/wiki/How-do-I-save-face-encodings-to-a-file%3F

import pickle

import face_recognition

all_face_encodings = {}

print('Start doing dataset')
img1 = face_recognition.load_image_file("obama.jpeg")
all_face_encodings["obama"] = face_recognition.face_encodings(img1)[0]

img2 = face_recognition.load_image_file("biden.jpeg")
all_face_encodings["biden"] = face_recognition.face_encodings(img2)[0]

with open('dataset_faces.dat', 'wb') as f:
    pickle.dump(all_face_encodings, f)

print('Finished creating dataset')

# หลังจากโหลดเสร็จจะได้ dataset_faces.dat ขึ้นมานะ
