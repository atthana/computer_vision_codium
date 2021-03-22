# https://github.com/ageitgey/face_recognition/wiki/How-do-I-save-face-encodings-to-a-file%3F

import face_recognition
import pickle

all_face_encodings = {}

img1 = face_recognition.load_image_file("obama.jpeg")
all_face_encodings["obama"] = face_recognition.face_encodings(img1)[0]

img2 = face_recognition.load_image_file("biden.jpeg")
all_face_encodings["biden"] = face_recognition.face_encodings(img2)[0]

# ... etc ...

with open('dataset_faces.dat', 'wb') as f:
    pickle.dump(all_face_encodings, f)


# หลังจากโหลดเสร็จจะได้ dataset_faces.dat ขึ้นมานะ
