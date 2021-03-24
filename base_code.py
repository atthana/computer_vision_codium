import pickle

import cv2
import face_recognition
import numpy as np
import known_faces as faces

# https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
# 1. Need to load all images and resize photos and make a new set in new folder.
# 2. Fixed red error during running and found.
# 3. Refactor code to collect all name and location in the same place like teacher's code.
# Use this code to be code base because it's faster than teacher's code.


# Next steps:
# 1. จะให้มัน logs เข้า Google sheets อย่างไร
# 2. จะเช็คยังไงว่า อันนี้คือ check-in อันนีี้คือ check-out
# 3. สร้าง function มาสำหรับถ่าย VDO แล้วก้อ capture หน้าออกมาเยอะๆเลย
# 4. แต่หลังจากได้หน้าออกมาเยอะๆแล้ว จะต้องสร้าง function มา rename ชื่อให้มันและตามด้วยตัวเลขเรียงกันไปเยอะๆด้วย

face_locations = []
face_encodings = []
face_names = []

process_this_frame = True
# ===================== Process to learn to make dataset ===================================


known_face_names = []
known_face_encodings = []

for face in faces.known_faces:
    try:
        print(face)
        known_face_names.append(face[0])
        face_image = face_recognition.load_image_file(face[1])
        face_encoding = face_recognition.face_encodings(face_image)[0]
        known_face_encodings.append(face_encoding)

        # with open('dataset_codium.dat', 'wb') as f:
        #     pickle.dump(known_face_encodings, f)
        #
        # print('Finished creating dataset')
    except IndexError as err:
        print('--- Exception ---')
        print(err)
        pass

# ======================= Below is process to compare faces from camera ==================================

video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]  # ทำอะไรอ่ะ ต้องถามกุ๊กกิ๊ก

    if process_this_frame:

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if min(face_distances) < 0.45:  # if distance is low that mean => very match
                name = known_face_names[best_match_index]
                face_names.append(name)
            else:
                face_names.append('Unknown')

            # if matches[best_match_index]:  # The example checking match from github face_recognition.
            #     name = known_face_names[best_match_index]
            # face_names.append(name)

    process_this_frame = not process_this_frame  # คืออะไรค้าบ

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
