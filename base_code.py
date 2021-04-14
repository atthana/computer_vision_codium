import pickle
from datetime import datetime

import cv2
import face_recognition
import numpy as np

# Next steps:
# 1. จะให้มัน logs เข้า Google sheets อย่างไร
# 2. เอาแค่ check-in เท่านั้นก่อน
# 3. สร้าง function มาสำหรับถ่าย VDO แล้วก้อ capture หน้าออกมาเยอะๆเลย
# 4. แต่หลังจากได้หน้าออกมาเยอะๆแล้ว จะต้องสร้าง function มา rename ชื่อให้มันและตามด้วยตัวเลขเรียงกันไปเยอะๆด้วย
# 5. ถามกุ๊กกิ๊ก ตรงไหนใน code ที่เป้นการ count frame แต่ละครั้งที่มันระบุชื่อถูก

process_this_frame = True

# ======================= Below is process to compare faces from camera ==================================
with open('dataset_codium_encoding.dat', 'rb') as f:
    known_face_encodings = pickle.load(f)

with open('dataset_codium_name.dat', 'rb') as f:
    known_face_names = pickle.load(f)

video_capture = cv2.VideoCapture(0)

count_correct_face = 0
count_name = {}


def get_date_time_now():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string


if __name__ == '__main__':
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

                if min(face_distances) < 0.45:  # if distance value is low that mean => very match
                    name = known_face_names[best_match_index]
                    face_names.append(name)  # ชื่อที่ไปโชว์ที่หน้า ตรงหน้าจอ
                    if name in count_name:  # เป็นการเช็ค name กับ key ใน dict นะ (เช็ค value แบบนี้ไม่ได้)
                        count_correct_face += 1  # ผมต้องนับเพื่อที่จะนับเฟรมที่ถูกต้อง
                        count_name[name] = count_name[name] + 1  # เวลาที่มีการบวก มันคือบวกไปที่ value นะ
                        print('----- count ----> ', count_correct_face)
                        if count_name[name] == 5:
                            print('This is = {}, Time check-in = {}'.format(name, get_date_time_now()))
                    else:
                        count_name[name] = 1  # เป็นการทำให้ key ที่ได้รับเข้ามา มีค่าเป็น 1
                else:
                    face_names.append('Unknown')
                    count_correct_face = 0
                    print('------ unknown ------')
                print('xxxxxxxxxxxx')
                print(count_name)

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
