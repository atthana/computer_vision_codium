# Slide 15 คราวนี้จะให้ detect และระบุชื่อด้วย VDO บ้างนะ ผมทำออกแล้วนะ
# คือเราต้องมาตั้งว่า ถ้า distance เท่าไหร่ก้อให้ปรับเป็น unknown ไปเลย มันจะได้ไม่ตีกันไปชนหน้าอื่น
# 1. ต้องทำต่อคือ ให้เช็คถ้า distance < 0.5 ให้เป็นชื่อตาม dataset ตั้งต้น ถ้าไม่ก้อให้เป็น Unknown เข้าไปเพราะมันจะไม่ค่อยตรงแล้ว
# 2. แล้วก้อเราจะเพิ่มรูปเข้าไปได้ยังไง จะได้ทั้งเงยได้ ก้มได้ด้วย แบบที่เป็น product ขายอยู่จริงๆเลย
# จิงๆ เราทำ dataset ตอนที่เราให้มันเซฟก่อนหน้าได้นะ สัก 50 รูป
# 3. ลองเอาไปรันใน Pycharm ดีกว่านะ เพราะทำที่นี่เหมือนมันช้า

import cv2
import face_recognition
import numpy as np

known_faces = [

    # ('Beer', 'image/face/beer.jpg'),
    # ('Pang', 'image/face/pang.jpg'),
    # ('Awesome', 'image/face/Awesome.jpg'),
    # ('Q', 'image/face/q.jpg')

    ('Q', 'codium_raw_photos/q1.jpg'),
    # ('Q', 'codium_raw_photos/q2.JPG'),
    ('Q', 'codium_raw_photos/q3.JPG'),
    # ('Q', 'codium_raw_photos/q4.JPG'),
    # ('Q', 'codium_raw_photos/q5.JPG'),
    #
    # ('Kukkik', 'codium_raw_photos/kukkik1.JPG'),
    # ('Kukkik', 'codium_raw_photos/kukkik2.JPG'),
    #
    # ('Jane', 'codium_raw_photos/jane1.JPG'),
    # ('Jane', 'codium_raw_photos/jane2.JPG'),
    # ('Jane', 'codium_raw_photos/jane3.JPG'),
    # ('Jane', 'codium_raw_photos/jane4.JPG'),
    #
    # ('Winner', 'codium_raw_photos/winner1.JPG'),
    #
    # ('Saeed', 'codium_raw_photos/saeed1.JPG'),
    # ('Saeed', 'codium_raw_photos/saeed2.JPG'),
    # ('Saeed', 'codium_raw_photos/saeed3.JPG'),

]

known_face_names = []
known_face_encodings = []
for face in known_faces:
    known_face_names.append(face[0])
    face_image = face_recognition.load_image_file(face[1])
    # print(face_image)
    print(face_recognition.face_encodings(face_image))
    face_encoding = face_recognition.face_encodings(face_image)[0]
    known_face_encodings.append(face_encoding)

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        # ============== check distances first =================
        if min(face_distances) < 0.5:
            print('------- if ---------')
            print(matches[best_match_index])
            print(face_distances, min(face_distances))
            name = known_face_names[best_match_index]
            face_names.append(name)
        else:
            print('------- else ---------')
            print(matches[best_match_index])
            print(face_distances, min(face_distances))
            face_names.append("Unknown")

    #         if matches[best_match_index]:
    #             name = known_face_names[best_match_index]
    #         face_names.append(name)
    #         print(face_distances, min(face_distances))

    for face_location, name in zip(face_locations, face_names):
        (top, right, bottom, left) = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
