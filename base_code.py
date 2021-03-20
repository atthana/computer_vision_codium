import cv2
import face_recognition
import numpy as np

# https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
# 1. Need to load all images and resize photos and make a new set in new folder.
# 2. Fixed red error during running and found.
# 3. Refactor code to collect all name and location in the same place like teacher's code.
# Use this code to be code base because it's faster than teacher's code.


# atthana_image = face_recognition.load_image_file('resize_raw_photos/q1.jpg')
# print(atthana_image)
# atthana_face_encoding = face_recognition.face_encodings(atthana_image)[0]
#
# known_face_encodings = [
#     atthana_face_encoding,
#
# ]
# known_face_names = [
#     "Atthana",
#
# ]

face_locations = []
face_encodings = []
face_names = []

process_this_frame = True
# ========================================================
known_faces = [

    # ('Beer', 'image/face/beer.jpg'),
    # ('Pang', 'image/face/pang.jpg'),
    # ('Awesome', 'image/face/Awesome.jpg'),
    # ('Q', 'resize_raw_photos/q1.jpg')

    ('Q', 'resize_raw_photos/q1.jpg'),
    ('Q', 'resize_raw_photos/q2.jpg'),  # พอใช้ q2.jpg มันก้อพังเลย แต่พอ comment บรรทัดนี้มันก้อใช้งานได้นะ
    # ('Q', 'resize_raw_photos/q2.jpg'),
    # ('Q', 'resize_raw_photos/q3.jpg'),
    # ('Q', 'resize_raw_photos/q4.jpg'),
    # ('Q', 'resize_raw_photos/q5.jpg'),
    #
    # ('Kukkik', 'resize_raw_photos/kukkik1.jpg'),
    # ('Kukkik', 'resize_raw_photos/kukkik2.jpg'),
    #
    # ('Jane', 'resize_raw_photos/jane1.jpg'),
    # ('Jane', 'resize_raw_photos/jane2.jpg'),
    # ('Jane', 'resize_raw_photos/jane3.jpg'),
    # ('Jane', 'resize_raw_photos/jane4.jpg'),
    #
    # ('Winner', 'resize_raw_photos/winner1.jpg'),
    #
    # ('Saeed', 'resize_raw_photos/saeed1.jpg'),
    # ('Saeed', 'resize_raw_photos/saeed2.jpg'),
    # ('Saeed', 'resize_raw_photos/saeed3.jpg'),

]


known_face_names = []
known_face_encodings = []
for face in known_faces:
    known_face_names.append(face[0])
    face_image = face_recognition.load_image_file(face[1])
    print('--- face_image ----')
    print(face_image)
    print('--------face_recognition.face_encodings----------')
    print(face_recognition.face_encodings(face_image))
    face_encoding = face_recognition.face_encodings(face_image)[0]
    known_face_encodings.append(face_encoding)

# =========================================================

video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

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
