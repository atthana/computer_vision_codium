import pickle

import face_recognition

import known_faces as faces

face_locations = []
face_encodings = []
face_names = []

process_this_frame = True

known_face_names = []
known_face_encodings = []

for face in faces.known_faces:
    try:
        print(face)
        known_face_names.append(face[0])
        face_image = face_recognition.load_image_file(face[1])
        face_encoding = face_recognition.face_encodings(face_image)[0]
        known_face_encodings.append(face_encoding)


    except IndexError as err:
        print('--- Exception ---')
        print(err)
        pass


with open('dataset_codium_encoding.dat', 'wb') as f:
    pickle.dump(known_face_encodings, f)

with open('dataset_codium_name.dat', 'wb') as f:
    pickle.dump(known_face_names, f)

print('Finished creating dataset')
