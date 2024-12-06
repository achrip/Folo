import cv2
import os
import numpy as np

def preprocess_image(img):
    img = cv2.equalizeHist(img)
    return img

dataset_path = 'dataset/'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_dict = {}

label_id = 0
target_size = (100, 100)


for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_path):
        label_dict[label_id] = person_name
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # print(f"Gambar {img_path} dapat dibaca")
            if img is None:
                # print(f"Gambar {img_path} tidak dapat dibaca.")
                continue
            
            faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            if len(faces_detected) == 0:
                # print(f"Tidak ada wajah yang terdeteksi di {img_path}.")
                continue

            for (x, y, w, h) in faces_detected:
                face = img[y:y+h, x:x+w]
                face_resized = cv2.resize(face, target_size)
                faces.append(face_resized.astype('uint8'))
                labels.append(label_id)
        
        label_id += 1


faces = np.array(faces, dtype='uint8')
labels = np.array(labels)

recognizer.train(faces, labels)
recognizer.save('face_recognizer_model.yml')
print("Model pengenalan wajah telah dilatih dan disimpan.")
model_path = 'face_recognizer_model.yml'

cap = cv2.VideoCapture('./images/face_test.mp4')

if not cap.isOpened():
    print("Kamera tidak dapat diakses.")
    cap.release()
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

label_dict = {0: "Person1", 1: "Person2", 2: "Person3", 3: "Person4"}

tracking_window = None
roi_hist = None
initialized = False
selected_segment = 0

while True:
    ret, frame = cap.read()

#    if ret:
#        frame = cv2.resize(frame, (640, 480))
#    else:
#        break

    height, width, _ = frame.shape
    segment_width = width // 3
    segments = [ frame[:, 0:segment_width], 
                frame[:, segment_width:2*segment_width],
                frame[:, 2*segment_width:] ]

    cv2.line(frame, (segment_width, 0), (segment_width, height), (0, 255, 0), 2)
    cv2.line(frame, (2 * segment_width, 0), (2 * segment_width, height), (0, 255, 0), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    cv2.imshow('Deteksi Wajah dan Segmentasi Kamera', frame)

    print('{0}, type: {1}'.format(len(faces), type(faces)))
    if len(faces) != 0: 
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (100, 100))

            label_id, confidence = recognizer.predict(face_resized)

            if confidence < 80:
                name = label_dict.get(label_id, "Tidak Dikenal")
            else:
                name = "Tidak Dikenal"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({int(confidence)})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            
            cx = x + w // 2

            if cx < segment_width:
                location = "Segment 1 (Kiri)"
                selected_segment = 0    
            elif cx < 2 * segment_width:
                location = "Segment 2 (Tengah)"
                selected_segment = 1
            else:
                location = "Segment 3 (Kanan)"
                selected_segment = 2
            
            cv2.putText(frame, location, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if selected_segment is not None:
            cv2.imshow('Segment Terpilih', segments[selected_segment])

    else: 
        # disini ga terdeteksi ada wajah di dalam frame, jadi kita pakai frame sebelumnya.
        print(selected_segment)
        cv2.imshow('Segment Terpilih', segments[selected_segment])

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(),
cv2.destroyAllWindows()

