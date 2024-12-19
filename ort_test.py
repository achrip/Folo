import cv2 as cv
import numpy as np
import onnxruntime as ort
import onnx

#model = onnx.hub.load("Tiny YOLOv3")
session = ort.InferenceSession("./models/yolo11n.onnx")

def preprocess(image, w, h): 
    height, width = image.shape[:2]
    scale = min(w/width, h/height)
    resized_image = cv.resize(image, (int(width*scale), int(height*scale)), 
                              interpolation=cv.INTER_LINEAR)
    resized_image = resized_image.transpose(2,0,1)
    print(resized_image.ndim)
    pad_height = (h - resized_image.shape[1])//2
    pad_width = (w - resized_image.shape[2])//2
    padded_image = cv.copyMakeBorder(resized_image, pad_height, pad_height, 
                                     pad_width, pad_width, cv.BORDER_REPLICATE, 
                                     None, [])
    return padded_image.astype(np.float32) / 255.0

def postprocess(outputs, confidence_threshold=0.5): 
    boxes = [] 
    confidences = [] 
    class_ids = [] 

    for output in outputs: 
        for detection in output: 
            scores = detection[5:]
            class_id = np.argmax(scores)

            confidence = scores[class_id]

            if confidence > confidence_threshold:
                center_x, center_y, width, height = (
                    detection[0, 4] * np.array([416, 416, 416, 416])
                ).astype(int)

                x1 = int(center_x - width / 2)
                y1 = int(center_y - height / 2)
                x2 = int(center_x + width / 2)
                y2 = int(center_y + height / 2)

                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

def detect_and_crop_region(video_souce:str | int=0): 
    cap = cv.VideoCapture(video_souce)

    while True: 
        ret, frame = cap.read()
        if not ret: 
            break

        input_image = preprocess(frame, 640, 640)
        input_image = np.expand_dims(input_image, axis=0)

        outputs = session.run(None, {session.get_inputs()[0].name: input_image})

        boxes, confs, ids = postprocess(outputs)

        for box in boxes: 
            x1, y1, x2, y2 = box

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            center_x = (x1 + x2) // 2

            if center_x <= 426: 
                cropped_frame = frame[:, :427]
            elif center_x <= 852: 
                cropped_frame = frame[:, 427:853]
            else: 
                cropped_frame = frame[:, 853:]

            cv.imshow('Cropped Frame', cropped_frame)
        cv.imshow("Video", frame)

        if cv.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv.destroyAllWindows()

detect_and_crop_region("./dataset/videos/MVI_0002.MOV")
