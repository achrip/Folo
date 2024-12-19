import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
cap = cv.VideoCapture("./dataset/videos/MVI_0014.MOV")
x, y, w, h = -1, -1, -1, -1

def update_view(frame, board_view, len, width): 
    frame_width = frame.shape[1]
    mid = frame_width // 2
    board_start = mid - len // 2
    board_end = mid + len // 2
    if board_view == "Papan 1": 
        return frame[:, :board_start]
    elif board_view == "Papan 2": 
        return frame[:, board_start:board_end]
    elif board_view == "Papan 3": 
        return frame[:, board_end:]
    elif board_view == "Full": 
        return frame

    width = width // 2
    if width < board_start: 
        return frame[:, :board_start]
    elif width <= board_end : 
        return frame[:, board_start:board_end]
    elif width > board_end: 
        return frame[:, board_end:]

def stop_streaming(): 
    global video_is_running
    video_is_running = False

def change_view(view): 
    global main_view 
    main_view = view

def change_width(width): 
    global middle_frame_width
    middle_frame_width = width

def otsu_thresh(image): 
    image_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_grayscale = cv.equalizeHist(image_grayscale)
    image_blurred = cv.GaussianBlur(image_grayscale, (5, 5), 0)

    _, binary_image = cv.threshold(image_blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return binary_image

def contouring(image): 
    image = image[:550, :]
    binary_image = otsu_thresh(image)
    contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv.contourArea)

    global x, y, w, h

    if -1 in (x, y, w, h):
        x, y, w, h = cv.boundingRect(largest_contour)

    #x = x-20 if x-20 > 0 else 0
    #y = y-20 if y-20 > 0 else 0
    roi = image[y:y+h+20, x:x+w+20]

    return roi
while True: 
    _, frame = cap.read()

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_board = contouring(frame_rgb)       
    frame_board = contouring(frame)       

    results = model(frame_board)

    frame_display = frame

    for result in results:
        boxes = result.boxes.xyxy  
        confidences = result.boxes.conf  
        classes = result.boxes.cls  

        for box, conf, cls in zip(boxes, confidences, classes):
            if cls == 0:  
                x1, y1, x2, y2 = map(int, box)
                cv.rectangle(frame_board, (x1, y1), (x2, y2), (0, 255, 0), 2)
                frame_display = update_view(frame_board, "", 500, x1+x2)

    cv.imshow('Video', frame_display)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
