import cv2 as cv
import gradio as gr
from ultralytics import YOLO
import time

video_is_running = True
(x, y, w, h) = -1,-1,-1,-1
main_view = "" 
middle_frame_width = -1

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

    if width < board_start: 
        return frame[:, :board_start]
    elif width <= board_end: 
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

def magic(model): 
    '''
    model   : object detection model that is used for person tracking.
    view    : view option for the boards.
    width_m : width of the middle board which varies according to rooms
    '''
    model = YOLO("yolo11n.pt")
    global video_is_running, main_view, middle_frame_width
    if not video_is_running: 
        video_is_running = True
    cap = cv.VideoCapture("./dataset/videos/MVI_0002.MOV")
    while video_is_running: 
        ret, frame = cap.read()
        if not ret: 
            return
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_board = contouring(frame_rgb)       
        frame_display = update_view(frame_board, main_view, middle_frame_width, w)
        yield frame_display
        #time.sleep(.02)

with gr.Blocks() as demo: 
    with gr.Row(): 
        gr.Markdown("""
                    # Lecturer Detection Aid
                    """
                    )
    with gr.Row(): 
        model = gr.Radio(choices=["Viola-Jones", "DPM", "Yolo", "MobileNet-SSD"],
                         value="Yolo",
                         label="Models", 
                         info="")
        view = gr.Radio(choices=["Papan 1", "Papan 2", "Papan 3", "Tracking", "Full"], 
                        value="Full",
                        label="View", 
                        info="")
        width_mid = gr.Slider(minimum=400, maximum=600, value=400, 
                              step=5, label="Lebar Papan")

    with gr.Column(): 
        output= gr.Image(streaming=True, type="numpy")
        main_button = gr.Button("Start", variant="primary")
        stop_button = gr.Button("Stop", variant="stop")

    main_button.click(magic, [model], output)
    stop_button.click(stop_streaming, None, None)
    view.change(change_view, [view], None)
    width_mid.change(change_width, [width_mid], None)

if __name__ == "__main__": 
    demo.launch()
    pass

'''
Flowchart 

capture video -> get frames -> thresholding -> contouring -> (A)
> setelah contouring, kita akan punya papan tulis aja. 

(A) -> board segmenting -> 0/1 classification

TODO: 
    - add a method that will be called when the program first initializes. 
      this method should assign new default values to global variables.
'''
