import time

import cv2 as cv
import gradio as gr
from ultralytics import YOLO

video_is_running = True
(x, y, w, h) = -1,-1,-1,-1
main_view = "" 
middle_frame_width = -1
bbox_width = 1

def update_view(frame, board_view, len, model): 
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

    objects = model(frame)

    global bbox_width
    for object in objects: 
        boxes = object.boxes.xyxy
        classes = object.boxes.cls

        for box, klass in zip(boxes, classes): 
            if klass != 0 : 
                # this means the object is not a person
                continue
            x1, _, x2, _ = map(int, box)
            bbox_width = (x1 + x2) // 2
            break

    if bbox_width < board_start: 
        return frame[:, :board_start]
    elif bbox_width <= board_end: 
        return frame[:, board_start:board_end]
    elif bbox_width > board_end: 
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

def model_picker(model_name): 
    if model_name.lower() == "viola-jones": 
        pass
    elif model_name.lower() == "dpm": 
        pass
    elif model_name.lower() == "yolo": 
        return YOLO("yolo11n.pt")
        pass
    elif model_name.lower() == "ssd-mobilenet":
        pass

def live_inference(frame, model_name, view): 
    global video_is_running, main_view, middle_frame_width
    main_view = view if main_view != "" else view
    model = model_picker(model_name)
    frame_board = contouring(frame)
    frame_display = update_view(frame_board, main_view, middle_frame_width, model)
    return frame_display

def magic(model_name, view): 
    global video_is_running, main_view, middle_frame_width
    main_view = view if main_view != "" else view
    cap = cv.VideoCapture("./dataset/videos/MVI_0014.MOV")
    model = model_picker(model_name)
    if not video_is_running: 
        video_is_running = True
    while video_is_running: 
        ret, frame = cap.read()
        if not ret: 
            return
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_board = contouring(frame_rgb)       
        frame_display = update_view(frame_board, main_view, middle_frame_width, model)
        yield frame_display
        time.sleep(.03)

    cap.release()

with gr.Blocks() as demo: 
    with gr.Row(): 
        gr.Markdown("""
                    # Lecturer Detection Aid
                    """
                    )
    with gr.Tab("Live Demo"): 
        with gr.Row(): 
            live_model = gr.Radio(choices=["Viola-Jones", "DPM", "Yolo", "MobileNet-SSD"],
                                  value="Yolo",
                                  label="Models", 
                                  info="")
            live_view = gr.Radio(choices=["Papan 1", "Papan 2", "Papan 3", "Tracking", "Full"], 
                                 value="Full",
                                 label="View", 
                                 info="")
        with gr.Row(): 
            live_input = gr.Image(sources=["webcam"])
            live_output = gr.Image(streaming=True)

        live_viewport_width = gr.Slider(minimum=300, maximum=600, value=400, 
                                            step=5, label="Lebar Papan")

    live_input.stream(
        live_inference,
        [live_input, live_model, live_viewport_width],
        [live_output],
        stream_every=0.0417083750,
        concurrency_limit=24,
    )

    with gr.Tab("Local Demo"): 
        with gr.Row(): 
            local_model = gr.Radio(choices=["Viola-Jones", "DPM", "Yolo", "MobileNet-SSD"],
                             value="Yolo",
                             label="Models", 
                             info="")
            local_view = gr.Radio(choices=["Papan 1", "Papan 2", "Papan 3", "Tracking", "Full"], 
                            value="Full",
                            label="View", 
                            info="")
            local_viewport_width = gr.Slider(minimum=300, maximum=600, value=400, 
                                            step=5, label="Lebar Papan")

        local_output= gr.Image(streaming=True, type="numpy")
        with gr.Row(): 
            local_stop_button = gr.Button("Stop", variant="stop")
            local_main_button = gr.Button("Start", variant="primary") 

    local_main_button.click(magic, [local_model, local_view], local_output)
    local_stop_button.click(stop_streaming, None, None)
    local_view.change(change_view, [local_view], None)
    local_viewport_width.change(change_width, [local_viewport_width], None)

if __name__ == "__main__": 
    demo.launch()

'''
Flowchart 

capture video -> get frames -> thresholding -> contouring -> (A)
> setelah contouring, kita akan punya papan tulis aja. 

(A) -> board segmenting -> 0/1 classification

TODO: 
    - add a method that will be called when the program first initializes. 
      this method should assign new default values to global variables.
'''
