import os

import cv2 as cv
import gradio as gr

flag = True

def extract_frames(in_dirpath, out_dirpath): 
    if not os.path.exists(out_dirpath): 
        os.makedirs(out_dirpath)

    videos = os.listdir(in_dirpath)

    frame_count = 0
    for filename in videos: 
        capture = cv.VideoCapture(os.path.join(in_dirpath, filename))
        
        while True: 
            ret, frame = capture.read()

            if not ret: 
                break

            cv.imwrite(os.path.join(out_dirpath, f'{frame_count:04d}.png'), frame)
            frame_count += 1

        capture.release()

def classify(dirpath, out_dirpath): 
    pass

def magic(): 
    global flag 
    if not flag: 
        flag = True
    cap = cv.VideoCapture(0)
    while flag: 
        _, frame = cap.read()
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_board = contouring(frame_rgb)       
        view = change_view(frame_board, None)
        yield view
    pass

def change_view(frame, label): 
    if label == "Papan 1": 
        pass
    elif label == "Papan 2": 
        pass
    elif label == "Papan 3": 
        pass
    return frame

def stop_streaming(): 
    global flag
    flag = False

def otsu_thresh(image): 
    image_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_blurred = cv.GaussianBlur(image_grayscale, (5, 5), 0)

    _, binary_image = cv.threshold(image_blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return binary_image

def contouring(image): 
    binary_image = otsu_thresh(image)
    contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv.contourArea)

    x, y, w, h = cv.boundingRect(largest_contour)

    roi = image[y:y+h, x:x+w]
    #roi = binary_image[y:y+h, x:x+w]
    # consider adding few more pixels for the sake of padding

    return roi

with gr.Blocks() as demo: 
    with gr.Row(): 
        gr.Markdown("""
                    # Lecturer Detection Aid
                    """
                    )
    with gr.Row(): 
        model = gr.Radio(choices=["Viola-Jones", "DPM", "Yolov8", "MobileNet-SSD"],
                         label="Models", 
                         info="")
        view = gr.Radio(choices=["Papan 1", "Papan 2", "Papan 3", "Track", "Dosen"], 
                        label="View", 
                        info="")

    with gr.Column(): 
        output= gr.Image(streaming=True)
        main_button = gr.Button("Start", variant="primary")
        stop_button = gr.Button("Stop", variant="stop")

    main_button.click(magic, None, output)
    stop_button.click(stop_streaming, None, None)

if __name__ == "__main__": 
    demo.launch()
    pass

'''
Flowchart 

capture video -> get frames -> thresholding -> contouring -> (A)
> setelah contouring, kita akan punya papan tulis aja. 

(A) -> board segmenting -> 0/1 classification

TODO: 
    - program each button to display a different segment of the image when pressed
    - 
Notes: 
    - video input dapet dari opencv ajah
    - connected components analysis itu bagus untuk developing, bukan untuk prod. 
      (ga perlu pake cc analysis lagi kita)
    - kualitas thresholding lebih diutamakan. 
    - pakai binary thresholding kayak Otsu dan Binary.
      **JANGAN PAKE ADAPTIVE ATAU NON-BINARY!**
'''
