import os

import cv2 as cv
import gradio as gr
import numpy as np


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
    positive_dirpath = os.path.join(out_dirpath, 'positive' )
    negative_dirpath = os.path.join(out_dirpath, 'negative' )
    cc = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_fullbody.xml')

    if not os.path.exists(positive_dirpath): 
        os.makedirs(positive_dirpath)

    if not os.path.exists(negative_dirpath): 
        os.makedirs(negative_dirpath)

    images = os.listdir(dirpath)
    for filename in images: 
        image = cv.imread(os.path.join(dirpath, filename), 0)
        obj = cc.detectMultiScale(image, 1.05, 3)

        if len(obj) < 1: 
            cv.imwrite(os.path.join(negative_dirpath, filename),
                       cv.cvtColor(image, cv.COLOR_GRAY2RGB))

        # an object (fullbody) is detected. draw a bounding box 
        # around it
        for (x, y, w, h) in obj: 
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        cv.imwrite(os.path.join(positive_dirpath, filename),
                   cv.cvtColor(image, cv.COLOR_GRAY2RGB))

def magic(frame, algo): 
    pass


with gr.Blocks() as demo: 
    with gr.Row(): 
        with gr.Column(): 
            model_picker = gr.Dropdown(choices=['Viola-Jones', 'DPM', 'Yolov8', 'MobileNet-SSD'],
                                       value='Yolov8', label='Models')
            input_image = gr.Image(sources=['webcam'], 
                                   type='numpy')
        
        with gr.Column(): 
            with gr.Row():
                btn_1 = gr.Button(value='Papan 1')
                btn_2 = gr.Button(value='Papan 2')
                btn_3 = gr.Button(value='Papan 3')
                btn_4 = gr.Button(value='Dosen')

                upscale_algorithm = gr.Dropdown(choices=['linear', 'cubic', 'none'], 
                                                value='none', label='Upscaling Algorithm')

            output_img = gr.Image(streaming=True)

        '''
        the arguments `concurrency_limit` is ideally based on the received input
        and desired output FPS. if this is not applicable, then just for the latter.

        the `stream_every` value is obtained by dividing 1 by the value of concurrency
        limit. this is to ensure that the video playback has exactly the FPS that
        we desire. 
        '''
        # input_image.stream(lambda f, m: f, [input_image, model_picker], [output_img], 
        #                    stream_every=0.0416666667, concurrency_limit=24)

        input_image.stream(magic, [input_image, upscale_algorithm], [output_img], 
                           stream_every=0.0417083750,  concurrency_limit=24)
if __name__ == "__main__": 
    # demo.launch()
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
    - connected components analysis itu bagus untuk developing, bukan untuk prod. 
      (ga perlu pake cc analysis lagi kita)
    - kualitas thresholding lebih diutamakan. 
    - pakai binary thresholding kayak Otsu dan Binary.
      **JANGAN PAKE ADAPTIVE ATAU NON-BINARY!**
'''
