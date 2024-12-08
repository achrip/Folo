import gradio as gr
import numpy as np
import cv2 as cv

def magic(): 
    return

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

        input_image.stream(lambda f, m: f, [input_image, model_picker], [output_img], 
                           stream_every=0.0417083750,  concurrency_limit=24)
demo.launch()
