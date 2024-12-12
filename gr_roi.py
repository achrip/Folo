import gradio as gr
import cv2
import numpy as np

# Global variables for drawing the rectangle
ix, iy = -1, -1
drawing = False
roi_selected = False

# Function to handle mouse events
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img_copy, roi_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = frame.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (255, 0, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_selected = True
        cv2.rectangle(img_copy, (ix, iy), (x, y), (255, 0, 0), 2)

# Function to process the webcam feed and select ROI
def process_frame(frame):
    global img_copy
    img_copy = frame.copy()

    # Set mouse callback to draw rectangle
    cv2.selectROI("Webcam Feed", img_copy)
    # cv2.setMouseCallback("Webcam Feed", draw_rectangle)

    while True:
        cv2.imshow("Webcam Feed", img_copy)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # Press 'Esc' to exit
            break
            
    cv2.destroyAllWindows()

    # Return the selected ROI as an image if selected
    if roi_selected:
        roi = frame[y:iy, x:ix]
        return roi
    return frame

# Gradio interface setup
with gr.Blocks() as demo:
    gr.Markdown("## Webcam ROI Selector")
    
    input_img = gr.Image(sources=["webcam"], type="numpy")
    output_img = gr.Image(type="numpy")
    
    # Stream the webcam feed and process each frame for ROI selection
    input_img.stream(process_frame, [input_img], [output_img], time_limit=30)

# Launch the Gradio app
demo.launch()

