import cv2, os
import numpy as np

image_dir = './images/'
image_files = os.listdir(image_dir)

image = cv2.imread(os.path.join(image_dir, image_files[1]))

height, width, _ = image.shape

start_point = (420, 254)
end_point = (870, 380)
color = (0, 255, 0)
thickness = 3

cv2.rectangle(image, start_point, end_point, color, thickness)

cropped_image = image[start_point[1]:end_point[1], 
                      start_point[0]:end_point[0]]

cv2.imshow('image', image)
cv2.imshow('frame', cropped_image)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
