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

original_height, original_width = cropped_image.shape[:2]

# keep aspect ratio
new_height = 720 
aspect_ratio = original_width / original_height

new_width = int(new_height / aspect_ratio)

upscaled_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

cv2.imshow('image', image)
cv2.imshow('cropped', cropped_image)
cv2.imshow('upscaled', upscaled_image)

cv2.imwrite('./out/papan_tulis.png', upscaled_image)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
