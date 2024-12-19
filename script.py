import cv2
import os
from skimage import filters
import numpy as np

def otsu_thresh(image): 
# Apply GaussianBlur to reduce noise
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Otsu"s thresholding
    otsu_threshold, binary_image = cv2.threshold(image_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print("Otsu's Threshold Value:", otsu_threshold)

# Display results
    cv2.imshow("Original Image", image)
    cv2.imshow("Binary Image", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return binary_image

def adaptive_thresh(image, opt): 
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Apply adaptive thresholding
    adaptive_threshold_mean = cv2.adaptiveThreshold(blurred_image, 255,
                                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                                    cv2.THRESH_BINARY, 7, 3)

    adaptive_threshold_gaussian = cv2.adaptiveThreshold(blurred_image, 255,
                                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                        cv2.THRESH_BINARY, 11, 2)

# Display results
    cv2.imshow("Adaptive Mean Thresholding", adaptive_threshold_mean)
    cv2.imshow("Adaptive Gaussian Thresholding", adaptive_threshold_gaussian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if opt == "gaussian": 
        return adaptive_threshold_gaussian
    return adaptive_threshold_mean

def savuola_thresh(image): 
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

    savuola_threshold = filters.threshold_sauvola(image_blurred, 
                                                  window_size=15,
                                                  k=0.3)
    binary_savuola = image_blurred > savuola_threshold

    cv2.imshow('Savuola Low Contrast Thresholding', binary_savuola.astype(np.uint8)*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return binary_savuola.astype(np.uint8) * 255

def contouring(image, opt): 
# Load the binary image
    binary_image = None
    if opt == "otsu": 
        binary_image = otsu_thresh(image)
    elif opt == "adp_gaussian": 
        binary_image = adaptive_thresh(image, 0)
    elif opt == "adp_mean":
        binary_image = adaptive_thresh(image, "gaussian")
    elif opt == "savuola":
        binary_image = savuola_thresh(image)
    else : 
        print("Choose a thresholding algorithm!")
        return
# Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the largest contour corresponds to the whiteboard
    largest_contour = max(contours, key=cv2.contourArea)

# Get bounding box coordinates
    x, y, w, h = cv2.boundingRect(largest_contour)

# Extract ROI
    roi = binary_image[y:y+h, x:x+w]
    # consider adding few more pixels for the sake of padding

# Display results
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cc_analysis(image, opt): 
    binary_image = None
    if opt == "otsu": 
        binary_image = otsu_thresh(image)
    elif opt == "adp_gaussian": 
        binary_image = adaptive_thresh(image, 0)
    elif opt == "adp_mean":
        binary_image = adaptive_thresh(image, "gaussian")
    elif opt == "savuola":
        binary_image = savuola_thresh(image)
    else : 
        print("Choose a thresholding algorithm!")
        return
    # Apply connected components analysis
    num_labels, labels_im = cv2.connectedComponents(binary_image)

    # Create an output image to visualize connected components
    output_connected = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)

    for label in range(1, num_labels):
        mask = (labels_im == label).astype(np.uint8) * 255
        output_connected[mask > 0] = [np.random.randint(0, 255) for _ in range(3)]  # Random colors

    # Display results
    cv2.imshow('Connected Components', output_connected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cc_analysis_with_stats(image, opt): 
    binary_image = None
    if opt == "otsu": 
        binary_image = otsu_thresh(image)
    elif opt == "adp_gaussian": 
        binary_image = adaptive_thresh(image, 0)
    elif opt == "adp_mean":
        binary_image = adaptive_thresh(image, "gaussian")
    elif opt == "savuola":
        binary_image = savuola_thresh(image)
    else : 
        print("Choose a thresholding algorithm!")
        return

    # Perform connected components analysis
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # Create an output image to visualize the labeled components
    output_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)

    # Create a color map for visualization
    colors = np.random.randint(0, 255, size=(num_labels, 3))

    # Loop over each component and assign colors
    for i in range(num_labels):
        # Skip the background label (label 0)
        if i == 0:
            continue
        
        # Create a mask for the current component
        mask = (labels_im == i).astype(np.uint8) * 255
        
        # Color the component in the output image
        output_image[mask > 0] = colors[i]

        # Optionally draw bounding box and centroid
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Draw bounding box
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw centroid
        cX, cY = centroids[i]
        cv2.circle(output_image, (int(cX), int(cY)), 5, (255, 0, 0), -1)

# Display results
    cv2.imshow('Labeled Components', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__== "__main__": 
    path = "./dataset/images"
    for filename in os.listdir(path): 
        if filename != "9300.png": 
            continue
        image = cv2.imread(os.path.join(path, filename), 0)
        #print(image.shape)
        image_cropped = image[:550, :]
        contouring(image_cropped, "otsu")
        #cc_analysis(image_cropped, "otsu")
        #cc_analysis_with_stats(image_cropped, "otsu")
