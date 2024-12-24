import os
import cv2 as cv
from sklearn.metrics import confusion_matrix, classification_report

DATASET_DIR = "../datasets/images/" # change this if necessary
POSITIVE_DIR = os.path.join(DATASET_DIR, "positive")
NEGATIVE_DIR = os.path.join(DATASET_DIR, "negative")

LABELS = [] 
PREDICTIONS = [] 


model = cv.dpm.DPMDetector_create("./inriaperson.xml")

for filename in os.listdir(POSITIVE_DIR):
    if filename.endswith(".png"): 
        image = cv.imread(os.path.join(POSITIVE_DIR, filename))
        grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        detections = model.detect(grayscale)

        LABELS.append(1)
        PREDICTIONS.append(1 if len(detections) > 0 else 0)

for filename in os.listdir(NEGATIVE_DIR):
    if filename.endswith(".png"): 
        image = cv.imread(os.path.join(NEGATIVE_DIR, filename))
        grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        detections = model.detect(grayscale)

        LABELS.append(0)
        PREDICTIONS.append(1 if len(detections) > 0 else 0)

conf_matrix = confusion_matrix(LABELS, PREDICTIONS)
report = classification_report(LABELS, PREDICTIONS, 
                               target_names=["No Person", "Person"])

TP = conf_matrix[1][1]
TN = conf_matrix[0][0]
FP = conf_matrix[0][1]
FN = conf_matrix[1][0]

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
