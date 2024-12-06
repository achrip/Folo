import cv2 as cv

image = cv.imread('./images/200620.jpg')

r = cv.selectROIs("select ROI", image, showCrosshair=False)

cropped = image[int(r[0][1]):int(r[0][1]+r[0][3]),
                int(r[0][0]):int(r[0][0]+r[0][2])]

cv.imshow('cropped', cropped)
cv.waitKey(0)
