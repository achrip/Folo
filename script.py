import cv2 as cv

img = cv.imread('./dataset/images/0111.png', 0)

boxes = cv.selectROIs('select rois', img, showCrosshair=False)

for box in boxes: 
    cv.rectangle(img, (int(box[0]),int(box[1])), 
                 (int(box[0]+box[2]),int(box[1]+box[3])), 
                 thickness=3, color=(0, 255, 0))

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()


