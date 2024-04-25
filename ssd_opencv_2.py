import numpy as np
import cv2
import argparse

# construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())


classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

proto = "MobileNetSSD_deploy.prototxt"
weights = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto , weights)
img = cv2.imread(args["image"])

img_resized = cv2.resize(img , (300 , 300))

blob = cv2.dnn.blobFromImage(img_resized , 0.007843 , (300 , 300) , 
                             (127.5 , 127.5 , 127.5) , False)
net.setInput(blob)
detections = net.forward()

height , width , _ = img.shape
final = detections.squeeze()

font = cv2.FONT_HERSHEY_SIMPLEX

for i in range(final.shape[0]):
  conf = final[i , 2]
  if conf > 0.5:
    class_name = classNames[final[i , 1]]
    x1 , y1 , x2 , y2 = final[i , 3:]
    x1 *= width
    y1 *= height
    x2 *= width
    y2 *= height
    top_left = (int(x1) , int(y1))
    bottom_right = (int(x2) , int(y2))
    img = cv2.rectangle(img , top_left , bottom_right , (0 , 255 , 0) , 3)
    img = cv2.putText(img , class_name , (int(x1) , int(y1) - 10) , font , 
                      1 , (255 , 0 , 0) , 2 , cv2.LINE_AA)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
