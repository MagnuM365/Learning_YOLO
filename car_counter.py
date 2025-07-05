from ultralytics import YOLO
import cv2 as cv
import math
from sort import *

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv.imread('D:\python files\YOLO\Photos\mask.png')

cap = cv.VideoCapture("D:\python files\YOLO\Videos\carcount1.mp4") #for videos

model = YOLO('yolov8n.pt')

#tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

#line for counter
limits = [360, 460, 1024, 460]
total_count = []

while True:
    ret, img = cap.read()
    frame =cv.resize(img, (1024, 570)) 
    frameRegion = cv.bitwise_and(frame, mask)

    result = model(frameRegion, stream=True)

    detection = np.empty((0, 5))

    for r in result:
        boxes = r.boxes
        for box in boxes:
            #bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            
            #confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # print(conf)

            #classlabel
            cls = int(box.cls[0])
            classLabel = classNames[cls]

            if classLabel in ['car'] and conf > 0.3:
                #cv.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)
                #cv.putText(frame, f'{classLabel}: {conf}', (max(0, x1), max(40, y1)), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv.LINE_4)
                classArray = np.array([x1, y1, x2, y2, conf])
                detection = np.vstack((detection, classArray))

    resultTracker = tracker.update(detection)

    #cv.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0,0, 255), 5)
    
    for result in resultTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv.rectangle(frame, (x1, y1), (x2,y2), (255, 0, 0), 3)
        cv.putText(frame, f'{Id}', (max(0, x1), max(40, y1)), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv.LINE_4)

        #count
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        #cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if Id not in total_count: 
                total_count.append(Id)
        
    cv.putText(frame, f'Count: {len(total_count)}', (100, 50), cv.FONT_HERSHEY_PLAIN, 3, (50, 180, 100), 2, cv.LINE_4)

    cv.imshow("webcam", frame)
    # cv.imshow("webcam", frameRegion)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv.destroyAllWindows()

