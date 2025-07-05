from ultralytics import YOLO
import cv2 as cv
import math

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

#cap = cv.VideoCapture(0) #for webcam

cap = cv.VideoCapture("YOLO\Videos\\carcount1.mp4") #for videos

model = YOLO('yolov8n.pt')

while True:
    ret, img = cap.read()

    frame =cv.resize(img, (1280, 720)) 
    result = model(frame, stream=True)
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

            if classLabel == 'car' or classLabel=='truck' or classLabel=='motorbike' or classLabel=='bus' and conf > 0.3:
                cv.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)
                cv.putText(frame, f'{classNames[cls]}: {conf}', (max(0, x1), max(40, y1)), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv.LINE_4)

    cv.imshow("webcam", frame)
    # cv.imshow("webcam", frameRegion)
    key = cv.waitKey(0) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv.destroyAllWindows()

