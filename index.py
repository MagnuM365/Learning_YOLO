from ultralytics import YOLO
import cv2
 
model = YOLO('yolov8n.pt')
results = model("YOLO\Photos\group1.jpg", show=True)

results[0].show()
cv2.waitKey(0)