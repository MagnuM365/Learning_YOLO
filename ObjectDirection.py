import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

TARGET_CLASS = 'person' 

# Initialize webcam
cap = cv2.VideoCapture(0)

frame_center = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    frame_center = frame_width // 2

    results = model(frame)[0]
    detected = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name == TARGET_CLASS:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detected = True

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Simulated movement
            if cx < frame_center - 80:
                direction = "Turn Left"
            elif cx > frame_center + 80:
                direction = "Turn Right"
            else:
                direction = "Move Forward"

            # Size-based stopping simulation
            object_width = x2 - x1
            if object_width > frame_width * 0.5:
                direction = "Stop (Object is close)"

            print(f"[Detected '{class_name}'] Action: {direction}")
            cv2.putText(frame, direction, (cx - 60, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            break  # Just take the first matching object

    if not detected:
        cv2.putText(frame, f"'{TARGET_CLASS}' not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.line(frame, (frame_center, 0), (frame_center, frame_height), (255, 0, 0), 1)  # Visual center line

    cv2.imshow("Object Tracking Simulation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
