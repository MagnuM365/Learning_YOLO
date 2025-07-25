import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Target object to detect
TARGET_CLASS = 'bottle' 

# Real width of the target object in meters
REAL_WIDTH = 0.07  # bottle width ~7cm

# Focal length (calibrated)
FOCAL_LENGTH = 1000  # example value, calibrate for your camera

# Distance to stop (meters)
STOP_DISTANCE = 0.2  # Stop when closer than 20 cm

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

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
            pixel_width = x2 - x1

            # Distance estimation
            if pixel_width > 0:
                distance = (REAL_WIDTH * FOCAL_LENGTH) / pixel_width
                distance = round(distance, 2)
            else:
                distance = None

            # Determine direction
            if cx < frame_center - 80:
                direction = "Turn Left"
            elif cx > frame_center + 80:
                direction = "Turn Right"
            else:
                direction = "Move Forward"

            # Check if car should stop
            if distance is not None and distance <= STOP_DISTANCE:
                action = "Stop"
            else:
                action = direction

            detected = True

            # Draw bounding box and info on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, (y1 + y2) // 2), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{class_name} | Dist: {distance}m", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Action: {action}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            print(f"[{class_name}] Distance: {distance} m | Action: {action}")

            break  # process only first detected target object

    if not detected:
        cv2.putText(frame, f"'{TARGET_CLASS}' not detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw center line for reference
    cv2.line(frame, (frame_center, 0), (frame_center, frame_height), (255, 0, 0), 1)

    cv2.imshow("Object Direction & Distance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
