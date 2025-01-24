import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO(r"D:\Titan\Projects\yololive\best.pt")

# Define class names and assign unique colors for each class
class_names = ["rsc", "looper", "thrips", "jassid", "rsm", "tmb", "healthy"]
class_colors = {
    "rsc": (255, 0, 0),       # Red
    "looper": (0, 255, 0),    # Green
    "thrips": (0, 0, 255),    # Blue
    "jassid": (255, 255, 0),  # Yellow
    "rsm": (255, 0, 255),     # Magenta
    "tmb": (0, 255, 255),     # Cyan
    "healthy": (128, 0, 128)  # Purple
}

# Open webcam
cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    # Perform object detection
    results = model(frame)

    # Draw detections on the frame
    for result in results:
        for box in result.boxes:
            label = class_names[int(box.cls)]  # Get class label
            confidence = float(box.conf)  # Confidence score
            bbox = [int(coord) for coord in box.xyxy[0]]  # Bounding box coordinates
            x1, y1, x2, y2 = bbox
            
            # Get color for the class
            color = class_colors.get(label, (255, 255, 255))  # Default to white
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f"{label}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (w, h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w + 5, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)

    # Display the frame with detections
    cv2.imshow("YOLO Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
