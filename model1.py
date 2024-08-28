import cv2
import numpy as np
from twilio.rest import Client

# Load YOLO model
def load_your_model(weights_path, cfg_path):
    return cv2.dnn.readNet(weights_path, cfg_path)

# Function to detect people in the frame
def detect_people(model, frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    model.setInput(blob)
    output_layers = model.getUnconnectedOutLayersNames()
    outs = model.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == classes.index('person'):
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detections = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            detections.append((x, y, w, h))
    
    return detections

# Function to detect pose and running
def analyze_behavior(detections, frame):
    # Implement pose detection here
    # Implement logic to detect if hands are raised
    
    # Implement running detection based on movement over frames
    
    is_hands_raised = False  # Placeholder
    is_running = False  # Placeholder
    
    return is_hands_raised, is_running

# Function to send an SMS alert
def send_sms_alert():
    account_sid = 'AC847c5aa5daf401ebe869724bfc94dc4d'
    auth_token = 'e898d5f599fbc84b0ced9ca90689217d'
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body="Emergency Alert: A woman is detected in a potentially dangerous situation.",
        from_='+your_twilio_number',  # Replace with your Twilio phone number
        to='+police_control_room_number'  # Replace with the recipient's phone number
    )

    print(f"Message sent with SID: {message.sid}")

# Initialize video capture
cap = cv2.VideoCapture(0)  # Or the path to a video file

# Load the pre-trained YOLO model
model = load_your_model('yolov3.weights', 'yolov3.cfg')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect people in the frame
    detections = detect_people(model, frame)
    
    # Analyze the behavior to check for hands raised or running
    is_hands_raised, is_running = analyze_behavior(detections, frame)
    
    # If the condition is met, send an SMS alert
    if is_hands_raised or is_running:
        send_sms_alert()
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
