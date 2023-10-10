import cv2
import time
import torch
import torchvision.transforms as T
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import functional as F

# Load the pre-trained SSD model
model = ssdlite320_mobilenet_v3_large(pretrained=True)

# Move the model to a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


model.eval()  # Set the model to evaluation mode

def preprocess(image):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((320, 320)),  # SSD320 model requires 320x320 input
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def detect_person(frame):
    input_tensor = preprocess(frame)
    with torch.no_grad():
        predictions = model(input_tensor)

    # Extract bounding box coordinates and labels
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    # Filter boxes to keep only person detections (class 1)
    person_boxes = boxes[labels == 1]

    return person_boxes

def draw_boxes(image, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

# Initialize the webcam or video source
cap = cv2.VideoCapture(0)  # Change to your video source if needed

# Set the time interval for person detection (3 seconds)
detection_interval = 5  # in seconds
last_detection_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))  # Adjust the size as needed


    current_time = time.time()
    if current_time - last_detection_time >= detection_interval:
        detected_boxes = detect_person(frame)
        result_frame = draw_boxes(frame.copy(), detected_boxes)
        cv2.imshow('Person Detection', result_frame)
        last_detection_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
