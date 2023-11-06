import torch
import cv2
import numpy as np

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# Set your default values here
weights_path = 'yolov8n.pt'  # Model weights path
img_size = 320  # Inference size (pixels)

def main():
    # Initialize the frame dimensions (we'll set them as soon as we read the first frame from the video)
    W = None
    H = None

    # Initialize device and model
    device = select_device('')
    model = attempt_load(weights_path, device=device)
    model.eval()

    # OpenCV VideoCapture for camera feed (usually camera index 0)
    cap = cv2.VideoCapture(0)

    person_info = {}  # Dictionary to store person information
    entry_count = 0  # Counter for people entering
    exit_count = 0   # Counter for people exiting

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # List of People detected
        rects = []

        # Draw a vertical line in the center of the frame
        cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 0, 0), 1)

        # Resize and preprocess the input image
        img = cv2.resize(frame, (img_size, img_size))
        img = img[:, :, ::-1]  # BGR to RGB
        img = img.copy()  # Make a copy of the array
        img = torch.from_numpy(img).to(device).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        # Perform object detection
        results = model(img)

        # Post-process detections
        results = non_max_suppression(results, conf_thres=0.4, iou_thres=0.4)[0]

        # Draw bounding boxes for class 0 (person) only
        if results is not None and len(results) > 0:
            for det in results:
                x1, y1, x2, y2, conf, class_id = det
                x1, y1, x2, y2 = int(x1 * 2), int(y1 * 2), int(x2 * 2), int(y2 * 2)
                if int(class_id) == 0:  # Class 0 is a person
                    label = f'Person'
                    confidence = float(conf)
                    if confidence < 0.7:
                        break
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label} ({confidence:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Track person entry
                    person_id = id(det)
                    if person_id not in person_info:
                        person_info[person_id] = {'entered': False, 'previous_position': (x1, y1)}

                    # Check the person's previous position to determine the direction
                    previous_x, previous_y = person_info[person_id]['previous_position']
                    if x1 > previous_x and x1 >= W // 2:
                        direction = 'Going to Right'
                        if not person_info[person_id]['entered']:
                            entry_count += 1
                            person_info[person_id]['entered'] = True
                    elif x1 < previous_x and x1 <= W // 2:
                        direction = 'Going to Left'
                        if not person_info[person_id]['entered']:
                            exit_count += 1
                            person_info[person_id]['entered'] = True
                    else:
                        direction = 'Undetermined'

                    # Update the previous position
                    person_info[person_id]['previous_position'] = (x1, y1)

                    if not person_info[person_id]['entered']:
                        print(f'Person {person_id} is {direction}')
                        rects.append((x1, y1, x2, y2))

        # Display the frame with detections and counters in real-time
        cv2.putText(frame, f'Entry Count: {entry_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Exit Count: {exit_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Real-time Person Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
