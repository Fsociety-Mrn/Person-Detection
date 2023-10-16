import cv2
import numpy as np
import imutils
from tracker.centroidtracker import CentroidTracker

# Load the object detection model
prototxt = "detector/MobileNetSSD_deploy.prototxt"
model = "detector/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Initialize variables
totalFrames = 0

# Initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# Initialize the centroid tracker
ct = CentroidTracker()

# Initialize the video stream (use your video source here)
cap = cv2.VideoCapture(0)

# Define class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

# Loop over frames from the video stream
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    # Resize the frame to have a maximum width of 500 pixels
    frame = imutils.resize(frame, width=500)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Check to see if we should run object detection
    if totalFrames % 30 == 0:
        # Initialize the list of people's centroids
        rects = []

        # Convert the frame to a blob and pass it through the network
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

# Inside the loop where you process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.8:
                class_id = int(detections[0, 0, i, 1])

                if CLASSES[class_id] == "person":
                    
                    # Extract the bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Draw bounding boxes
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            #        Compute the centroid of the bounding box
                    centroid_x = int((startX + endX) / 2)
                    centroid_y = int((startY + endY) / 2)

                    # Store the bounding box coordinates in 'rects'
                    rects.append((startX, startY, endX, endY))

        # # Update the centroid tracker with the new centroids
        # objects = ct.update(rects)

        # for (objectID, centroid) in objects.items():
        #     # Draw the ID and centroid on the frame
        #     text = "Person {}".format(objectID)
        #     cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #     cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # Show the frame
    cv2.imshow("People Counter", frame)

    # Exit the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close the window
cap.release()
cv2.destroyAllWindows()
