import cv2
import numpy as np
import imutils

def people_counter():
    # Specify the full paths to the deploy.prototxt and mobilenet_ssd.caffemodel files
    deploy_file = "detector/MobileNetSSD_deploy.prototxt"
    model_file = "detector/MobileNetSSD_deploy.caffemodel"

    # Load the pre-trained model
    net = cv2.dnn.readNetFromCaffe(deploy_file, model_file)

    # Open the default camera (camera index 0)
    cap = cv2.VideoCapture(0)

    # Initialize variables
    total_people = 0

    while True:
        
        # setup camera
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        if not ret:
            break

        # Resize the frame
        frame = imutils.resize(frame, width = 500)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]


        # Display the count on the frame
        cv2.putText(frame, f"Total People: {total_people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("People Counter", frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    people_counter()
