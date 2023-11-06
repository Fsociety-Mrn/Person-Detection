from ultralytics import YOLO
import cv2
import torch 



# load yolov8 model
model = YOLO('yolov5s.pt')


cap = cv2.VideoCapture(0)

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:
        
        

        # detect objects
        # track objects
        results = model.track(cv2.resize(frame, (320,320), interpolation=cv2.INTER_AREA), persist=True)

        # plot results
        # cv2.rectangle
        # cv2.putText
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break