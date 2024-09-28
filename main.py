from ultralytics import YOLO
import cv2
import os

weight_path = os.path.join("runs", "detect", "YoloMediumModelTest22", "weights")
model = YOLO(os.path.join(weight_path, "best.pt"))

def yolo_prediction(frame):
    
    return model(frame)
                   
def yolo_streamer():

    cap = cv2.VideoCapture(0)
    weight_path = os.path.join("runs", "detect", "YoloMediumModelTest22", "weights")
    model = YOLO(os.path.join(weight_path, "best.pt"))
    CONF_THRESHOLD = 0.65
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame)

        # Parse results
        for result in results:
            
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls)
                conf = box.conf
                
        
                if conf >= CONF_THRESHOLD:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    label = model.names[cls]
                    confidence = conf.item() * 100

                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.1f}%", 
                                (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                    # Here you can add logic to verify if the gesture is performed correctly
                    # For example, check if the same gesture is detected consistently
                    
                        

        # Display the frame
        cv2.imshow('Sign Language Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    yolo_streamer()