from flask import Flask, request, jsonify
import numpy as np
from main import yolo_prediction, model
import cv2
import base64


app = Flask(__name__)

@app.route('/', methods=["POST"])
def stream_frame():

    data = request.json
    
    frame_data = data.get("frame", None)
    target = data.get("target", None)
    print(data)
    if frame_data is None or target is None:
        return jsonify({"error":"No frame nor target provided."}), 400

    
    # Decode the base64-encoded frame to bytes
    frame_bytes = base64.b64decode(frame_data)
    img = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    h, w, c = frame.shape
    CONF_THRESHOLD = 0.7
    
    # Perform inference
    results = yolo_prediction(frame)
   
    if results is not None:

        # Parse results
        for result in results:
            
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls)
                conf = box.conf 
                label = model.names[cls]
                
                if conf >= CONF_THRESHOLD:
                    print("DETAILS: ",conf, label, target)
                    if str(label).upper() == target.upper():
                        return jsonify({'message': 'Frame received', 'width': w, 'height': h, "channels":c, "conf":round(float(conf), 2), "target_detected":True}), 200

    return jsonify({'message': 'Frame received', 'width': w, 'height': h, "channels":c, "conf":0, "target_detected":False}), 200

if __name__ == '__main__':
    app.run(host='10.108.151.62', port=5000, debug=True)



