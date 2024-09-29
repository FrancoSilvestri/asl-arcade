
import cv2
import requests
import base64

# URL de tu API Flask
url = 'http://10.108.151.62:5000'

# Inicializa la captura de video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

# Target string to send
target = input("Please provide a target letter: ")
while True:
    # Captura frame a frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: No se puede recibir el frame.")
        break

    
    # Encode frame as JPEG byte string
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()

    # Encode the byte string to base64 for JSON
    encoded_frame = base64.b64encode(frame_bytes).decode('utf-8')

    # Create JSON data
    data = {
        'frame': encoded_frame,
        'target': target
    }

    # Send POST request to the API
    response = requests.post(url, json=data)
    json_response = response.json()

    if json_response.get("target_detected"):
        print("Detected letter: ", target)
    # Check response
    if response.status_code == 200:
        pass
    else:
        print("Error:", response.status_code, response.text)


# Release the webcam
cap.release()
cv2.destroyAllWindows()
   
