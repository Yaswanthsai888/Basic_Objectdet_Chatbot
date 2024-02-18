from flask import Flask, render_template, Response
import cv2
import torch
import requests

app = Flask(__name__)

# Load the YOLOv5 model
model_path = "path/to/yolov5s.pt"  # Replace with your YOLOv5 model path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Rasa server URL
rasa_url = "http://localhost:5005/webhooks/rest/webhook"

def send_message_to_rasa(message):
    response = requests.post(rasa_url, json={"message": message})
    return response.json()

@app.route('/')
def index():
    return render_template('index.html')

def object_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(frame_rgb)
        detected_objects = []

        for x_min, y_min, x_max, y_max, confidence, class_id in results.xyxy[0]:
            if confidence > 0.5:
                class_label = model.names[int(class_id)]
                detected_objects.append(f"{class_label}: {confidence:.2f}")

                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_label}: {confidence:.2f}', (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # Send detected_objects to Rasa API
        if detected_objects:
            message = f"Detected objects: {', '.join(detected_objects)}"
            response = send_message_to_rasa(message)
            chatbot_response = response[0]['text']
            print("Chatbot Response:", chatbot_response)

    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(object_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

