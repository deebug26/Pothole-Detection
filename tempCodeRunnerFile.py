from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load YOLO model
net = cv2.dnn.readNet("E:\Pothole_detection\Pothole_detection\yolov4-tiny.weights", 
                       "E:\Pothole_detection\Pothole_detection\yolov4-tiny.cfg")

with open("E:\Pothole_detection\Pothole_detection\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

pothole_info_list = []  # Store pothole detection details

def detect_potholes():
    global pothole_info_list
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []
        detected_potholes = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

                detected_potholes.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'confidence': round(confidences[i], 2)
                })

        pothole_info_list = detected_potholes  # Update global list

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')

    if email and password:
        session['user'] = email
        return redirect(url_for('camera'))
    return redirect(url_for('index'))

@app.route('/camera')
def camera():
    if 'user' in session:
        return render_template('camera.html')
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    if 'user' in session:
        return Response(detect_potholes(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return redirect(url_for('index'))

@app.route('/pothole_data')
def pothole_data():
    if 'user' in session:
        return render_template('pothole_data.html', potholes=pothole_info_list)
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
