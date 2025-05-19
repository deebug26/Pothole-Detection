import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet(r"E:\Pothole_detection\Pothole_detection\yolov4-tiny.weights",
                      r"E:\Pothole_detection\Pothole_detection\yolov4-tiny.cfg")


# Load the classes
with open(r"E:\Pothole_detection\Pothole_detection\coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for front camera

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Prepare the image for the neural network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


    # Run forward pass to get predictions
    outputs = net.forward(output_layers)

    # Process the predictions
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

    # Display the output frame
    cv2.imshow("Pothole Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()


