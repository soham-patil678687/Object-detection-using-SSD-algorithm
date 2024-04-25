from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import io

app = Flask(__name__)

classNames = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
              5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
              10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
              14: 'motorbike', 15: 'person', 16: 'pottedplant',
              17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

proto = "MobileNetSSD_deploy.prototxt"
weights = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto, weights)

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Object Detection</title>
    </head>
    <body>
        <h1>Object Detection</h1>
        <form method="post" enctype="multipart/form-data" id="uploadForm">
            <input type="file" name="image">
            <button type="submit">Detect Objects</button>
        </form>
        <div id="result"></div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const formData = new FormData(this);

                try {
                    const response = await fetch('/detect_objects', {
                        method: 'POST',
                        body: formData
                    });
                    const imageData = await response.blob();
                    const url = URL.createObjectURL(imageData);
                    document.getElementById('result').innerHTML = `<img src="${url}" alt="Detected Objects">`;
                } catch (error) {
                    console.error('Error detecting objects:', error);
                }
            });
        </script>
    </body>
    </html>
    """

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    image_file = request.files['image']
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            class_name = classNames[class_id]
            box = detections[0, 0, i, 3:7] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            x1, y1, x2, y2 = box.astype('int')
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert modified image to bytes and return
    ret, buffer = cv2.imencode('.jpg', img)
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
