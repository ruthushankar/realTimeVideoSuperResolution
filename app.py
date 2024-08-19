from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
from model import Model
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
model = Model('./models/super-resolution-10.onnx')

@app.route('/<path:path>')
def send_js(path):
    return send_from_directory('static', path)

@app.route('/')
def index():
    return render_template('index.html')

def decode_image(base64_data):
    header, encoded = base64_data.split(",", 1)
    data = base64.b64decode(encoded)
    image = Image.open(BytesIO(data))
    return image

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

@socketio.on('send_frame')
def handle_frame(data):
    frame_data = data['frame']
    image = decode_image(frame_data)
    processed_image = model.process_image(image)
    image_base64 = image_to_base64(processed_image)
    socketio.emit('frame_response', {'image': image_base64})


if __name__ == '__main__':
    socketio.run(app, debug=True)