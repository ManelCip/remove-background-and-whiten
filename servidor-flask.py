from flask import Flask, request, send_file
import cv2
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

def remove_background_and_whiten(image_path, sensitivity=30):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 4:
        rgb = image[:, :, :3]
    else:
        rgb = image

    Z = rgb.reshape((-1, 3))
    Z = np.float32(Z)
    _, labels, centers = cv2.kmeans(Z, 2, None,
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                     10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    bg_label = np.argmax(np.bincount(labels.flatten()))
    bg_color = centers[bg_label]
    diff = cv2.absdiff(rgb, bg_color)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, sensitivity, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 5)

    white = np.ones_like(rgb) * 255
    rgba = cv2.merge((white[:, :, 0], white[:, :, 1], white[:, :, 2], mask))

    result = Image.fromarray(rgba)
    img_io = io.BytesIO()
    result.save(img_io, format='PNG')
    img_io.seek(0)
    return img_io

@app.route('/process', methods=['POST'])
def process():
    file = request.files.get('image')
    if not file:
        return "No image provided", 400

    temp_path = "temp.png"
    file.save(temp_path)
    result = remove_background_and_whiten(temp_path, sensitivity=40)
    os.remove(temp_path)
    return send_file(result, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
