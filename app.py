from flask import Flask , request , jsonify
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
import os

from io import BytesIO

app = Flask(__name__)

@app.route('/api', methods = ['POST'])
def predict_img():
    d={}
    json_data = request.get_json()

    # Reading the imgUrl argument
    imgUrl = json_data.get('imgUrl')
    response = requests.get(imgUrl)
    img_data = response.content

    interpreter = tf.lite.Interpreter(model_path="model/DIsh Identification.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = Image.open(BytesIO(img_data))
    image_array = np.array(image.resize((150, 150)), dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0

    interpreter.set_tensor(input_details[0]['index'], image_array)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    predicted_class_index = np.argmax(output)

    print("Predicted class index: ", predicted_class_index)
    print("Predicted class score: ", output[0][predicted_class_index] * 100, "%")

    with open('model/labelss.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    label_map = {i: label for i, label in enumerate(labels)}
    predicted_label = label_map[predicted_class_index]

    result = {}

    # print(predicted_label)

    result['breed'] = predicted_label
    result['score'] = output[0][predicted_class_index] * 100
    print("predicted dish : ", predicted_label)
    print("Predicted class score: ", output[0][predicted_class_index] * 100, "%")
    d['output'] = result

    return jsonify(d)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    
