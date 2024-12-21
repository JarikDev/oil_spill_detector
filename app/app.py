import base64
import io
import os
import os.path
import warnings
import yaml

import cv2
import numpy as np
from flask import Flask, Response, render_template, request
from PIL import Image
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import img_to_array

from flask_cors import CORS

warnings.filterwarnings('ignore')

ml_model = None


def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


config = load_config(r'D:\my_space\mifi_ml\hakaton1\oil_spill\app\config.yaml')
port = config['app']['port']
train_path = config['datasets']['train']
validation_path = config['datasets']['validation']
test_path = config['datasets']['test']


def base64_to_image(image_base64):
    img = Image.open(io.BytesIO(base64.decodebytes(bytes(image_base64, "utf-8"))))
    img = img.resize((150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def populate_data(data):
    features, labels = zip(*data)
    return np.array(features) / 255.0, np.array(labels)


def get_model():
    labels = ['Non Oil Spill', 'Oil Spill']
    img_size = 150

    def load_data(directory):
        data = []
        for label in labels:
            path = os.path.join(directory, label)
            class_num = labels.index(label)

            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                img_arr = cv2.imread(img_path)

                if img_arr is not None:
                    img_arr = cv2.resize(img_arr, (img_size, img_size))
                    data.append([img_arr, class_num])
        return np.array(data, dtype=object)

    train_data = load_data(train_path)
    test_data = load_data(test_path)
    validation_data = load_data(validation_path)
    print("Shape of training data:", train_data.shape)
    print("Shape of test data:", test_data.shape)
    print("Shape of valdata:", validation_data.shape)

    x_train, y_train = populate_data(train_data)
    x_val, y_val = populate_data(load_data(validation_path))
    x_test, y_test = populate_data(load_data(test_path))

    print("Shapes:")
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_val: {x_val.shape}, y_val: {y_val.shape}")
    print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3), padding="same"))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), shuffle=True, batch_size=64)

    return model


app = Flask(__name__)
CORS(app)


@app.route("/status", methods=['GET'])
def get_status():
    return "Oil Spill App operational !!!"


@app.route("/upload", methods=['GET'])
def get_upload_page():
    return render_template('index.html', server_port=port)


def get_response(message, status):
    return Response(message, status=status, mimetype='text/plain')


@app.route("/check", methods=['POST'])
def check_oil_spill():
    if ml_model is None:
        return get_response("ML Model is None", 500)

    content = request.json
    print(content)
    img = content['image']
    print(img)
    processed_image = base64_to_image(img)
    prediction = ml_model.predict(processed_image)
    predicted_class = (prediction > 0.5).astype("int32")
    print("Predicted class:", predicted_class[0][0])

    msg = "Oil spill" if predicted_class[0][0] == 1 else "Not oil spill"
    return Response(msg, status=200, mimetype='text/plain')


if __name__ == "__main__":
    # ml_model = get_model()
    app.run(host='localhost', port=port, debug=True, use_reloader=False)
