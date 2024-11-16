from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.applications import EfficientNetB3, DenseNet121, InceptionV3
from tensorflow.keras import regularizers
import pandas as pd
import tensorflow as tf
import io

app = Flask(__name__)

# Load the metadata
test_csv_file_path = r'DATA\testing\testing.csv'  # Update this path
test_metadata = pd.read_csv(test_csv_file_path)

# Define the shared input layer
shared_input = Input(shape=(224, 224, 3))

# Function to build the model
def build_model(base_model, fine_tune_at=-1, l2_strength=0.01):
    base = base_model(weights='imagenet', include_top=False, input_tensor=shared_input)
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base.layers[fine_tune_at:]:
        layer.trainable = True
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_strength))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(test_metadata['Disease Name'].unique()), activation='softmax', kernel_regularizer=regularizers.l2(l2_strength))(x)
    model = Model(inputs=shared_input, outputs=outputs)
    return model

# Load the base models and weights
efficientnet_model = build_model(EfficientNetB3, fine_tune_at=-20)
densenet_model = build_model(DenseNet121, fine_tune_at=-15)
inceptionnet_model = build_model(InceptionV3, fine_tune_at=-10)

efficientnet_model.load_weights('weights/efficientnet_model_finetuned_weights.h5')
densenet_model.load_weights('weights/densenet_model_finetuned_weights.h5')
inceptionnet_model.load_weights('weights/inceptionnet_model_finetuned_weights.h5')

# Build the ensemble model
def build_ensemble(models, shared_input):
    outputs = [model.output for model in models]
    x = tf.keras.layers.Average()(outputs)
    ensemble_model = Model(inputs=shared_input, outputs=x)
    return ensemble_model

ensemble_model = build_ensemble([efficientnet_model, densenet_model, inceptionnet_model], shared_input)
ensemble_model.load_weights('weights/ensemble_model_finetuned_weights.h5')

# Function to get disease info
def get_disease_info(metadata, disease_class):
    info = metadata[metadata['Disease Name'] == disease_class]
    description = info['Description'].values[0]
    self_care = info['Self care treatment'].values[0]
    return description, self_care

# Function to classify and get info
def classify_and_get_info(image, model, metadata):
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction)
    predicted_class = list(metadata['Disease Name'].unique())[predicted_class_idx]

    description, self_care = get_disease_info(metadata, predicted_class)
    return predicted_class, description, self_care

from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.applications import EfficientNetB3, DenseNet121, InceptionV3
from tensorflow.keras import regularizers
import pandas as pd
import tensorflow as tf
import io

from flask_cors import CORS
from werkzeug.utils import secure_filename  # Import CORS

app = Flask(__name__)
CORS(app)  # Apply CORS to your Flask app to handle cross-origin requests

# Load the metadata
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
test_csv_file_path = os.path.join(dir_path, 'weights', 'testing.csv')
test_metadata = pd.read_csv(test_csv_file_path)

# Define the shared input layer
shared_input = Input(shape=(224, 224, 3))

# Function to build the model
def build_model(base_model, fine_tune_at=-1, l2_strength=0.01):
    base = base_model(weights='imagenet', include_top=False, input_tensor=shared_input)
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base.layers[fine_tune_at:]:
        layer.trainable = True
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_strength))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(test_metadata['Disease Name'].unique()), activation='softmax', kernel_regularizer=regularizers.l2(l2_strength))(x)
    model = Model(inputs=shared_input, outputs=outputs)
    return model

# Load the base models and weights
efficientnet_model = build_model(EfficientNetB3, fine_tune_at=-20)
densenet_model = build_model(DenseNet121, fine_tune_at=-15)
inceptionnet_model = build_model(InceptionV3, fine_tune_at=-10)

efficientnet_model.load_weights('weights/efficientnet_model_finetuned_weights.h5')
densenet_model.load_weights('weights/densenet_model_finetuned_weights.h5')
inceptionnet_model.load_weights('weights/inceptionnet_model_finetuned_weights.h5')

# Build the ensemble model
def build_ensemble(models, shared_input):
    outputs = [model.output for model in models]
    x = tf.keras.layers.Average()(outputs)
    ensemble_model = Model(inputs=shared_input, outputs=x)
    return ensemble_model

ensemble_model = build_ensemble([efficientnet_model, densenet_model, inceptionnet_model], shared_input)
ensemble_model.load_weights('weights/ensemble_model_finetuned_weights.h5')

# Function to get disease info
def get_disease_info(metadata, disease_class):
    info = metadata[metadata['Disease Name'] == disease_class]
    description = info['Description'].values[0]
    self_care = info['Self care treatment'].values[0]
    return description, self_care

# Function to classify and get info
def classify_and_get_info(image, model, metadata):
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction)
    predicted_class = list(metadata['Disease Name'].unique())[predicted_class_idx]

    description, self_care = get_disease_info(metadata, predicted_class)
    return predicted_class, description, self_care

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = load_img(io.BytesIO(file.read()), target_size=(224, 224))
    predicted_class, description, self_care = classify_and_get_info(image, ensemble_model, test_metadata)
    return jsonify({
        "predictedClass": predicted_class,
        "description": description,
        "selfCare": self_care
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
