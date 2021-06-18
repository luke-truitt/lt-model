import io
import json
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request


app = Flask(__name__)

def load_model():
    return None
    
model = load_model()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([ # add correct parameters for what you built your model as
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    ## Load tensor of image

    ## Predict with Model

    ## Take outputs and map to class

    return None

@app.route('/predict', methods=['POST'])
def predict():
    ## If its a post, take the file string, load it, and predict the class, then return the class
    
    return jsonify({})

if __name__ == '__main__':
    app.run()