import io
import json
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request


app = Flask(__name__)
# imagenet_class_index = json.load(open('<PATH/TO/.json/FILE>/imagenet_class_index.json'))
model_ft_new = models.resnet18(pretrained=True)
num_ftrs = model_ft_new.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft_new.fc = nn.Linear(num_ftrs, 2)
model_ft_new.load_state_dict(torch.load("models/state_dict_model_new.pt"))
model_ft_new.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model_ft_new.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return y_hat, predicted_idx

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # file = request.files['file']
        file = open("real/train/dog/dog.0.jpg", "rb")
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        print(class_id, class_name)
        return jsonify({'class_id': "class_id", 'class_name': "class_name"})

if __name__ == '__main__':
    app.run()