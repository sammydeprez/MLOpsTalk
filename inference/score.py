import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import json
from PIL import Image
import requests
from io import BytesIO
import time
import datetime

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'pytorch','model.pth')
    labels_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'pytorch','labels.txt')
    
    print('Loading model...', end='')
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.eval()
    
    print('Loading labels...', end='')
    with open(labels_path, 'rt') as lf:
        global labels
        labels = [l.strip() for l in lf.readlines()]
    print(len(labels), 'found. Success!')

def run(input_data):
    url = json.loads(input_data)['image']
    prev_time = time.time()

    if(url.startswith('http')):
        response = requests.get(url)
        input_image = Image.open(BytesIO(response.content))
    else:
        input_image = Image.open(url)

    preprocess = transforms.Compose([
        transforms.Resize(225),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    index = output.data.cpu().numpy().argmax()
    probability = torch.nn.functional.softmax(output[0], dim=0).data.cpu().numpy().max()

    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)

    predictions = {}
    predictions[labels[index]] = str(round(probability*100,2))

    result = {
        'time': str(inference_time.total_seconds()),
        'prediction': labels[index], 
        'scores': predictions
    }

    return result