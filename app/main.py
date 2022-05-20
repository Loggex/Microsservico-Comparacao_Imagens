# uvicorn app.main:app --reload
# https://pt.linkedin.com/pulse/python-microservice-alex-jos%C3%A9-silva-msc

import base64
from fastapi import FastAPI, File, Form, UploadFile
from matplotlib.font_manager import json_dump
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
from scipy.spatial import distance
import shutil
from pathlib import Path
import sys
import os
import json

app = FastAPI()

UPLOAD_FOLDER = '/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


@app.get("/")
async def index():
    return {"Hello": "Woliugiugyiugld"}


@app.post("/comparar/")
async def compararImagens(imagemBase: UploadFile = File(...), imagemNova: UploadFile = File(...)):
    model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"

    IMAGE_SHAPE = (224, 224)

    layer = hub.KerasLayer(model_url)
    model = tf.keras.Sequential([layer])

    def extract(file):
        file = Image.open(file).convert('L').resize(IMAGE_SHAPE)
        # display(file)

        file = np.stack((file,)*3, axis=-1)

        file = np.array(file)/255.0

        embedding = model.predict(file[np.newaxis, ...])
        # print(embedding)
        vgg16_feature_np = np.array(embedding)
        flattended_feature = vgg16_feature_np.flatten()

        # print(len(flattended_feature))
        print(flattended_feature)
        # print('-----------')
        return flattended_feature

    # path1: Path = Path("images/imagemBase.jpg")
    # path2: Path = Path("images/imagemNova.jpg")

    # try:
    #     with path1.open("wb") as buffer:
    #         shutil.copyfileobj(imagemBase.file, buffer)
    # finally:
    #     imagemBase.file.close()

    # try:
    #     with path2.open("wb") as buffer:
    #         shutil.copyfileobj(imagemNova.file, buffer)
    # finally:
    #     imagemNova.file.close()

    # contents1 = await imagemBase.read()
    # with open(imagemBase.filename, 'wb') as a:
    #     a.write(contents1)

    # contents2 = await imagemNova.read()
    # with open(imagemNova.filename, 'wb') as b:
    #     b.write(contents2)

    with open(f'{imagemBase.filename}', 'wb') as buffer:
        shutil.copyfileobj(imagemBase.file, buffer)

    with open(f'{imagemNova.filename}', 'wb') as buffer:
        shutil.copyfileobj(imagemNova.file, buffer)

    scriptDir = os.path.dirname(__file__)
    img1 = extract(os.path.join(scriptDir, f'../{imagemBase.filename}'))
    img2 = extract(os.path.join(scriptDir, f'../{imagemNova.filename}'))

    metric = 'cosine'

    dc = distance.cdist([img1], [img2], metric)[0]
    return json.dumps(dc)
