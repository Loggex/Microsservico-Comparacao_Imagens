# uvicorn app.main:app --reload
# https://pt.linkedin.com/pulse/python-microservice-alex-jos%C3%A9-silva-msc

from ast import Return
from PIL import Image
from distutils.command.upload import upload
from fastapi import FastAPI, File, Form, UploadFile
import numpy as np
import cv2
from matplotlib import pyplot as plt


app = FastAPI()


@app.get("/")
async def index():
    return {"Hello": "Woliugiugyiugld"}


@app.post("/comparar/")
async def compararImagens(imagemBase: UploadFile = File(...), imagemNova: UploadFile = File(...)):

    # img1 = cv2.imdecode(np.fromstring(imagemBase.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    # img2 = cv2.imdecode(np.fromstring(imagemNova.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    img1 = cv2.imread(Image.open(imagemBase))  # queryImage
    img2 = cv2.imread(Image.open(imagemNova))  # trainImage

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    des1 = np.float32(des1)
    des2 = np.float32(des2)

    matches = flann.knnMatch(des1, des2, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]

    good_points = []

    threshold = 0.7

    for i, (m, n) in enumerate(matches):
        if m.distance < threshold*n.distance:
            matchesMask[i] = [1, 0]
            good_points.append(m)

    percentage = 100 / 500

    return{"Resultado": str(len(good_points) * percentage) + '% de semelhança'}
