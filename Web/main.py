import os
from flask import Flask, request, redirect, url_for, render_template, make_response
import numpy as np
import cv2
from Deepfake_Detection import *
import base64

app = Flask(__name__)

@app.route("/", methods=["GET"]) # with the below function, route ("") is the url of file 
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"]) # with the below function, route ("") is the url of file 
def upload():
    file = request.files["file"]
    filename = file.filename
    full_path = os.path.join("D:/Programming/Machine_Learning/Deep_Fake_Detection/Web/test", filename)
    file.save(full_path)
    test_dataGenerator = ImageDataGenerator(rescale=1./255)
    img = cv2.imread(full_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256)) / 255.
    img = np.asarray([img], dtype=np.float32)
    # Instantiating generator to feed images through the network
    # test_generator = test_dataGenerator.flow_from_directory("D:\\Programming\\Machine_Learning\\Deep_Fake_Detection\\Web\\test",target_size=(256, 256),
    # batch_size=1,class_mode='binary')
    # ret_str = "asdf"
    # print(len(test_generator))
    # for i in range(len(test_generator)):
    # # Rendering image X with label y for MesoNet
    #     X, y = test_generator[i] # reshape(1, 256, 256, 3) and rescaling
    # # Evaluating prediction
    

    with open(full_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        req_file = encoded_image.decode('utf-8')
    ret_str = f"<br><img src='data:image/jpg;base64, {req_file}'><br>"   

    ret_str += f"Predicted likelihood: {meso.predict(img)[0][0]:.4f}<br>"
   
    if round(meso.predict(img)[0][0])==1:
        ret_str += "Real"
    else: 
        ret_str += "DeepFake"

    os.remove(full_path)
    
    return ret_str


app.run()