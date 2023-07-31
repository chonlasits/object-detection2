from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import numpy as np
import os
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import shutil

app = Flask(__name__)

model = YOLO('yolov8n.pt', 'v8')

Format_image = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in Format_image

def read_img(filename):
    img = cv2.imread(filename)
    return img

def detect_and_draw_box(img_filepath, model=model, confidence=0.5):
    img = cv2.imread(img_filepath)
    bbox, label, conf = cv.detect_common_objects(img,confidence=0.5, model=model)
    output_image = draw_bbox(img, bbox, label, conf)
    output_image_path = os.path.join('static', 'output_image.jpg')
    cv2.imwrite(output_image_path, output_image)
    return output_image_path

def detect_boundingbox(img_filepath, model=model, confidence=0.5):
    img = cv2.imread(img_filepath)
    bbox, label, conf = cv.detect_common_objects(img,confidence=0.5, model=model)
    return bbox, label, conf

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/result',methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        img_file = request.files['image']
        if img_file and allowed_file(img_file.filename):
            filename = img_file.filename
            file_path = os.path.join('static', filename)
            img_file.save(file_path)
            img = read_img(file_path)
            output_image_path= detect_and_draw_box(file_path)
            return render_template('result.html', image=output_image_path,)        
    else:
         return "Unable to read the file. Please check file extension"

@app.route('/jsonform', methods=['GET', 'POST'])
def Image_to_Json():
    if request.method == 'POST':
        img_file = request.files['image2']
        if img_file and allowed_file(img_file.filename):
            filename = img_file.filename
            file_path = os.path.join('static', filename)
            img_file.save(file_path)
            bbox, label, conf = detect_boundingbox(file_path)
            sub_dict = {}
            for k in range(len(label)):
                sub_dict.setdefault('class'+str(k+1),label[k])
                sub_dict.setdefault('bboxes'+str(k+1),bbox[k])
                sub_dict.setdefault('confident'+str(k+1),conf[k])
            return jsonify(sub_dict)

if __name__=='__main__':
    app.run(debug=True)