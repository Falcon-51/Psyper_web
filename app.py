from flask import Flask, flash, request, redirect, url_for, render_template
from PyQt5.QtGui import QImage
import os,shutil
from werkzeug.utils import secure_filename
from PIL import Image
from ultralytics import YOLO
import numpy as np
import folium
from folium.plugins import HeatMap
import pickle
import random

app = Flask(__name__)
 

UPLOAD_FOLDER = 'static/uploads/'
detection_count = [0,0,0,0]
map_obj = folium.Map(location = [61.919186, 34.065328], zoom_start = 13)


for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
 

def generate_coord(detections):
    nl=(random.uniform(61902, 61922))/1000
    el=(random.uniform(34040, 34064))/1000
    intensity = 0
    berry_count = sum(detections.values())
    if berry_count > 10:
        intensity = 1
    else:
        intensity = berry_count/10
    coord=[nl,el,intensity]
    return coord


def updateHitmap(coord):
    data = None
    with open('heatmap.pkl', 'rb') as f:
        data = pickle.load(f)
    data.append(coord)
    with open('heatmap.pkl', 'wb') as f:
        pickle.dump(data, f)
    print('APPENDED: ', coord)



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 


def updateDetectionsLabel(detections):
        berries_dict = {0: "Raspberry", 1: "Blueberry", 2: "Cloudberry", 3: "Strawberry"}
        detections_text = "Detections:\n"
        detection_count = [0,0,0,0]
        for class_id, count in detections.items():
            detection_count[int(class_id)] += count
        for class_id, count in enumerate(detection_count):
            if count == 0:
                continue
            detections_text += f"{berries_dict[class_id]} - {detection_count[int(class_id)]};\n"
        return detections_text



@app.route('/')
def home():
    return render_template('index.html')



@app.route('/heat_map')
def openHeatmap():
    coords = None
    with open('heatmap.pkl', 'rb') as f:
        coords = pickle.load(f)
    HeatMap(coords).add_to(map_obj)
    map_obj.save("templates/heat_map.html")
    return render_template('heat_map.html')



@app.errorhandler(413)
def too_large(e):
    flash('File is too large...')
    return redirect(request.url)



@app.route('/', methods=['POST'])
def upload_image():
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = Image.open(UPLOAD_FOLDER+filename) # Open image
        img = img.resize((520, 360)) # Resize image
        img.save(UPLOAD_FOLDER+filename) # Save resized image

        model = YOLO('weights/base-m.pt')
        img = UPLOAD_FOLDER + filename
        results = model.predict([img], conf=0.5)
        unique_elements, counts = np.unique(results[0].boxes.cls, return_counts=True)
        detections = dict(zip(unique_elements, counts))
        im_np = results[0].plot(conf=True, labels=True)
        height, width, channels = im_np.shape
        bytes_per_line = channels * width
        file2 = QImage(im_np.data, width, height, bytes_per_line, QImage.Format_BGR888)
        filename2 = 'PROC' + filename
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
        coord = generate_coord(detections)
        updateHitmap(coord)
        DTK = updateDetectionsLabel(detections)
        filename = [filename,filename2]
        return render_template('index.html', filename=filename, DTK=DTK)
    else:
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)
 


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 


if __name__ == "__main__":
    app.run()

    