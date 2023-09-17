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
 

UPLOAD_FOLDER = 'static/uploads/'   #Папка хранения загруженных фото
detection_count = [0,0,0,0]         #Переменная хранящая обнаруженные объекты
map_obj = folium.Map(location = [61.919186, 34.065328], zoom_start = 13)    #Задание начальных координат и зума для карты 


#Безопасная очистка директории хранящей загруженные фото
for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


#Конфигурация приложения
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER         #Указание папки хранения фото
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  #Указание максимального размера фото
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])    #Указание допустимых типов файлов
 


#Фунция генерации координат для тепловой карты
def generate_coord(detections):

    #Генерирование координат
    nl=(random.uniform(61902, 61922))/1000
    el=(random.uniform(34040, 34064))/1000
    intensity = 0
    berry_count = sum(detections.values())

    #Задание интенсивности
    if berry_count > 10:
        intensity = 1
    else:
        intensity = berry_count/10
    #Сохранение в список
    coord=[nl,el,intensity]
    return coord


#Функция обновления тепловой карты
def updateHitmap(coord):
    data = None

    #Чтение конфигурационного файла
    with open('heatmap.pkl', 'rb') as f:
        data = pickle.load(f)

    #Загрузка новых координат
    data.append(coord)

    #Сохранение в конфигурационный файл
    with open('heatmap.pkl', 'wb') as f:
        pickle.dump(data, f)
    print('APPENDED: ', coord)


#Функция проверки расширения загружаемого файла
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 

#Обновления списка обнаружений на фото
def updateDetectionsLabel(detections):
        berries_dict = {0: "Raspberry", 1: "Blueberry", 2: "Cloudberry", 3: "Strawberry"}
        detections_text = "Detections:\n"
        detection_count = [0,0,0,0]

        #Занесение элементов в список с соответствующим индексом
        for class_id, count in detections.items():
            detection_count[int(class_id)] += count

        #Занесение в строку результата
        for class_id, count in enumerate(detection_count):
            if count == 0:
                continue
            detections_text += f"{berries_dict[class_id]} - {detection_count[int(class_id)]};\n"
        return detections_text



#Открытие начальной страницы
@app.route('/')
def home():
    return render_template('index.html')



#Открытие страницы с тепловой картой
@app.route('/heat_map')
def openHeatmap():
    coords = None

    #Загрузка данных с конфигурационного файла
    with open('heatmap.pkl', 'rb') as f:
        coords = pickle.load(f)

    #Загрузка данных в карту
    HeatMap(coords).add_to(map_obj)

    #Сохранение и открытие страницы
    map_obj.save("templates/heat_map.html")
    return render_template('heat_map.html')



#Вывод сообщения об ошибке (размер файла) на страницу
#Обработка ошибки 413
@app.errorhandler(413)
def too_large(e):
    flash('File is too large...')
    return redirect(request.url)


#Обработка изображения после загрузки на сервер методом POST
@app.route('/', methods=['POST'])
def upload_image():
    
    #Проверка на наличие файла в запросе
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    #Проверка на корректное имя файла
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    
    #Проверка расширения файла
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        #Сохранение файла на сервере
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        #Изменение размеров файла
        img = Image.open(UPLOAD_FOLDER+filename) 
        img = img.resize((440, 300)) 
        img.save(UPLOAD_FOLDER+filename) 

        #Указание весов для модели
        model = YOLO('weights/base-m.pt')
        #Указание пути файла           
        img = UPLOAD_FOLDER + filename
        #Загрузка файла в модель              
        results = model.predict([img], conf=0.5)
        unique_elements, counts = np.unique(results[0].boxes.cls, return_counts=True)
        #Запись результата обработки
        detections = dict(zip(unique_elements, counts))
        #Запись обработанного фото в переменную и сохранение её параметорв
        im_np = results[0].plot(conf=True, labels=True)
        height, width, channels = im_np.shape
        bytes_per_line = channels * width
        #Цветовая обработка с использованием ранее сохранённых параметров
        file2 = QImage(im_np.data, width, height, bytes_per_line, QImage.Format_BGR888)
        filename2 = 'PROC' + filename
        #Сохранение обработанного фото
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
        #Генерация координат и обновление тепловой карты
        coord = generate_coord(detections)
        updateHitmap(coord)
        #Переменная для вывода результата обнаружений на страницу
        DTK = updateDetectionsLabel(detections)
        filename = [filename,filename2]
        return render_template('index.html', filename=filename, DTK=DTK)
    else:
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)
 

#Передача адреса изображений
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 

#Запуск в режиме debug с указанием host`а
if __name__ == "__main__":
    app.run('172.20.10.2',debug=True)

    