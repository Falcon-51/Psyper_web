# Psyper_web(🥇 Хакатон IT-КЕМП 3.0)

## Задачи проекта:
Разработать сервис для определения урожайности плодов на основе компьютерного зрения. На предоставленных изображениях детектировать и классифицировать ягоды, определить количество каждого вида. Также предусмотреть возможность построения тепловой карты по общему количеству детектированных ягод.

## Описание проекта:  
- В качестве детектора и классификатора использовалась модель YOLOv8s обученная на размеченных данных.
- Для построения тепловой карты на реальной географической карте использовалась библиотека Folium.
- В качестве инференcа всей системы использовалось самописное web-приложение на Flask, а также вариант на PyQt.

## Инструменты:
* Python
* Folium
* YoloV8
* CocoAnnotator
* Flask
* GIT
* Docker

## Классы детектируемых объектов.

| class                | view |
|----------------------|------|
| raspberry            |![raspberry96](https://github.com/user-attachments/assets/b0c35674-fef6-472c-96c5-fff92ec57cf6)|
| blueberry            |![blueberry34](https://github.com/user-attachments/assets/8cb12157-5d23-4ece-a2de-7cc40fa01d1b)|
| cloudberry           |![cloudberry117](https://github.com/user-attachments/assets/5dee972a-3849-437d-8af2-0f469f079032)|
| strawberry           |![strawberry170](https://github.com/user-attachments/assets/eb9f6090-5cc8-4898-9dae-fb094bda8a09)|



## Результаты работы
### Детектирование (Инференс на Flask)
![image](https://github.com/Falcon-51/Psyper_web/assets/92328230/123afd85-23d9-4e6a-b2ff-821f9cee4ce0)
### Детектирование (Инференс на PyQt)
![image](https://github.com/user-attachments/assets/77f718d0-f3ff-465d-baa7-3ead4ccd2e63)
### Heatmap
![image](https://github.com/Falcon-51/Psyper_web/assets/92328230/81728367-61da-4a1c-a1c7-01a67b80722c)

