#!/bin/bash
cd /home/ubuntu/yolov5-deepsort-web-main
source ./venv/bin/activate
cd stream
python3 manage.py runserver 0.0.0.0:8002
