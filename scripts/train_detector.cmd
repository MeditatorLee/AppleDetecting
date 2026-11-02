@echo off
REM Run from repository root after installing upstream YOLOv5 and preparing dataset.
python external\yolov5\train.py --img 640 --batch 4 --epochs 60 --data datasets\apple_pseudo\data.yaml --weights yolov5s.pt --device 0 --workers 2 --project runs\train --name apple_pseudo_v1 --seed 42 --patience 15
