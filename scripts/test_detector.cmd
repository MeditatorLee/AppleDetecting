@echo off
python external\yolov5\val.py --weights runs\train\apple_pseudo_v1\weights\best.pt --data datasets\apple_pseudo\data.yaml --task test --img 640 --device 0
