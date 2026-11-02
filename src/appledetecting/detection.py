"""Official YOLOv5 subprocess adapter, avoiding modification of third-party sources."""
import subprocess
import sys
import tempfile
from pathlib import Path
from .geometry import center_bottom_left

def parse_labels(path, size):
    width,height=size
    if not path.is_file():return []
    result=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        fields=line.split()
        if len(fields)!=6:raise ValueError(f'Invalid YOLOv5 prediction row: {line}')
        cls=int(fields[0]);xc,yc,bw,bh,confidence=map(float,fields[1:])
        box=[max(0,(xc-bw/2)*width),max(0,(yc-bh/2)*height),min(width,(xc+bw/2)*width),min(height,(yc+bh/2)*height)]
        if box[2]<=box[0] or box[3]<=box[1]:continue
        result.append({'class_id':cls,'confidence':confidence,'xyxy':box,'center_bottom_left':center_bottom_left(box,height)})
    return result

def predict(image, weights, repo, confidence=.25, device='0'):
    weights=Path(weights).resolve();repo=Path(repo).resolve()
    if not weights.is_file():raise FileNotFoundError(f'Missing checkpoint: {weights}')
    script=repo/'detect.py'
    if not script.is_file():raise FileNotFoundError(f'Missing YOLOv5 detect.py: {script}')
    with tempfile.TemporaryDirectory(prefix='appledetecting_') as folder:
        temp=Path(folder);source=temp/'input.png';image.convert('RGB').save(source)
        cmd=[sys.executable,str(script),'--weights',str(weights),'--source',str(source),'--imgsz','640','--conf-thres',str(confidence),'--device',device,'--save-txt','--save-conf','--nosave','--project',str(temp/'runs'),'--name','infer','--exist-ok']
        p=subprocess.run(cmd,cwd=repo,capture_output=True,text=True,timeout=300,check=False)
        if p.returncode:raise RuntimeError((p.stderr or p.stdout)[-4000:])
        return parse_labels(temp/'runs/infer/labels/input.txt',image.size)
