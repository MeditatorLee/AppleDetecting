"""Clean overlays and CSV serialization."""
import csv
import io
from PIL import ImageDraw

def draw(image,detections,names=None,thickness=2):
    result=image.convert('RGB').copy();pen=ImageDraw.Draw(result)
    if names is None:names=['Apple']*len(detections)
    if len(names)!=len(detections):raise ValueError('Label count mismatch')
    for det,name in zip(detections,names):
        box=tuple(map(int,det['xyxy']));pen.rectangle(box,outline='#28d4a2',width=thickness)
        pen.text((box[0],max(0,box[1]-14)),str(name),fill='white',stroke_width=1,stroke_fill='black')
    return result

def detections_csv(detections,names):
    buffer=io.StringIO();writer=csv.writer(buffer)
    writer.writerow(['apple_id','class_id','confidence','xmin','ymin','xmax','ymax','center_x','center_y_bottom_left','maturity'])
    for i,(d,name) in enumerate(zip(detections,names),1):writer.writerow([i,d['class_id'],d['confidence'],*d['xyxy'],*d['center_bottom_left'],name])
    return buffer.getvalue()
