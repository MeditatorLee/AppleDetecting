"""Historical competition-inspired uncalibrated mass proxy; output is NOT grams."""
from dataclasses import dataclass
import math
DENSITY={'Raw':.80,'Fully Mature':.88,'Half Mature':.84}
@dataclass(frozen=True)
class Parameters:
    scale:float=1.0
    a:float=.0002
    b:float=.0249

def calculate(detections,depths,names,params=Parameters()):
    if not(len(detections)==len(depths)==len(names)):raise ValueError('Mismatched inputs')
    if not math.isfinite(params.scale) or params.scale<=0:raise ValueError('Invalid scale')
    out=[]
    for i,(d,dep,name) in enumerate(zip(detections,depths,names),1):
        x1,y1,x2,y2=map(float,d['xyxy']);w=x2-x1;h=y2-y1;z=dep.get('depth')
        row={'apple_id':i,'maturity':name,'depth_estimate':z,'width_px':w,'height_px':h,'shape_index':h/w if w>0 else None,'diameter_proxy':None,'volume_proxy':None,'density_assumption':DENSITY.get(name),'mass_proxy':None,'status':'unavailable'}
        if w>0 and h>0 and z is not None and math.isfinite(float(z)) and z>0 and name in DENSITY:
            diameter=((w+h)/2 if .5<=h/w<=1.5 else max(w,h))/float(z)*params.scale
            volume=params.a*diameter**3+params.b*diameter**2
            row.update(diameter_proxy=diameter,volume_proxy=volume,mass_proxy=volume*DENSITY[name],status='uncalibrated proxy')
        out.append(row)
    return out
