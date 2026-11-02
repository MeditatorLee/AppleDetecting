"""Optional ZoeDepth adapter; first run may download upstream model weights."""
from pathlib import Path
import numpy as np

def load_model(repo,device='cuda'):
    import torch
    repo=Path(repo).resolve()
    if not (repo/'hubconf.py').is_file():raise FileNotFoundError(f'Missing ZoeDepth repository: {repo}')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    model=torch.hub.load(str(repo),'ZoeD_NK',source='local',pretrained=True,trust_repo=True)
    return model.to(device).eval()

def predict(model,image):
    arr=np.asarray(model.infer_pil(image.convert('RGB')),dtype=np.float32)
    if arr.shape!=(image.height,image.width):raise ValueError('Unexpected depth map shape')
    if not np.any(np.isfinite(arr)&(arr>0)):raise ValueError('No positive finite depth values')
    return arr

def apple_depths(depth,detections):
    h,w=depth.shape;out=[]
    for i,d in enumerate(detections,1):
        x1,y1,x2,y2=map(float,d['xyxy'])
        x1,x2=sorted((max(0,min(w,x1)),max(0,min(w,x2))))
        y1,y2=sorted((max(0,min(h,y1)),max(0,min(h,y2))))
        l,t,r,b=int(np.floor(x1)),int(np.floor(y1)),int(np.ceil(x2)),int(np.ceil(y2))
        if r<=l or b<=t:out.append({'apple_id':i,'depth':None,'valid_pixels':0});continue
        yy,xx=np.mgrid[t:b,l:r];cx,cy=(x1+x2)/2,(y1+y2)/2
        mask=((xx+.5-cx)/max((x2-x1)*.35,.5))**2+((yy+.5-cy)/max((y2-y1)*.35,.5))**2<=1
        patch=depth[t:b,l:r];v=patch[mask&np.isfinite(patch)&(patch>0)]
        out.append({'apple_id':i,'depth':float(np.median(v)) if len(v) else None,'valid_pixels':int(len(v))})
    return out

def heatmap(depth):
    from PIL import Image
    from matplotlib import colormaps
    finite=depth[np.isfinite(depth)&(depth>0)]
    if not len(finite):raise ValueError('Empty depth')
    low,high=np.percentile(finite,[2,98]);v=np.clip((np.nan_to_num(depth,nan=low)-low)/max(high-low,1e-6),0,1)
    return Image.fromarray((colormaps['viridis'](v)[...,:3]*255).astype('uint8'))
