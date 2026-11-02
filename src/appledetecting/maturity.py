"""HSV spectral clustering and nearest-centroid assignment for new images."""
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import SpectralClustering

def pixels(image,box):
    w,h=image.size;x1,y1,x2,y2=map(float,box)
    x1=max(0,int(x1));y1=max(0,int(y1));x2=min(w,int(np.ceil(x2)));y2=min(h,int(np.ceil(y2)))
    if x2<=x1 or y2<=y1:raise ValueError('Empty apple crop')
    arr=np.asarray(image.convert('HSV').crop((x1,y1,x2,y2)))
    hh,ww=arr.shape[:2];yy,xx=np.ogrid[:hh,:ww]
    mask=((xx-(ww-1)/2)/max(ww/2,1))**2+((yy-(hh-1)/2)/max(hh/2,1))**2<=1
    return arr[mask]

def features(image,box,bins=16):
    p=pixels(image,box)
    return np.concatenate([np.histogram(p[:,i],bins=bins,range=(0,256))[0]/len(p) for i in (0,1)])

def fit(x,k=3):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or len(x)<k or k<2:raise ValueError('Not enough feature rows')
    labels=SpectralClustering(n_clusters=k,affinity='nearest_neighbors',n_neighbors=min(10,len(x)-1),random_state=42,assign_labels='kmeans').fit_predict(x)
    centers=np.stack([x[labels==i].mean(axis=0) for i in range(k)])
    return centers,labels

def assign(x,centers):
    x=np.atleast_2d(np.asarray(x,dtype=float));c=np.asarray(centers,dtype=float)
    if x.shape[1]!=c.shape[1]:raise ValueError('Feature dimensions do not match')
    return np.argmin(((x[:,None,:]-c[None,:,:])**2).sum(axis=2),axis=1)

def save(path,centers,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,centers=centers,metadata=json.dumps(metadata))

def load(path):
    with np.load(path,allow_pickle=False) as data:return data['centers'],json.loads(str(data['metadata']))
