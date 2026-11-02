import numpy as np
import pytest
from PIL import Image
from appledetecting.geometry import center_bottom_left
from appledetecting.detection import parse_labels
from appledetecting.maturity import features,fit,assign,save,load
from appledetecting.depth import apple_depths,heatmap
from appledetecting.mass import calculate,Parameters
from appledetecting.hsv_report import summarize
from appledetecting.visualization import draw,detections_csv

def test_geometry():assert center_bottom_left([10,20,30,60],100)==(20,60)
def test_parse(tmp_path):
    p=tmp_path/'input.txt';p.write_text('0 0.5 0.5 0.2 0.4 0.9\n')
    d=parse_labels(p,(100,200));assert len(d)==1;assert np.allclose(d[0]['xyxy'],[40,60,60,140])
def test_maturity(tmp_path):
    image=Image.new('RGB',(40,40),'red');assert features(image,[0,0,40,40]).shape==(32,)
    rng=np.random.default_rng(42);x=rng.normal(0, .2, (36,32))
    centers,labels=fit(x);assert centers.shape==(3,32)
    p=tmp_path/'ref.npz';save(p,centers,{'n':36});c,metadata=load(p)
    assert metadata['n']==36 and assign(x[:3],c).shape==(3,)
def test_depth():
    arr=np.full((30,30),2.5,dtype=np.float32)
    assert apple_depths(arr,[{'xyxy':[5,5,25,25]}])[0]['depth']==2.5
    assert heatmap(arr).size==(30,30)
def test_mass():
    d=[{'xyxy':[0,0,20,30]}];depth=[{'depth':2.0}]
    assert calculate(d,depth,['Fully Mature'])[0]['mass_proxy']>0
    assert calculate(d,depth,['Raw'])[0]['density_assumption']==.8
    assert calculate(d,[{'depth':None}],['Raw'])[0]['mass_proxy'] is None
    with pytest.raises(ValueError):calculate([],[],[],Parameters(scale=0))
def test_hsv_summary():
    p=[np.tile(np.array([[0,200,200]],dtype=np.uint8),(10,1)),np.tile(np.array([[85,100,100]],dtype=np.uint8),(10,1)),np.tile(np.array([[170,150,150]],dtype=np.uint8),(10,1))]
    s,d=summarize(p,[0,1,2]);assert len(s)==3 and len(d)==324
    assert np.allclose(d.groupby(['cluster','channel']).fraction.sum(),1)
def test_overlay():
    im=Image.new('RGB',(100,100));d=[{'class_id':0,'confidence':.9,'xyxy':[5,5,30,30],'center_bottom_left':(17.5,82.5)}]
    assert draw(im,d,['Raw']).size==(100,100)
    assert 'Raw' in detections_csv(d,['Raw'])
