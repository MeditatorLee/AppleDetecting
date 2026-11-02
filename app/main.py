"""Streamlit entrypoint for the AppleDetecting research demo."""
from pathlib import Path
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from appledetecting import detection, maturity, depth, mass, visualization

ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title='AppleDetecting',layout='wide')
st.title('AppleDetecting 🍎')
st.caption('APMCM reconstruction · YOLOv5s · HSV spectral clustering · ZoeDepth · geometric mass proxy')
st.warning('Research prototype. Detection was trained on unreviewed historical pseudo-labels; mass values are not calibrated grams.')
with st.sidebar:
    st.header('Models')
    weights=st.text_input('YOLOv5 checkpoint',str(ROOT/'runs/train/apple_pseudo_v1/weights/best.pt'))
    yolo=st.text_input('YOLOv5 checkout',str(ROOT/'external/yolov5'))
    ref=st.text_input('Maturity reference',str(ROOT/'models/maturity_reference.npz'))
    mapping=st.text_input('Maturity mapping',str(ROOT/'models/cluster_names.json'))
    zoe=st.text_input('ZoeDepth checkout',str(ROOT/'external/ZoeDepth'))
    device=st.selectbox('Inference device',['0','cpu'])
    confidence=st.slider('Detection confidence',.05,.95,.25,.05)
    thickness=st.slider('Bounding-box thickness',1,6,2)
    scale=st.number_input('Uncalibrated mass scaling ratio',min_value=.001,value=1.0,step=.1)
file=st.file_uploader('Upload an apple photograph',type=['jpg','jpeg','png'])
if file:
    raw=file.getvalue();image=Image.open(io.BytesIO(raw)).convert('RGB')
    fingerprint=__import__('hashlib').sha256(raw).hexdigest()
    if st.button('Detect apples',type='primary'):
        try:
            with st.spinner('Running YOLOv5...'):
                detections=detection.predict(image,weights,yolo,confidence,device)
            st.session_state['inference']={'fingerprint':fingerprint,'detections':detections}
            st.session_state.pop('depth_output',None)
        except Exception as exc:st.exception(exc)
    state=st.session_state.get('inference')
    if state and state['fingerprint']==fingerprint:
        dets=state['detections'];names=['Apple']*len(dets)
        if Path(ref).is_file() and dets:
            try:
                centers,_=maturity.load(ref)
                assigned=maturity.assign(np.stack([maturity.features(image,d['xyxy']) for d in dets]),centers)
                name_map={str(i):f'Cluster {i}' for i in range(len(centers))}
                if Path(mapping).is_file():name_map.update(json.loads(Path(mapping).read_text(encoding='utf-8')))
                names=[name_map[str(int(k))] for k in assigned]
            except Exception as exc:st.error(f'Maturity unavailable: {exc}')
        annotated=visualization.draw(image,dets,names,thickness)
        c1,c2=st.columns(2)
        with c1:st.image(image,caption='Original image',width='stretch')
        with c2:st.image(annotated,caption=f'Detected apples: {len(dets)}',width='stretch')
        st.metric('Apple detections',len(dets))
        st.download_button('Download detections CSV',visualization.detections_csv(dets,names),'detections.csv','text/csv')
        buffer=io.BytesIO();annotated.save(buffer,format='PNG')
        st.download_button('Download annotated image',buffer.getvalue(),'annotated.png','image/png')
        st.divider();st.subheader('Optional ZoeDepth and mass analysis')
        if st.button('Estimate depth'):
            try:
                with st.spinner('Running ZoeDepth (first load may download weights)...'):
                    model=depth.load_model(zoe,'cuda' if device=='0' else 'cpu')
                    dm=depth.predict(model,image)
                    st.session_state['depth_output']={'fingerprint':fingerprint,'rows':depth.apple_depths(dm,dets),'heatmap':depth.heatmap(dm)}
            except Exception as exc:st.exception(exc)
        result=st.session_state.get('depth_output')
        if result and result['fingerprint']==fingerprint:
            st.image(result['heatmap'],caption='Per-image scaled depth heatmap',width='stretch')
            depth_df=pd.DataFrame(result['rows']);st.dataframe(depth_df,width='stretch')
            st.download_button('Download depth CSV',depth_df.to_csv(index=False),'depth.csv','text/csv')
            rows=mass.calculate(dets,result['rows'],names,mass.Parameters(scale=scale))
            st.warning('Mass proxy is in arbitrary model-dependent units, NOT grams.')
            mass_df=pd.DataFrame(rows);st.dataframe(mass_df,width='stretch')
            st.download_button('Download mass proxy CSV',mass_df.to_csv(index=False),'mass_proxy.csv','text/csv')
else:st.info('Upload an image to begin. Models and images stay on your computer.')
