"""Offline HSV training-crop distributions; never computed from uploaded demo images."""
from pathlib import Path
import numpy as np
import pandas as pd

def summarize(crops,labels,bins=36):
    if not len(crops) or len(crops)!=len(labels):raise ValueError('Invalid crop collection')
    rows=[];hist=[]
    for pix,label in zip(crops,labels):
        k=int(label);h=pix[:,0].astype(float)*360/256;theta=np.deg2rad(h)
        rows.append({'cluster':k,'cos':np.cos(theta).mean(),'sin':np.sin(theta).mean(),'saturation':pix[:,1].mean()/255,'value':pix[:,2].mean()/255})
        for channel,v,limit in [('Hue',h,360),('Saturation',pix[:,1]/255,1),('Value',pix[:,2]/255,1)]:
            counts=np.histogram(v,bins=bins,range=(0,limit))[0]/len(v)
            for b,f in enumerate(counts):hist.append({'cluster':k,'channel':channel,'bin':b,'fraction':float(f)})
    frame=pd.DataFrame(rows);summary=[]
    for k,g in frame.groupby('cluster'):
        c,s=g.cos.mean(),g.sin.mean()
        summary.append({'cluster':int(k),'n_apples':len(g),'mean_hue_degrees':float(np.degrees(np.arctan2(s,c))%360),'hue_concentration':float(np.hypot(c,s)),'mean_saturation':float(g.saturation.mean()),'mean_value':float(g.value.mean())})
    dist=pd.DataFrame(hist).groupby(['cluster','channel','bin'],as_index=False).fraction.mean()
    return pd.DataFrame(summary),dist

def save_report(crops,labels,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    summary,dist=summarize(crops,labels)
    summary.to_csv(out/'cluster_summary.csv',index=False);dist.to_csv(out/'hsv_distributions.csv',index=False)
    clusters=sorted(summary.cluster.unique());n=36
    for channel,name in [('Hue','hue_distribution.png'),('Saturation','saturation_distribution.png'),('Value','value_distribution.png')]:
        fig,axes=plt.subplots(1,len(clusters),figsize=(5*len(clusters),4.8),subplot_kw={'projection':'polar'} if channel=='Hue' else {})
        for ax,k in zip(np.atleast_1d(axes),clusters):
            y=dist[(dist.cluster==k)&(dist.channel==channel)].sort_values('bin').fraction.to_numpy()
            if channel=='Hue':
                theta=np.linspace(0,2*np.pi,n,endpoint=False)
                colors=[hsv_to_rgb((float(i)/n,1,1)) for i in range(n)]
                ax.bar(theta,y,width=2*np.pi/n,align='edge',color=colors,edgecolor='white',linewidth=.3)
                ax.set_thetagrids([0,60,120,180,240,300])
            else:
                x=np.arange(n)/n
                colors=[hsv_to_rgb((0,float((i+.5)/n) if channel=='Saturation' else 0,1 if channel=='Saturation' else float((i+.5)/n))) for i in range(n)]
                ax.bar(x,y,width=1/n,align='edge',color=colors,edgecolor='#777',linewidth=.25)
                ax.set_xlim(0,1);ax.set_xlabel(channel+' (0–1)')
            ax.set_title(f'Cluster {k}');
        fig.suptitle('Training apple crops — '+channel+' distribution');fig.tight_layout();fig.savefig(out/name,dpi=170,bbox_inches='tight');plt.close(fig)
    return summary,dist
