"""Fit spectral clusters from training split only; export reference and HSV distribution plots."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from appledetecting import maturity
from appledetecting.hsv_report import save_report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=Path('datasets/apple_pseudo'))
    parser.add_argument('--out',type=Path,default=Path('models/maturity_reference.npz'))
    parser.add_argument('--max-crops',type=int,default=100000)
    args=parser.parse_args()
    x=[];pixels=[]
    folder=args.dataset/'images/train'
    for image_file in sorted(folder.rglob('*')):
        if image_file.suffix.lower() not in ('.jpg','.jpeg','.png','.webp'):continue
        label_file=args.dataset/'labels/train'/image_file.relative_to(folder).with_suffix('.txt')
        if not label_file.is_file():continue
        with Image.open(image_file) as image:
            image=image.convert('RGB');w,h=image.size
            for line in label_file.read_text(encoding='utf-8').splitlines():
                cols=line.split()
                if len(cols)!=5 or int(cols[0])!=0:continue
                xc,yc,bw,bh=map(float,cols[1:]);box=((xc-bw/2)*w,(yc-bh/2)*h,(xc+bw/2)*w,(yc+bh/2)*h)
                if min(bw*w,bh*h)<8:continue
                try:
                    x.append(maturity.features(image,box));pixels.append(maturity.pixels(image,box))
                except ValueError:continue
                if len(x)>=args.max_crops:break
        if len(x)>=args.max_crops:break
    if len(x)<3:raise SystemExit('Insufficient training crops')
    centers,labels=maturity.fit(np.stack(x))
    maturity.save(args.out,centers,{'source':'train pseudo-labels','n_crops':len(x),'seed':42,'bins':16,'clusters':3,'semantic_labels':'unverified'})
    save_report(pixels,labels,args.out.parent/'maturity_report')
    print(f'Saved {args.out}, training crops={len(x)}. Recheck cluster-name mapping after every refit.')
if __name__=='__main__':main()
