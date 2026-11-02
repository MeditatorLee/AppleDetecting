"""Print installed runtime versions; does not change environment."""
import importlib.metadata as meta
import subprocess
import sys
print('Python:',sys.version)
for pkg in ('torch','torchvision','timm','streamlit','scikit-learn','numpy','Pillow'):
    try:print(pkg,meta.version(pkg))
    except meta.PackageNotFoundError:print(pkg,'not installed')
try:
    import torch
    print('CUDA available:',torch.cuda.is_available())
    if torch.cuda.is_available():print('GPU:',torch.cuda.get_device_name(0))
except ImportError:pass
for repo in ('external/yolov5','external/ZoeDepth'):
    p=subprocess.run(['git','-C',repo,'rev-parse','HEAD'],capture_output=True,text=True,check=False)
    print(repo,p.stdout.strip() or 'not found')
