# Windows + RTX 4060 Laptop: setup

The verified working configuration reported during development was Python **3.11.18**, PyTorch **2.7.1+cu128**, `timm==0.6.13`, and ZoeDepth Git commit `d87f17b2f5fdcb174cf4fb115491f4a6c60de152`. YOLOv5 was observed at `v7.0-561-gaf08ba15` in the evaluation log. The upstream projects are external dependencies.

1. Install Git and Conda, create `appledetecting` with Python 3.11.
2. Install a CUDA-enabled PyTorch wheel matching your NVIDIA driver. Previously working:

```cmd
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

3. Install project and dependencies:

```cmd
python -m pip install -e ".[dev]"
python -m pip install timm==0.6.13
```

4. Clone third-party repos from the **project root**:

```cmd
git clone https://github.com/ultralytics/yolov5.git external/yolov5
git clone https://github.com/isl-org/ZoeDepth.git external/ZoeDepth
git -C external/ZoeDepth checkout d87f17b2f5fdcb174cf4fb115491f4a6c60de152
```

Install upstream YOLOv5 requirements, taking care not to replace the working PyTorch wheel. ZoeDepth is older; if its dependencies conflict, use a separate environment or a pinned, verified setup. Check `python -m pip check` and `python scripts/check_environment.py` after installing.

5. Put your locally trained checkpoint at `runs/train/apple_pseudo_v1/weights/best.pt`, and generated maturity reference at `models/maturity_reference.npz`. The JSON mapping is versioned in `models/cluster_names.json`.
6. Run `python -m streamlit run app/main.py`.

**Note:** Model weights are not bundled. ZoeDepth may download its checkpoint on first use. Do not modify upstream source files unless necessary and documented.
