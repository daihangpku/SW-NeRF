# NeRF_Fans
## Quick Start
create a virtual environment and clone the repository
  ```bash
  conda create -n nerf_fans python=3.9
  conda activate nerf_fans
  git clone <repository_url>
  cd NeRF_Fans
  pip install -r requirements.txt
  ```
get torch ready(replace XXX with your cuda version)
```bash  
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
```


##  SW-nerf
  Train a vallina nerf from scratch using example dataset.
  ```bash
  cd nerf
  python run.py --config configs/lego.txt
  cd ..
  ```
  extract mesh with checkpoint
  ```bash
  cd nerf
  python extract_mesh.py --config configs/drill.txt
  cd ..
  ```
  you will get logs/drill/mesh.obj


  if you are using a dataset with aruco
  ```bash
  cd nerf
  python transform_mesh.py --config configs/drill.txt
  cd ..
  ```
  you will get a real-scaled mesh as transformed_mesh.obj
## D-nerf
  ```bash
  cd d_nerf
  python run_dnerf.py --config configs/bouncingballs.txt
  ```




