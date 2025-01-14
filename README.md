# NeRF_Fans
## Quick Start
create a virtual environment and clone the repository
  ```bash
  conda create -n sw-nerf python=3.9
  conda activate sw-nerf
  git clone <repository_url>
  cd SW-NeRF
  pip install -r requirements.txt
  ```
get torch ready(replace XXX with your cuda version)
```bash  
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
```


###  SW-nerf(including Vanilla NeRF)
  .txt under dir nerf/configs are prepared for several classic dataset.
  for quick start, you can
  ```bash
  cd nerf
  bash download_example_data.sh
  cd ..
  ```
  Train a vallina nerf from scratch using example dataset.
  ```bash
  cd nerf
  python run.py --config configs/lego.txt
  cd ..
  ```
  extract mesh with checkpoint like logs/expname(in configs/datasetname.txt)/010000.tar
  ```bash
  cd nerf
  python extract_mesh.py --config configs/drill.txt --resolution 128 --threshold 30
  cd ..
  ```
  you will get logs/drill/mesh.obj
  tips: the bounds in extra_mesh.py should be specified according to your data.


  if you are using a dataset with aruco which has a structure like
  - drill
    - images(segmented images)
    - images_ori(images with aruco(unsegmented))
    - transforms.json
  and already has a mesh in logs/datasetname/mesh.obj
  ```bash
  cd nerf
  python transform_mesh.py --config configs/drill.txt --real_length xx
  cd ..
  ```
  you will get a real-scaled mesh as transformed_mesh.obj in the same dirctory.


  if you want to test the model with checkpoint in nerf/logs/expname/xxxxxx.tar
  ```bash
  cd nerf
  python run.py --config configs/xxxx.txt --render_only --render_test
  cd ..
  ```
For the following experiments, the dataset is blender data in DNeRF dataset; it should be put in the same directory as the python runner script.
An example of the dataset directory is data/bouncingballs

### D-nerf
  To train a D-NeRF, first download the dataset. Then, 
  ```bash
  cd d_nerf
  python run_dnerf.py --config configs/bouncingballs.txt
  cd ..
  ```
  if you want to train the model with auxiliary tv-loss:
  ```bash
  cd d_nerf
  python run_dnerf.py --config configs/bouncingballs.txt --add_tv_loss
  cd ..
  ```
  To test the D-NeRF model, first download pre-trained weights and dataset. Then,    
  ```bash
  cd d_nerf
  python run_dnerf.py --config configs/bouncingballs.txt  --render_only --render_test
  cd ..
  ```
  This command will run the `bouncingballs` experiment. When finished, results are saved to `./d_nerf/logs/bouncingballs/renderonly_test_800000` To quantitatively evaluate model run `metrics.ipynb` notebook.
### T-nerf
  To train a T-NeRF, first download the dataset. Then, 
  ```bash
  cd t_nerf
  python run_tnerf.py --config configs/bouncingballs.txt
  cd ..
  ```
### MultiRes-nerf
  To train a MultiRes-NeRF, first download the dataset. Then, 
  ```bash
  cd multi_res_nerf
  python multires_dnerf.py --config configs/bouncingballs.txt
  cd ..
  ```