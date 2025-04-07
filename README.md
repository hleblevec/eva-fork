# 💥🔵 Low-Latency Event-based Obstacle Avoidance on a FPGA. 💥🔵

## [🗂️ ABCD Dataset](https://zenodo.org/records/14711527)

## References & Citation ✉️ 

Code base to reproduce results in :
``` 
@article{Bonazzi2025CVPRW,
    author    = {Bonazzi, Pietro and Vogt, Christian and Jost, Michael and Khacef, Lyes and Paredes-Valles, Federico and Magno, Michele},
    title     = {Towards Low-Latency Event-based Obstacle Avoidance on a FPGA-Drone.},
    journal   = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshop (CVPRW)}, 
    month     = {June},
    year      = {2025}, 
} 
@article{Bonazzi2025IJCNN,
    author    = {Bonazzi, Pietro and Vogt, Christian and Jost, Michael and Qin, Haotong and Khacef, Lyes and Paredes-Valles, Federico and Magno, Michele},
    title     = {Fusing Events and Frames with Self-Attention Network for Ball Collision Detection.},
    journal = {International Joint Conference on Neural Networks (IJCNN)},
    month     = {June},
    year      = {2025}, 
}
```

Leave a star to support our open source initiative!⭐️ 

The name of the main folder with all the modules is `eva` which stands for Event Vision for Action prediction.

## 1. Installation instructions 🚀

### 1.1. Downloads 📥

To download the dataset click [here](https://zenodo.org/records/14711527).

Click on this [link](https://zenodo.org/records/15166553) to download the models.   

### 1.2. Docker containers 🛠️

Replace the path variable of the dataset (`ABCD_DATA_PATH`) to your installation directory. Build the docker container:
```
docker build -t eva-pytorch .
```

Run the docker container:
```
docker run --gpus all -it --rm \
  -v /datasets/pbonazzi/sony-rap/glc_dataset/vicon_aggregate/:/workspace/data \
  -e DISPLAY=$DISPLAY \
  --network host \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  eva-pytorch
``` 

In your shell, inside the code directory, install [OpenEB](https://docs.prophesee.ai/stable/installation/linux_openeb.html), following these instructions :
```
git clone https://github.com/prophesee-ai/openeb.git --branch 4.6.0 
cd openeb && mkdir build && cd build 
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF 
cmake --build . --config Release -- -j $(nproc) 
. utils/scripts/setup_env.sh 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib 
export HDF5_PLUGIN_PATH=$HDF5_PLUGIN_PATH:/usr/local/lib/hdf5/plugin 
export HDF5_PLUGIN_PATH=$HDF5_PLUGIN_PATH:/usr/local/hdf5/lib/plugin
cd ../..
``` 

## 2. Reproduce the results 🚀

First, visualize a recording 👀
```
python3 -m eva.data.subclasses.flight --id=1
``` 

Then, evaluate the EVS and RGB models 📚 For example Tab. 3 [1] can be reproduced as follows:
```
python3 -m scripts.eval.cvprw.tab_3
```

## 3. Train a model 🏋️‍♂️

To train the fusion model: 
```
python3 -m scripts.train --inputs_list=["dvs","rgb"] --name=fusion-model --frequency="rgb"
```

To train the event-based model: 
```
python3 -m scripts.train --inputs_list=["dvs"] --name=dvs-model --frequency="dvs"
```

To train the rgb-based model: 
```
python3 -m scripts.train --inputs_list=["rgb"] --name=rgb-model --frequency="rgb" 
```

## 4. Deploy the event-based model on AMD Kria K26  📦

Deploy instructions coming soon.
