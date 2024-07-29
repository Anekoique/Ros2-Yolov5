# ros2-yolov5
Use Raspberry Pi with a camera to publish images, and use a local device to subscribe and perform DNN/tensorRT inference with acceleration.

--------

# Requirements

OS: ubuntu: 22.04

## Ros2

version: humble

## Cuda / Cudnn

Cuda version: 12.1

```
nvcc -V
```

Cudnn version:  8.9.6

```
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

## opencv

version: 4.9.0

## tensorRT

### installation

一、find a version that suits you

refer to https://docs.nvidia.com/deeplearning/tensorrt/release-notes/

二、install 

1. Go to: https://developer.nvidia.com/tensorrt.
2. Click **GET STARTED**, then click **Download Now**.
3. Select the version of TensorRT.

- deb安装

```
// 使用deb安装
// cuda必须也是用deb安装，否则会报错
os="ubuntuxx04"
tag="10.x.x-cuda-x.x"
sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt
```

- tar安装

```
version="10.x.x.x"
arch=$(uname -m)
cuda="cuda-x.x"
tar -xzvf TensorRT-${version}.Linux.${arch}-gnu.${cuda}.tar.gz
export LD_LIBRARY_PATH=<TensorRT-${version}/lib>:$LD_LIBRARY_PATH // lib绝对路径加入bashrc
cd TensorRT-${version}/python
python3 -m pip install tensorrt-*-cp3x-none-linux_x86_64.whl
python3 -m pip install tensorrt_lean-*-cp3x-none-linux_x86_64.whl
python3 -m pip install tensorrt_dispatch-*-cp3x-none-linux_x86_64.whl
```

## docker

```
// ERROR: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: 
sudo groupadd docker               #添加用户组
sudo gpasswd -a username docker    #将当前用户添加至用户组
newgrp docker                      #更新用户组
```

# TensorRT

ARCH parameters:https://developer.nvidia.cn/cuda-gpus#compute

## References

https://github.com/kalfazed/tensorrt_starter

https://deployment.gitbook.io/love

https://github.com/Melody-Zhou/tensorRT_Pro-YOLOv8?tab=readme-ov-file

https://github.com/shouxieai/tensorRT_Pro
