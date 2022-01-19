# cog-replicated-docker

The Cog server from replicated https://github.com/replicate/cog that helps running Machine Learning applications using an API server through a well-defined interface. 
This is an optional image in case you need to customize your base image.

> **Base images**: The base of our project this image has the supporting base binaries that's hard to install on a host.
> * https://github.com/marcellodesales/nvidea-cuda-ubuntu-docker 
>   * NVidea CUDA 
>   * python

# How to Use

There are a few steps to run you Machine Learning model using cog:

* Create the driver `predict.py`
* Create the builder `cog.yaml`

> **DOCS**: More at https://github.com/replicate/cog/blob/main/docs/getting-started-own-model.md.

## Create a driver `predict.py`

* It will define your arguments, their respective types, etc.
* You will implement the call to your model library
* You will have an interface to return the types such as images, texts, etc.

## Create teh builder `cog.yaml` 

It helps describing your dependencies such as system-level, python, and others.

* System-dependencies: what needs to be in the container to run your model.
  * For instance,
* Model dependencies: pypi dependendencies that is part of your implementation
  * For instance, the correct versions should be properly described.
  * An example of failure: versions of pytorch must match the python version used: https://github.com/pytorch/vision#installation
* Pre-install dependencies: Those that are required to be installed after

# Example: Face improvement

* https://replicate.com/tencentarc/gfpgan/examples

## predict.py

* Define a cog input like `@cog.input("image", type=Path, help="input image")`
  * https://github.com/TencentARC/GFPGAN/pull/67/files#diff-73c1982d8a085dc10fda2ac7b6f202ae3ff9530ee6a15991c5339051eb10a49aR79

```python
# import subprocess
# subprocess.call(['sh', './run_setup.sh'])

import cog
import tempfile
import os
from pathlib import Path
import argparse
import cv2
import shutil
from basicsr.utils import imwrite
import torch
from gfpgan import GFPGANer
import glob
import numpy as np


class Predictor(cog.Predictor):
    def setup(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--upscale', type=int, default=2)
        parser.add_argument('--arch', type=str, default='clean')
        parser.add_argument('--channel', type=int, default=2)
        parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth')
        parser.add_argument('--bg_upsampler', type=str, default='realesrgan')
        parser.add_argument('--bg_tile', type=int, default=400)
        parser.add_argument('--test_path', type=str, default='inputs/whole_imgs')
        parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
        parser.add_argument('--only_center_face', action='store_true')
        parser.add_argument('--aligned', action='store_true')
        parser.add_argument('--paste_back', action='store_false')
        parser.add_argument('--save_root', type=str, default='results')

        self.args = parser.parse_args(["--upscale", "2", "--test_path", "cog_temp", "--save_root", "results"])
        os.makedirs(self.args.test_path, exist_ok=True)
        # background upsampler
        if self.args.bg_upsampler == 'realesrgan':
            if not torch.cuda.is_available():  # CPU
                import warnings
                warnings.warn('The unoptimized RealESRGAN is very slow on CPU. We do not use it. '
                              'If you really want to use it, please modify the corresponding codes.')
                bg_upsampler = None
            else:
                from realesrgan import RealESRGANer
                bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    tile=self.args.bg_tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=True)  # need to set False in CPU mode
        else:
            bg_upsampler = None

        # set up GFPGAN restorer
        self.restorer = GFPGANer(
            model_path=self.args.model_path,
            upscale=self.args.upscale,
            arch=self.args.arch,
            channel_multiplier=self.args.channel,
            bg_upsampler=bg_upsampler)

    @cog.input("image", type=Path, help="input image")
    def predict(self, image):
        input_dir = self.args.test_path

        input_path = os.path.join(input_dir, os.path.basename(image))
        shutil.copy(str(image), input_path)

        os.makedirs(self.args.save_root, exist_ok=True)

        img_list = sorted(glob.glob(os.path.join(input_dir, '*')))

        out_path = Path(tempfile.mkdtemp()) / "output.png"

        for img_path in img_list:
            # read image
            img_name = os.path.basename(img_path)
            print(f'Processing {img_name} ...')
            basename, ext = os.path.splitext(img_name)
            input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            cropped_faces, restored_faces, restored_img = self.restorer.enhance(
                input_img, has_aligned=self.args.aligned, only_center_face=self.args.only_center_face, paste_back=self.args.paste_back)

            # save faces
            for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
                # save cropped face
                save_crop_path = os.path.join(self.args.save_root, 'cropped_faces', f'{basename}_{idx:02d}.png')
                imwrite(cropped_face, save_crop_path)
                # save restored face
                if self.args.suffix is not None:
                    save_face_name = f'{basename}_{idx:02d}_{self.args.suffix}.png'
                else:
                    save_face_name = f'{basename}_{idx:02d}.png'
                save_restore_path = os.path.join(self.args.save_root, 'restored_faces', save_face_name)
                imwrite(restored_face, save_restore_path)
                # save cmp image
                cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                imwrite(restored_img, str(out_path))
                clean_folder(self.args.test_path)

        return out_path

def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

```

## cog.yaml

```yaml
build:
  gpu: true
  python_version: "3.8"
  system_packages:
    - "libgl1-mesa-glx"
    - "libglib2.0-0"
    - "libtinfo5"
  python_packages:
    # Make sure to use the correct combination of https://github.com/pytorch/vision#installation
    - "torch==1.7.0"
    - "torchvision==0.8.1"
    - "numpy==1.21.1"
    - "ipython==7.21.0"
    - "lmdb==1.2.1"
    - "opencv-python==4.5.3.56"
    - "PyYAML==5.4.1"
    - "tqdm==4.62.2"
    - "yapf==0.31.0"
    - "tb-nightly==2.7.0a20210825"
  pre_install:
    - pip install facexlib==0.2.1.1
    - pip install realesrgan

predict: "predict.py:Predictor"
```

## Dockerfile

* You can define your dockerfile with the parent image from this repo

```dockerfile
$ cat Dockerfile
ARG TENCENT_ARC_BASE_IMAGE
ARG TRAINING_FILE1
ARG TRAINING_FILE2

FROM ${TENCENT_ARC_BASE_IMAGE}

# weights
ARG TRAINING_FILE1
ENV TRAINING_FILE1 ${TRAINING_FILE1:-v0.2.0/GFPGANCleanv1-NoCE-C2.pth}
RUN echo "Downloading training file '${TRAINING_FILE1}'" && \
    wget https://github.com/TencentARC/GFPGAN/releases/download/${TRAINING_FILE1} -P experiments/pretrained_models

ARG TRAINING_FILE2
ENV TRAINING_FILE2 ${TRAINING_FILE2:-v0.1.0/GFPGANv1.pth}
RUN echo "Downloading training file '${TRAINING_FILE1}'" && \
    wget https://github.com/TencentARC/GFPGAN/releases/download/${TRAINING_FILE2} -P experiments/pretrained_models
```

## Docker-Compose

```yaml
version: "3.8"

###
### Running in MacOS
### https://stackoverflow.com/questions/64439278/gpg-invalid-signature-error-while-running-apt-update-inside-arm32v7-ubuntu20-04/64553153#64553153
###
services:

  GFPGAN:
    image: marcellodesales/tencent-arc-gfpgan-runtime
    build:
      context: .
      args:
        TENCENT_ARC_BASE_IMAGE: marcellodesales/replicated-cog-server:python3.8_nvidea1.11.1
        TRAINING_FILE1: v0.2.0/GFPGANCleanv1-NoCE-C2.pth
        TRAINING_FILE2: v0.1.0/GFPGANv1.pth
```

## Building

> NOTE: Make sure to have disk space and memory. (15GB)
> * The first time running it might takes more than 10min depending on your location. 
>   * Subsequent Builds take advantage of Docker Caches when specific layers aren't invalidated
> * Problem running: "RGPG invalid signature error while running `apt-get update`": running in MacOS you can have errors like disk space, etc. Just make sure you have enough. 
>   * https://stackoverflow.com/questions/64439278/gpg-invalid-signature-error-while-running-apt-update-inside-arm32v7-ubuntu20-04/64553153#64553153

```console
$ docker-compose build
Building GFPGAN
[+] Building 0.2s (18/18) FINISHED
 => [internal] load build definition from Dockerfile                                                                                                          0
 => => transferring dockerfile: 674B                                                                                                                          0
 => [internal] load .dockerignore                                                                                                                             0
 => => transferring context: 35B                                                                                                                              0
 => [internal] load metadata for docker.io/marcellodesales/replicated-cog-server:python3.8_nvidea1.11.1                                                       0
 => [1/3] FROM docker.io/marcellodesales/replicated-cog-server:python3.8_nvidea1.11.1                                                                         0
 => [internal] load build context                                                                                                                             0
 => => transferring context: 4.33kB                                                                                                                           0
 => CACHED [2/3] COPY cog.yaml .                                                                                                                              0
 => CACHED [3/3] RUN cat cog.yaml | yq e . - -o json | jq -r -c '.build.system_packages[]' | sed -r 's/^([^,]*)(,?)$/ \1 \2/' | tr -d '\n' > cog.pkgs &&      0
 => CACHED [4/3] RUN apt-get update -qq && apt-get install -qqy $(cat cog.pkgs) &&     rm -rf /var/lib/apt/lists/* # buildkit 85.8MB buildkit.dockerfile.v0   0
 => CACHED [5/3] RUN cat cog.yaml | yq e . - -o json | jq -r -c '.build.python_packages[]' | sed -r 's/^([^,]*)(,?)$/\1 \2/' | tr -d '\n' > cog.python-pkgs   0
 => CACHED [6/3] RUN pip install -f https://download.pytorch.org/whl/torch_stable.html $(cat cog.python-pkgs)                                                 0
 => CACHED [7/3] RUN cat cog.yaml | yq e . - -o json | jq -r -c '.build.pre_install[]' > cog.pre-inst &&     echo "Installing the pre-install packages: $(ca  0
 => CACHED [8/3] RUN sh cog.pre-inst                                                                                                                          0
 => CACHED [9/3] WORKDIR /src                                                                                                                                 0
 => CACHED [10/3] COPY predict.py .                                                                                                                           0
 => CACHED [11/3] COPY . .                                                                                                                                    0
 => CACHED [12/3] RUN echo "Downloading training file 'v0.2.0/GFPGANCleanv1-NoCE-C2.pth'" &&     wget https://github.com/TencentARC/GFPGAN/releases/download  0
 => CACHED [13/3] RUN echo "Downloading training file 'v0.2.0/GFPGANCleanv1-NoCE-C2.pth'" &&     wget https://github.com/TencentARC/GFPGAN/releases/download  0
 => exporting to image                                                                                                                                        0
 => => exporting layers                                                                                                                                       0
 => => writing image sha256:71684982ed27156781c54ef5e2f7d18a110a7aa0e150bfb49b207e1709102ceb                                                                  0
 => => naming to docker.io/marcellodesales/tencent-arc-gfpgan-runtime                                                                                         0
```

## Running

You can just create a container in the background

```console
$ docker-compose up -d
Recreating gfpgan_GFPGAN_1 ... done
```

* You can make sure that the container loaded your app and models...

```console
$ docker-compose logs -f
Attaching to gfpgan_GFPGAN_1
GFPGAN_1  | /root/.pyenv/versions/3.8.12/lib/python3.8/site-packages/torch/cuda/__init__.py:52: UserWarning: CUDA initialization: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx (Triggered internally at  /pytorch/c10/cuda/CUDAFunctions.cpp:100.)
GFPGAN_1  |   return torch._C._cuda_getDeviceCount() > 0
GFPGAN_1  | /src/predict.py:41: UserWarning: The unoptimized RealESRGAN is very slow on CPU. We do not use it. If you really want to use it, please modify the corresponding codes.
GFPGAN_1  |   warnings.warn('The unoptimized RealESRGAN is very slow on CPU. We do not use it. '
GFPGAN_1  | Downloading: "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" to /root/.pyenv/versions/3.8.12/lib/python3.8/site-packages/facexlib/weights/detection_Resnet50_Final.pth
GFPGAN_1  |
100%|------| 104M/104M [00:04<00:00, 22.7MB/s]
GFPGAN_1  |  * Serving Flask app 'http' (lazy loading)
GFPGAN_1  |  * Environment: production
GFPGAN_1  |    WARNING: This is a development server. Do not use it in a production deployment.
GFPGAN_1  |    Use a production WSGI server instead.
GFPGAN_1  |  * Debug mode: off
GFPGAN_1  |  * Running on all addresses.
GFPGAN_1  |    WARNING: This is a development server. Do not use it in a production deployment.
GFPGAN_1  |  * Running on http://172.19.0.2:5000/ (Press CTRL+C to quit)
```

## Testing: HTTP POST image=PATH

* Choose an image as the input

> Using [viu](https://github.com/atanunq/viu) to open the image on terminal

![Screen Shot 2022-01-18 at 1 56 22 PM](https://user-images.githubusercontent.com/131457/150036554-da9e637b-1b3f-4950-ae18-4b8d236e113e.png)

* Execute the Machine Learning service using the interface built by cog, which exposes the user-defined parameters.
  * In this example, `image` is a parameter

```console
$ curl http://localhost:5000/predict -X POST -F image=@$(pwd)/inputs/whole_imgs/Blake_Lively.jpg -o $(pwd)/super.jpg
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 2087k  100 1996k  100 93345   276k  12943  0:00:07  0:00:07 --:--:--  499k
```

### Output

> Using [viu](https://github.com/atanunq/viu) to open the image on terminal

![Screen Shot 2022-01-18 at 1 56 17 PM](https://user-images.githubusercontent.com/131457/150036575-7f60da84-b89e-4a1a-abcd-084472cebf80.png)

# Development 

## Build this image

* You can specify the python version, etc.

```console
docker-compose build
```

# Thoughts

# Research

* Expose Cog as a Kubernetes CRD
