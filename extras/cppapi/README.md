# RetinaNet C++ Inference API - Sample Code

The C++ API allows you to build a TensorRT engine for inference using the ONNX export of a core model.

The following shows how to build and run code samples for exporting an ONNX core model (from RetinaNet or other toolkit supporting the same sort of core model structure) to a TensorRT engine and doing inference on images.

## Building

Building the example requires the following toolkits and libraries to be set up properly on your system:
* A proper C++ toolchain (MSVS on Windows)
* [CMake](https://cmake.org/download/) version 3.9 or later
* NVIDIA [CUDA](https://developer.nvidia.com/cuda-toolkit)
* NVIDIA [CuDNN](https://developer.nvidia.com/cudnn)
* NVIDIA [TensorRT](https://developer.nvidia.com/tensorrt)
* [OpenCV](https://opencv.org/releases.html)

### Linux
```bash
mkdir build && cd build
cmake -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" ..
make
```


### Jetpack
Make sure to clone branch 20.03 as below for compatibility

```bash
git clone --branch 20.03 https://github.com/NVIDIA/retinanet-examples.git

mkdir build && cd build
cmake -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" ..
make
```


### Windows
```bash
mkdir build && cd build
cmake -G "Visual Studio 15 2017" -A x64 -T host=x64,cuda=10.0 -DTensorRT_DIR="C:\path\to\tensorrt" -DOpenCV_DIR="C:\path\to\opencv\build" ..
msbuild retinanet_infer.sln
```

## Running

If you don't have an ONNX core model, generate one from your RetinaNet model:
```bash
retinanet export model.pth model.onnx
```

Load the ONNX core model and export it to a RetinaNet TensorRT engine (using FP16 precision):
```bash
export{.exe} model.onnx engine.plan
```

You can also export the ONNX core model to an INT8 TensorRT engine if you have already done INT8 calibration:
```bash
export{.exe} model.onnx engine.plan INT8CalibrationTable
```

Run a test inference (default output if none provided: "detections.png"):
```bash
infer{.exe} engine.plan image.jpg [<OUTPUT>.png]
```

Extra functionality has been added with this fork, and all of the follwing has been tested with bot a Xavier NX and an AGX Xavier. First of all go ahead and use infervideo as follows:

```bash
infervideo{.exe} engine.plan input.mov output.mp4 label_map.txt
```

The resulting video will have both the object class and confidence written above each bounding box per frame, given the correct label map is provided.

Furthermore for an example of realtime usage take a look at an example using a Zed2 camera as follows:

```bash
infervideozed{.exe} engine.plan label_map.txt
```

Once again the resulting window will display an object's class and confidence per bounding boxes. Further to this find FPS and inference speed in ms in the top left corner of the window. Note that the example ensures that the Zed2's left camera is used but this example can easily be adapted to a monocular UVC camera.


Note: make sure the TensorRT, CuDNN and OpenCV libraries are available in your environment and path.

We have verified these steps with the following configurations:
* DGX-1V using the provided Docker container (CUDA 10, cuDNN 7.4.2, TensorRT 5.0.2, OpenCV 3.4.3)
* Jetson AGX Xavier with JetPack 4.1.1 Developer Preview (CUDA 10, cuDNN 7.3.1, TensorRT 5.0.3, OpenCV 3.3.1)




