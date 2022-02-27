# Perceptual Computer Generated Holography

### [CMMPE Group Page](http://www-g.eng.cam.ac.uk/CMMPE/index.html)

[Fan Yang](http://www-g.eng.cam.ac.uk/CMMPE/Team/FanYang.html),
[Andrew Kadis](http://www-g.eng.cam.ac.uk/CMMPE/Team/AndrewKadis.html),
[Ralf Mouthaan](https://www.ralfmouthaan.com/home),
[Benjamin Wetherfield](http://www-g.eng.cam.ac.uk/CMMPE/Team/BenjaminWetherfield.html),
[Andrzej Kaczorowski](http://www.drandrzejka.com/),
[Tim D. Wilkinson](http://www-g.eng.cam.ac.uk/CMMPE/Team/TimWilkinson.html)

[Paper in submission] | [[Dataset]](https://drive.google.com/drive/folders/1n2AsozJ0Z1_pMrvjVmeEDdQGHN2X8vn2?usp=sharing)

This repository contains the scripts associated with the paper "Perceptually-Motivated Loss Functions for
Computer-Generated Holographic Displays".

## Installation

### Requirements:

- python=3.9
- IQA_pytorch==0.1
- matplotlib==3.5.1
- numpy==1.20.3
- Pillow==9.0.1
- piq==0.6.0
- screeninfo==0.8
- pyTorch==1.10.2
- opencv-python==4.5.*
- tensorboard ==2.8.0
- cudatoolkit=11.3

You can set up a new conda environment with pip:

```
conda create --name perceptual_cgh python=3.9
conda activate perceptual_cgh
pip install -r requirements.txt
```

or with all dependencies in windows:

```
conda env create -f environment_windows.yml
conda activate perceptual_cgh
```

For ubuntu:

```
conda env create -f environment_ubuntu.yml
conda activate perceptual_cgh
```

Test:

```
python main.py
```

Related files in the  ```./utils/``` folder may need to be **modified** without camera SDKs.

For Canon Camera capture, please download ```EDSDK.dll```and```EDsImage.dll```
from [Canon EDSDK](https://developers.canon-europe.com/developers/s/camera) to the ```./utils/pyEdsdk/``` folder before
the running. For HikRobot camera, please download its SDK
from [Hikvision SDK](https://www.hikrobotics.com/en/machinevision/service/download?module=0).

## Project structure

This project includes:

### main scripts

* ```main.py``` generates phase patterns with perceptual losses.
* ```eval.py```  evaluates optimized phase patterns with hardware captures or simulated results.

### wave propagation methods

* ```asm.py``` includes the angular spectrum wave propagation model.
* ```slm_prop.py``` includes the hardware wave propagation using SLM for experimental reconstruction.

### CGH optimization methods

* ```gerchberg_saxton.py``` includes the Gerchberg-Saxton algorithm for CGH optimization.
* ```gradient_descent.py``` includes the gradient descent algorithm with perceptual loss function for CGH optimization.

### utility scripts

* ```./utils/MvImport/``` includes utility functions for Hikvision camera operations.
* ```./utils/pyEdsdk/``` includes Python wrapper modules for Cannon camera operations.
* ```./utils/Licenses/``` includes required licenses.
* ```image_processing.py``` includes image processing functions for image operation and display.
* ```image_dataset.py``` image dataset class and hologram dataset class.
* ```slmDisplay.py ``` includes the SLM display controller.
* ```torch_complex.py ``` includes utility functions for complex tensor operations.
* ```configs.py ``` includes CGH and algorithm parameters with other utility functions.

### Data files

* ```./cameraCapture/``` folder for saving raw images and grid pattern for camera calibration.
* ```./data/``` folder for target images, captured image amps, reconstructed images and generated holograms.

## Running the test

### Phase optimization

The SLM phase patterns can be reproduced with:

#### ASM with Gradient Descent

The default algorithm is ASM + gradient descent optimization.

```
python main.py --loss ['MSE']
```

For other colour simulation,  ```--channel```,can be set to ```0/1/2``` which corresponds to `R/G/B`, respectively.
Details on other default settings can be seen in ```configs.py```.

This simulation code reads ```./data/image``` folder and writes the hologram/replay/captured images in
the ```./data/temp``` folder as default.

#### Gerchberg-Saxton

```
python main.py --channel=1 --optimize_method=gerchberg_saxton
```

### Simulation/Evaluation

Different evaluation methods for perceptual image evaluation can be selected
from  [IQA](https://github.com/dingkeyan93/IQA-optimization)
and [PIQ](https://github.com/photosynthesis-team/piq) libraries.

The simulated the holographic image reconstructions are generated along with optimized phase patterns by default. With
Canon SDK enabled ,we can capture the replay image and evaluate its image quality with the physical setup:

```
python eval.py --channel 1 --camera_model=1
```

### SLM display, Camera Control and Calibration

To choose different type of cameras for image capture: ```--camera_model```, ```0/1/2``` corresponds
to `Hikvison/Cannon/NoCamera`, respectively.

The camera is pre-calibrate the camera with a grid pattern by calling: ```slm_prop.hardware_calibr``` function.Once the
camera is calibrated, the function will return a camera object ```camera``` and an SLM object ```slm``` with a
homography matrix ```HMatrix``` and calibrated image resolution ```dSize```. Then, the
function ```slm_prop.hardware_prop``` can get camera-captured images by simply sending SLM phase patterns for physical
display. The returned arguments include captured image amplitude(monochrome)```captured_amp```, captured raw image saved
in bmp```raw_image```, and captured image amplitude(rgb)```cap_rgb```.

## PerceptualCGH Dataset


We are making a large hardware collected dataset -PerceptualCGH dataset- publicly-available for CGH optimization and
evaluation with perceptual losses. It contains experimentally captured images with corresponding holograms and simulated
reconstructed images.

The dataset is structured as:

- methods 01-06: MAE, MSE, and NLPD.
- methods 04-06:  SSIM, MS_SSIM and PIQ_FSIM.
- methods 07-09: PIQ_MS_GMSD, PIQ_VSI, and PIQ_HaarPSI.
- methods 10-12: PIQ_VIF, LPIPSVgg and DISTS.
- target images: target images modified from the DIV2K dataset.
- readme.txt:  CGH parameters and algorithms parameters used.

Each method includes the optimized phase pattern in 8-bit, simulated reconstructed images and captured images.

- Holograms are named as: "$imageName_holo_$PerceptualLossName_$PerceptualLossValue_MSE_$MSELossValue.bmp"
- Reconstructed images are named as: "$imageName_rpf_$PerceptualLossName_$PerceptualLossValue_MSE_$MSELossValue.bmp"
- Captured images are named as: "$imageName_cap_RGB_MSE_$MSELossValue_SSIM_$SSIMLossValue.bmp"

## Reference

1. Comparison of Image Quality Models for Optimization of Image Processing Systems
   [[Paper]](https://arxiv.org/abs/2005.01338)
   [[Code]](https://github.com/dingkeyan93/IQA-optimization)
2. DIV2K dataset: DIVerse 2K resolution high quality images as used for the challenges
   [[Paper]](https://piq.readthedocs.io/en/latest/)
   [[Dataset]](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
3. Neural Holography with Camera-in-the-loop Training
   [[Paper]](https://dl.acm.org/doi/abs/10.1145/3414685.3417802)
   [[Code]](https://github.com/computational-imaging/neural-holography)
4. PyTorch Image Quality(PIQ)
   [[Paper]](https://piq.readthedocs.io/en/latest/)
   [[Code]](https://github.com/photosynthesis-team/piq)
5. Machine Vision Software for Hikrobot
   [[SDK]](https://www.hikrobotics.com/en/machinevision/service/download?module=0)
6. Canon EOS Digital SDK
   [[SDK]](https://developers.canon-europe.com/developers/s/camera)

## License

This project is licensed under the MIT license, excepting the image  "./data/temp/0006.jpg" and images
in dataset "target_images.rar" for [academic research purpose only](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

MIT License

Copyright (c) 2022 fy255

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Contact

If you have any questions, please contact:

* Email: fy255@cam.ac.uk
* Website: http://www-g.eng.cam.ac.uk/CMMPE/
* Address: Centre of Molecular Materials for Photonics and Electronics (CMMPE), Centre for Advanced Photonics and
  Electronics, University of Cambridge, 9 JJ Thomson Avenue, Cambridge, UK, CB3 0FF