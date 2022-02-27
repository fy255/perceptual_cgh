import os
import sys
import cv2
import torch
import glob
import time
from utils.slmDisplay import slmDisplay
from utils import image_processing as ip
from utils.MvImport import CamOpreationClass as camOp
import utils.pyEdsdk as pyEdsdk
import pickle


def hardware_calibr(calibr_grid_size=(18, 10),
                    calibr_ellipse_size=(15, 15), calibr_spacing=(80, 80), calibr_pad_pixels=(0, 0),
                    expoPara=None, cameraCapture_folder='cameraCapture', calibr_folder='PatternCalibration',
                    show_preview=False,
                    camModel=0):
    calibr_folder = os.path.join(os.getcwd(), cameraCapture_folder, calibr_folder)
    patternName = os.path.join(calibr_folder, '15cm_6.4um_532nm_Grid_13_22.bmp')
    # 1.1 Connect Hikivision Camera
    if camModel == 0:
        camera = camOp.enum_devices(0)
        camera.Open_device()  # specify the camera to use, 0 for main cam, 1 for the second cam
        camera.get_parameter()
        if expoPara is not None:
            camera.set_parameter(15, expoPara, 0)

    # 1.2 Connect Cannon Camera
    elif camModel == 1:
        camera = pyEdsdk.CanonController()
        folderName = os.path.join(cameraCapture_folder, calibr_folder)
    else:
        sys.exit("Invalid camera model")

    # 2. Connect SLM
    slm = slmDisplay()
    slm.open()

    # 3. Calibrate hardware using homography

    HMatrix, dSize = calibrate(camera, slm, patternName, calibr_folder, calibr_grid_size, calibr_ellipse_size,
                               calibr_spacing, calibr_pad_pixels, camModel, show_preview=show_preview)
    return camera, slm, HMatrix, dSize


def hardware_prop(camera, slm, slm_phase, H, dSize, camModel=1, expoPara=None, num_grab_images=3):
    # TODO: display the pattern and capture linear intensity, after perspective transform
    slm.update(slm_phase)
    if camModel == 0:
        if expoPara is not None:
            camera.set_parameter(15, expoPara, 0)
        camera.Start_grabbing()
        camera.bmp_save(num_grab_images, 'ResultImage', 'unCalibr')
        camera.Stop_grabbing()
        raw_image = camera.return_image()
        raw_image = cv2.resize(raw_image, (1920, 1080))
        calibr_folder = os.path.join(os.getcwd(), 'cameraCapture', 'ResultImage')
        calibr_path = os.path.join(calibr_folder, '*.bmp')
    if camModel == 1:
        # TODO:Canon Camera
        result_folder = os.path.join(os.getcwd(), 'cameraCapture', 'ResultImage')
        calibr_path = os.path.join(result_folder, '*.jpg')
        camera.Take_pic(num_grab_images, result_folder, 'unCalibr')
    # TODO: average the intensity for arbitrary images

    images = glob.glob(calibr_path)
    image_data = []
    for fName in images:
        # this_image = cv2.resize(cv2.imread(fName), (1920, 1080))
        this_image = cv2.imread(fName)
        image_data.append(this_image)
    avg_image = image_data[0]
    for i in range(len(image_data)):
        if i == 0:
            pass
        else:
            alpha = 1.0 / (i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(image_data[i], alpha, avg_image, beta, 0.0)
    raw_image = image_data[-1]
    img_undistorted_RGB = ip.calibr_undistortImg(raw_image, H, dSize)
    img_undistorted_mono = ip.img_linearize(img_undistorted_RGB, 1).squeeze()
    img_undistorted_mono_tc = torch.tensor(img_undistorted_mono, dtype=torch.double)
    img_undistorted_amp = torch.sqrt(img_undistorted_mono_tc)

    return img_undistorted_amp, raw_image, img_undistorted_RGB


def calibrate(camera, slm, patternName, calibr_folder, calibr_grid_size, calibr_ellipse_size, calibr_spacing,
              calibr_pad_pixels, camModel, show_preview=False):
    slm.update(patternName)
    if camModel == 0:
        calibr_path = os.path.join(calibr_folder, '*.bmp')
        camera.Start_grabbing()
        time.sleep(3)
        camera.bmp_save(2, 'PatternCalibration', 'test_pattern')
        camera.Stop_grabbing()
        H, dSize, calibr_success = ip.calibr_autoCalibration(calibr_path, calibr_grid_size,
                                                             calibr_ellipse_size,
                                                             calibr_spacing, calibr_pad_pixels)
    elif camModel == 1:
        camera.Take_pic(numImg=2, folderName=calibr_folder, imageName='test_pattern')
        calibr_path = os.path.join(calibr_folder, '*.jpg')
        H, dSize, calibr_success = ip.calibr_autoCalibration(calibr_path, calibr_grid_size,
                                                             calibr_ellipse_size,
                                                             calibr_spacing, calibr_pad_pixels,
                                                             show_preview=show_preview)
    else:
        sys.exit("Invalid camera model")

    if calibr_success:
        print('   - calibration success')
        return H, dSize
    else:
        raise ValueError('  - Calibration failed')


def loadCamSettings(setting_path, camModel=1):
    # 1.2 Connect Cannon Camera
    if camModel == 1:
        camera = pyEdsdk.CanonController()
    # 2. Connect SLM
    slm = slmDisplay()
    slm.open()

    # TODO:. Calibrate hardware using homography
    f = open(setting_path, 'rb')
    obj = pickle.load(f)
    f.close()
    HMatrix = obj[0]
    dSize = obj[1]
    return camera, slm, HMatrix, dSize
