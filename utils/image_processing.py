import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt

import utils
import utils.torch_complex as tc
import glob


##Basic Image processing

# pad input field into target resolution
def img_pad(u_in, target_res, padval=0, mode='constant'):
    if torch.is_tensor(u_in):
        size_diff = target_res - torch.tensor(u_in.size())
        # pad images when target resolution is larger than the original resolution
        if (size_diff > 0).any():
            pad_total = torch.maximum(size_diff, torch.tensor(0))
            pad_size = torch.div(pad_total, 2, rounding_mode='floor')
            pad_axes = [pad_size[1], pad_size[1], pad_size[0], pad_size[0]]
            return nn.functional.pad(u_in, pad_axes, mode=mode, value=padval)
        return u_in
    else:
        size_diff = np.array(target_res) - np.array(u_in.shape[-2:])
        if (size_diff > 0).any():
            pad_total = np.maximum(size_diff, 0)
        # TODO: pad non-pyTorch field
        print("TODO: pad non-pyTorch field")
        pass
    return u_in


# crop input field into target resolution
def img_crop(u_in, target_res, padval=0, mode='constant'):
    if target_res is None:
        return u_in
    if torch.is_tensor(u_in):
        size_diff = torch.tensor(u_in.size()) - target_res
        # crop images when target resolution is smaller than the original resolution
        if (size_diff > 0).any():
            crop_total = torch.maximum(size_diff, torch.tensor(0))
            crop_size = torch.div(crop_total, 2, rounding_mode='floor')
            pad_axes = [-crop_size[1], -crop_size[1], -crop_size[0], -crop_size[0]]
            return nn.functional.pad(u_in, pad_axes, mode=mode, value=padval)
        return u_in

    else:
        size_diff = np.array(u_in.shape[-2:]) - np.array(target_res)
        if (size_diff > 0).any():
            crop_total = np.maximum(size_diff, 0)
            # TODO: crop non-pyTorch field
            print("TODO: crop non-pyTorch field")
        return u_in


# linearize input image and convert to amplitude
def img_linearize(image, channel, dtype=np.float):
    image = cv2.normalize(image.astype(dtype), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image = image[..., channel, np.newaxis]
    # linearize intensity
    low_val = image <= 0.04045
    image[low_val] = 25 / 323 * image[low_val]
    image[np.logical_not(low_val)] = ((200 * image[np.logical_not(low_val)] + 11)
                                      / 211) ** (12 / 5)
    return image


def torch_img_normalize_01(image):
    range_image = image.max() - image.min()
    normalised = (image - image.min()) / (range_image + 10e-15)
    return normalised

##Image display

def surf(array3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(1, array3D.shape[-1] + 1)
    y = np.arange(1, array3D.shape[-2] + 1)
    X, Y = np.meshgrid(x, y)
    if torch.is_tensor(array3D):
        Z = tc.torch2np(array3D)
    else:
        Z = array3D

    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def imshow(array2D):
    if torch.is_tensor(array2D):
        Z = tc.torch2np(array2D)
    else:
        Z = array2D
    plt.imshow(Z)
    plt.show()


def imout(path, name, image):
    if 'holo' in path:
        fileName = name + '_holo' + '.bmp'
    elif 'rpf' in path:
        fileName = name + '_rpf' + '.bmp'
    elif 'targetAmp' in path:
        fileName = name + '_targetAmp' + '.bmp'
    elif 'capAmp' in path:
        fileName = name + '_capAmp' + '.bmp'
    os.path.join(path, fileName)
    targetAmp_name = os.path.join(path, fileName)
    if torch.is_tensor(image):
        image = utils.torch2np(image)
    if image.max() <= 1.:
        image = image * 255
    cv2.imwrite(targetAmp_name, image)


##Image Calibration functions

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Read image in the current folder and resize to 1920*1080
def calibr_imread2gray(fName):
    img = cv2.imread(fName)
    imgROI = img
    grayImag = cv2.cvtColor(imgROI, cv2.COLOR_BGR2GRAY)
    return grayImag


# Create ellipse kernel and add gaussian blur for a loose threshold
def calibr_ellipseThreshold(grayImag, ellipse_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ellipse_size)
    res = cv2.morphologyEx(grayImag, cv2.MORPH_CLOSE, kernel)
    blur = cv2.GaussianBlur(res, (3, 3), 0)
    ret3, BinaryImg = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return BinaryImg


# Find contours in a binary image and enclose contours as circles.
def calibr_contours2Circle(BinaryImg):
    contours, hierarchy = cv2.findContours(BinaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # imCopy = cv2.drawContours(grayImag, contours, -1, (0, 0, 255), 3)
    imCir = BinaryImg.copy()
    imCir[:] = 0
    for ls in contours:
        cnt = ls
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(imCir, center, radius, (255, 255, 255), 2)
    # fill the circle
    h, w = imCir.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(imCir, mask, (0, 0), 255)
    return imCir


# Find circle grid on the filled contours
def calibr_findCircleGrid(image_in, imag_contours, grid_size, detector):
    rets, centers = cv2.findCirclesGrid(imag_contours, grid_size, blobDetector=detector,
                                        flags=cv2.CALIB_CB_SYMMETRIC_GRID)
    img = cv2.drawChessboardCorners(image_in, grid_size, centers, rets)
    return rets, centers, img


# Return Homography matrix
def calibr_findHomography(grid_size, spacing, pad_pixels, centers):
    # Find transformation
    H = np.array([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]], dtype=np.float32)
    # Generate reference points to compute the homography
    ref_pts = np.zeros((grid_size[0] * grid_size[1], 1, 2), np.float32)
    pos = 0
    for i in range(0, grid_size[1]):
        for j in range(0, grid_size[0]):
            ref_pts[pos, 0, :] = spacing * np.array([j, i]) + np.array(pad_pixels)
            pos += 1

    H, mask = cv2.findHomography(centers, ref_pts, cv2.RANSAC, 1)
    dSize = [int((num - 1) * space + 2 * pixs)
             for num, space, pixs in zip(grid_size, spacing, pad_pixels)]
    return H, dSize


# Use Homography matrix for image restoration
def calibr_undistortImg(image_in, H, dSize):
    img_out = cv2.warpPerspective(image_in, H, tuple(dSize))
    return img_out


def calibr_autoCalibration(files, grid_size, ellipse_size, spacing, pad_pixels, show_preview=False):
    objPoints = []  # 3d point in real world space
    imgPoints = []  # 2d points in image plane
    images = glob.glob(files)
    for fName in images:
        pattern_name = cv2.imread(fName)
        pattern_gray = calibr_imread2gray(fName)
        pattern_binary = calibr_ellipseThreshold(pattern_gray, ellipse_size)
        pattern_contours = calibr_contours2Circle(pattern_binary)
        detector = calibr_blobDetection()
        rets, centers, pattern_circle = calibr_findCircleGrid(pattern_name, pattern_contours, grid_size, detector)
        if not rets:
            keypoints = detector.detect(pattern_contours)
            pointsNum = grid_size[0] * grid_size[1]
            img2 = cv2.drawKeypoints(pattern_name, keypoints, 0, color=(128, 125, 186), flags=0)
            imshow(img2)
            raise ValueError('  - Calibration on findCircleGrid failed： expecting ', pointsNum,
                             ' points, but have ', len(keypoints))
        else:
            H, dSize = calibr_findHomography(grid_size, spacing, pad_pixels, centers)
        if show_preview:
            captured_img_warp = cv2.warpPerspective(pattern_circle, H, tuple(dSize))
            plt.imshow(captured_img_warp)
            plt.show()
        return H, dSize, rets

    return objPoints, imgPoints


# Create a blob detector to find circle
def calibr_blobDetection():
    # Create a Blob detector parameter setter
    params = cv2.SimpleBlobDetector_Params()

    # Filter by image intensity
    params.filterByColor = True
    params.minThreshold = 128

    # Filter by detector area
    params.filterByArea = False
    params.minArea = 60

    # Filter by circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8

    # Filter by convexity
    params.filterByConvexity = True
    params.minConvexity = 0.8

    # Filter by inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)
    return detector
