import math
import torch
import numpy as np
import utils.image_processing as ip
import utils.torch_complex as tc
import torch.fft as fft


def ASM_precal(slm_res, pix_pitch, wavelength, prop_dist, device,
               linear_conv=True, dtype=torch.float64):
    input_resolution = slm_res
    if linear_conv:
        input_resolution = [i * 2 for i in input_resolution]
    num_y, num_x = input_resolution[0], input_resolution[1]
    dy, dx = pix_pitch[0], pix_pitch[1]
    x, y = dx * num_x, dy * num_y
    xx = np.linspace(-(num_x / 2), num_x / 2, num_x)
    yy = np.linspace(-(num_y / 2), num_y / 2, num_y)
    fy = yy / y
    fx = xx / x
    FX, FY = np.meshgrid(fx, fy)

    H_angle = 2 * np.pi * np.sqrt(1 / wavelength ** 2 - (FX ** 2 + FY ** 2))
    H_angle_z = H_angle * prop_dist

    # Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields
    # - Kyoji Matsushima and Tomoyoshi Shimobaba (2009)
    fy_max = 1. / np.sqrt((2 * prop_dist * (1. / y)) ** 2 + 1) // wavelength
    fx_max = 1. / np.sqrt((2 * prop_dist * (1. / x)) ** 2 + 1) // wavelength
    H_amp = (abs(FX) < fx_max) & (abs(FY) < fy_max)

    trans_f = np.fft.ifftshift(np.multiply(H_amp, np.exp(1j * H_angle_z)))
    trans_b = np.fft.ifftshift(np.multiply(H_amp, np.exp(-1j * H_angle_z)))
    tensor_trans_f = tc.np2tensor_complex(trans_f)
    tensor_trans_b = tc.np2tensor_complex(trans_b)

    return tensor_trans_f.to(device), tensor_trans_b.to(device)


def ASM_prop(slm_field, linear_conv, tensor_trans):
    if linear_conv:
        # preprocess with padding for linear conv.
        input_resolution = torch.tensor(slm_field.size())
        conv_size = torch.multiply(input_resolution, 2)
        padval = 0
        u_in = ip.img_pad(slm_field, conv_size, padval)
    else:
        input_resolution = slm_field.size()
        u_in = slm_field

    H = tensor_trans
    U1 = fft.fftn(tc.torch_ifftshift(u_in))  # angular spectrum
    U2 = H * U1  # system convolution
    u_out = tc.torch_fftshift(fft.ifftn(U2))  # Fourier transform of the convolution to the observation plane
    if linear_conv:
        return ip.img_crop(u_out, input_resolution)
    else:
        return u_out
