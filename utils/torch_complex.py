import numpy as np
import torch

eps = 2.2204e-16


##Tensor operation

def np2tensor_complex(complex_num, dtype):
    # numpy complex to torch complex
    real = torch.tensor(complex_num.real, dtype=dtype)
    imag = torch.tensor(complex_num.imag, dtype=dtype)
    tensor_complex = torch.complex(real, imag)
    return tensor_complex


def replace_amp(field, amplitude):
    # replace the amplitude of a complex number
    input_phase = field.angle()
    out_field = torch.polar(amplitude, input_phase)
    return out_field


def cart2polar(real, imag):
    # Converts the polar form to polar form
    mag = torch.pow(real ** 2 + imag ** 2, 0.5)
    ang = torch.atan2(imag, real)
    return torch.polar(mag, ang)


def polar2cart(mag, ang):
    # Converts the polar form to cartesian form
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return torch.complex(real, imag)


def torch_fftshift(arg_in):
    # torch fftshift
    real = arg_in.real
    imag = arg_in.imag
    for dim in range(0, len(real.size())):
        real = torch.roll(real, dims=dim, shifts=real.size(dim) // 2)
        imag = torch.roll(imag, dims=dim, shifts=imag.size(dim) // 2)
    arg_out = torch.complex(real, imag)
    return arg_out


def torch_ifftshift(arg_in):
    # torch ifftshift
    real = arg_in.real
    imag = arg_in.imag
    for dim in range(0, len(real.size())):
        real = torch.roll(real, dims=dim, shifts=-real.size(dim) // 2)
        imag = torch.roll(imag, dims=dim, shifts=-imag.size(dim) // 2)
    arg_out = torch.complex(real, imag)

    return arg_out


def angle2map(phasemap):
    # continuous phase angle to 8 bit
    out_slm = phasemap.cpu().detach().squeeze().numpy() + 2 * np.pi
    out_slm = np.remainder(out_slm, 2 * np.pi) / (2 * np.pi)
    phase_out_8bit = (out_slm * 255).round().astype(np.uint8)  # quantized to 8 bits

    return phase_out_8bit


def torch2np(tensor_in):
    return tensor_in.cpu().detach().numpy().squeeze()
