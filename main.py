import math
import os
import utils
import torch
import torch.nn as nn
from propMethods import asm, slm_prop
from optimizeMethods import gerchberg_saxton, gradient_decent

# Basic setting
dType = torch.double  # default datatype
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # default device

if __name__ == '__main__':

    # Config
    args = utils.get_args()
    utils.print_params(args, device)
    prop_dist = args.prop_dist[args.channel]  # single wave propagation distance
    wavelength = args.wavelength[args.channel]  # single wave wavelength

    image_path = os.path.join(os.getcwd(), os.path.normpath(args.image_path))  # path for image target images
    result_path = os.path.join(os.getcwd(), os.path.normpath(args.result_path))  # path for generated data
    if not os.path.exists(result_path): os.mkdir(result_path)

    # Iterate over all listed losses
    for x in args.loss:
        lossName = x
        print(f'\toptimizing with loss: {lossName}')

        if args.optimize_method == 'gerchberg_saxton': lossName = args.optimize_method
        holo_path = os.path.join(result_path, 'holo', lossName)  # path for saving optimized phases
        rpf_path = os.path.join(result_path, 'rpf', lossName)  # path for saving optimized replay images
        targetAmp_path = os.path.join(result_path, 'targetAmp', lossName)  # path for saving target image amps
        if not os.path.exists(holo_path): os.mkdir(holo_path)
        if not os.path.exists(rpf_path): os.mkdir(rpf_path)
        if not os.path.exists(targetAmp_path): os.mkdir(targetAmp_path)

        imageData = utils.imageDataset(imgs_dir=image_path, channel=args.channel, image_res=args.image_res,
                                       homography_res=args.homo_res)  # create image dataset

        init_angle = (2 * math.pi * torch.ones(args.slm_res, dtype=dType) - math.pi)  # initial hologram phase

        # 1. Precompute ASM forward and backward transfer functions:
        if args.prop_method == 'asm':
            trans_f, trans_b = asm.ASM_precal(args.slm_res, args.pix_pitch, wavelength, prop_dist,
                                              linear_conv=True, device=device)

        # 2. Optimization with/without camera
        if args.camera_model == 2:  # Optimization without camera
            camera, slm, HMatrix, dSize = None, None, None, [1680, 960]
        else:  # Optimization with Camera: calibration
            camera, slm, HMatrix, dSize = slm_prop.hardware_calibr(calibr_grid_size=(22, 13),
                                                                   calibr_ellipse_size=(8, 8),
                                                                   calibr_spacing=(80, 80),
                                                                   calibr_pad_pixels=(0, 0), expoPara=None,
                                                                   cameraCapture_folder='cameraCapture',
                                                                   calibr_folder='PatternCalibration',
                                                                   show_preview=True, camModel=args.camera_model)

        # 3. Select optimization methods:
        if args.optimize_method == 'gerchberg_saxton':
            phase_only_algorithm = gerchberg_saxton(prop_dist, wavelength, args.pix_pitch, device, phase_path=None,
                                                    trans_f=trans_f, trans_b=trans_b, num_iters=args.num_iters,
                                                    loss=nn.MSELoss(), prop_method='ASM')

        if args.optimize_method == 'gradient_decent':
            phase_only_algorithm = gradient_decent(camera, slm, HMatrix, dSize, args.camera_model, prop_dist,
                                                   wavelength, args.pix_pitch, device, dType=dType, trans_f=trans_f,
                                                   trans_b=trans_b, num_iters=args.num_iters, lossName=lossName,
                                                   prop_method='ASM', lr=args.lr, lr_s=args.lr_s, s0=args.s0)
        # 4. Iterate over image dataset
        for target in imageData:
            target_amp, name = target['image'], target['image_name']
            print(f'start  image:{name} ')

            phase_only_algorithm.s0 = args.s0 * target_amp.mean()
            final_phase, MSEValue, SSIMValue, rpfAmp, targetAmp = phase_only_algorithm(target_amp, init_angle)
            phase_out_8bit = utils.angle2map(final_phase)

            # Output data
            utils.imout(holo_path, name, phase_out_8bit)
            utils.imout(rpf_path, name, rpfAmp)
            utils.imout(targetAmp_path, name, targetAmp)
            print(f'\tfinish one image with MSE loss:{MSEValue:.4f} and SSIM loss: {SSIMValue:.4f}')
