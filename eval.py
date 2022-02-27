import math
import os
import utils
import torch
import torch.nn as nn
from propMethods import asm, slm_prop
from optimizeMethods import gerchberg_saxton, gradient_decent
import IQA_pytorch as IQA
import piq as PIQ

# Basic setting
dType = torch.double  # default datatype
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # default device

# Config
args = utils.get_args()
utils.print_params(args, device)
prop_dist = args.prop_dist[args.channel]  # single wave propagation distance
wavelength = args.wavelength[args.channel]  # single wave wavelength

image_path = os.path.join(os.getcwd(), os.path.normpath(args.image_path))  # path for image target images
result_path = os.path.join(os.getcwd(), os.path.normpath(args.result_path))  # path for generated data

for x in args.loss:
    lossName = x
    print(f'\toptimizing with loss: {lossName}')

    if args.optimize_method == 'gerchberg_saxton': lossName = args.optimize_method
    holo_path = os.path.join(result_path, 'holo', lossName)  # path for saving optimized phases
    rpf_path = os.path.join(result_path, 'rpf', lossName)  # path for saving optimized replay images
    targetAmp_path = os.path.join(result_path, 'targetAmp', lossName)  # path for saving target image amps
    capAmp_path = os.path.join(result_path, 'capAmp', lossName)  # path for saving target image amps

    if not os.path.exists(holo_path): os.mkdir(holo_path)
    if not os.path.exists(rpf_path): os.mkdir(rpf_path)
    if not os.path.exists(targetAmp_path): os.mkdir(targetAmp_path)
    if not os.path.exists(capAmp_path): os.mkdir(capAmp_path)

    imageData = utils.imageDataset(imgs_dir=image_path, channel=args.channel, image_res=args.image_res,
                                   homography_res=args.homo_res)  # create image dataset
    holoData = utils.holoDataset(imgs_dir=holo_path, image_res=args.image_res, homography_res=args.homo_res)
    rpfData = utils.imageDataset(imgs_dir=rpf_path, channel=args.channel, image_res=args.image_res,
                                 homography_res=args.homo_res)  # create image dataset

    # Quick slm update
    # holo = holoData.__getitem__(0)['holo']
    # slm = utils.slmDisplay()
    # slm.open()
    # slm.update(holo)

    # Create a txt file for saving results
    result_file_path = os.path.join(result_path, lossName + ".txt")
    f = open(result_file_path, "w+")
    f.close()

    # 1. Evaluation with/without camera
    if args.camera_model == 2:  # Evaluation without camera
        camera, slm, HMatrix, dSize = None, None, None, [1680, 960]
    else:  # Evaluation with Camera: calibration
        camera, slm, HMatrix, dSize = slm_prop.hardware_calibr(calibr_grid_size=(22, 13),
                                                               calibr_ellipse_size=(8, 8),
                                                               calibr_spacing=(80, 80),
                                                               calibr_pad_pixels=(0, 0), expoPara=None,
                                                               cameraCapture_folder='cameraCapture',
                                                               calibr_folder='PatternCalibration',
                                                               show_preview=True, camModel=args.camera_model)

    # TORCH ERROR VISIBILITY METHODS:
    L1Loss = nn.L1Loss().to(device)
    MSELoss = nn.MSELoss().to(device)

    # IQA LIBRARY
    # IQA ERROR VISIBILITY METHODS:
    IQA_NLPD_loss = IQA.NLPD(channels=1).to(device=device, dtype=torch.float)

    # IQA STRUCTURAL SIMILARITY METHODS:
    IQA_SSIM_loss = IQA.SSIM(channels=1).to(device=device, dtype=torch.double)
    IQA_MS_SSIM_loss = IQA.MS_SSIM(channels=1).to(device=device, dtype=torch.double)

    # IQA LEARNING BASED METHODS:
    IQA_LPIPSvgg_loss = IQA.LPIPSvgg().to(device=device, dtype=torch.double)

    # PIQ INFORMATION THEORETICAL METHODS:
    PIQ_DISTS_loss: torch.Tensor = PIQ.DISTS(reduction='none')

    k = 1
    for image, holo, rpf in zip(imageData, holoData, rpfData):
        # TODO: HOW TO DEAL WITH THE BATCH PROCESS?
        target_amp, target_filename = image['image'], image['image_name']
        rpf_amp = rpf['image']
        holo = holo['holo']
        target_amp = target_amp.clone().detach().squeeze().double().to(device)
        target_amp = utils.img_crop(target_amp, torch.tensor([dSize[1], dSize[0]]),
                                    padval=0, mode='constant').to(device)

        if args.camera_model == 2:
            out_amp = utils.img_crop(rpf_amp.squeeze(), torch.tensor([dSize[1], dSize[0]]),
                                     padval=0, mode='constant').to(device)
            maxVal = '%.4f' % utils.torch2np(out_amp.max())
            outMean = '%.4f' % utils.torch2np(out_amp.mean())
            targetMean = '%.4f' % utils.torch2np(target_amp.mean())
        else:
            captured_amp, img_out, cap_rgb = slm_prop.hardware_prop(camera=camera, slm=slm, slm_phase=holo, H=HMatrix,
                                                                    dSize=dSize, camModel=args.camera_model, expoPara=0,
                                                                    num_grab_images=3,
                                                                    cameraCapture_folder='cameraCapture',
                                                                    rawImg_folder='ResultImage')

            croped_captured_amp = utils.img_crop(captured_amp, torch.tensor([dSize[1], dSize[0]]),
                                                 padval=0, mode='constant').to(device)
            outMean = '%.4f' % utils.torch2np(croped_captured_amp.mean())
            targetMean = '%.4f' % utils.torch2np(target_amp.mean())
            s0 = target_amp.mean() / croped_captured_amp.mean()
            # out_amp = croped_captured_amp * s0
            out_amp = croped_captured_amp
            maxVal = '%.4f' % utils.torch2np(out_amp.max())
            print('target mean: ', targetMean, 'capture mean: ', outMean,
                  'out mean: ', '%.4f' % utils.torch2np(out_amp.mean()))

            utils.imout(capAmp_path, target_filename, croped_captured_amp)
            utils.imout(capAmp_path, target_filename+"_rgb", cap_rgb)

        IQA_out_amp = out_amp.unsqueeze(0).unsqueeze(0).float()
        IQA_out_amp_RGB = torch.stack([out_amp, out_amp, out_amp], dim=0).unsqueeze(0).float()
        IQA_target_amp = target_amp.unsqueeze(0).unsqueeze(0).float()
        IQA_target_amp_RGB = torch.stack([target_amp, target_amp, target_amp], dim=0).unsqueeze(0).float()

        L1LossVal = '%.4f' % (L1Loss(out_amp, target_amp))
        MSELossVal = '%.4f' % (MSELoss(out_amp, target_amp))
        IQA_NLPD_lossVal = '%.4f' % (IQA_NLPD_loss(IQA_out_amp, IQA_target_amp, as_loss=False))
        IQA_SSIM_lossVal = '%.4f' % (IQA_SSIM_loss(IQA_out_amp, IQA_target_amp, as_loss=False))
        IQA_MS_SSIM_lossVal = '%.4f' % (IQA_MS_SSIM_loss(IQA_out_amp, IQA_target_amp, as_loss=False))
        IQA_LPIPSvgg_lossVal = '%.4f' % (IQA_LPIPSvgg_loss(IQA_out_amp, IQA_target_amp, as_loss=False))
        PIQ_PSNR_lossVal = '%.4f' % (PIQ.psnr(IQA_out_amp, IQA_target_amp, data_range=1., reduction='none'))

        PIQ_SSIM_lossVal = '%.4f' % (PIQ.ssim(IQA_out_amp, IQA_target_amp))
        PIQ_FSIM_lossVal = '%.4f' % (PIQ.fsim(IQA_out_amp_RGB, IQA_target_amp_RGB, data_range=1., reduction='none'))
        PIQ_MS_SSIM_lossVal = '%.4f' % (PIQ.multi_scale_ssim(IQA_out_amp, IQA_target_amp))
        PIQ_GMSD_lossVal = '%.4f' % (PIQ.gmsd(IQA_out_amp, IQA_target_amp))
        PIQ_MS_GMSD_lossVal = '%.4f' % (PIQ.multi_scale_gmsd(IQA_out_amp, IQA_target_amp))
        PIQ_VSI_lossVal = '%.4f' % (PIQ.vsi(IQA_out_amp_RGB, IQA_target_amp_RGB))
        PIQ_HaarPSI_lossVal = '%.4f' % (PIQ.haarpsi(IQA_out_amp, IQA_target_amp))

        PIQ_DISTS_lossVal = '%.4f' % (PIQ_DISTS_loss(IQA_out_amp, IQA_target_amp))
        PIQ_VIF_lossVal = '%.4f' % (PIQ.vif_p(IQA_out_amp, IQA_target_amp))
        PIQ_MDSI_lossVal = '%.4f' % (PIQ.mdsi(IQA_out_amp_RGB, IQA_target_amp_RGB))

        writeOutData = [k, target_filename, maxVal, L1LossVal, PIQ_PSNR_lossVal, MSELossVal, IQA_NLPD_lossVal,
                        IQA_SSIM_lossVal,
                        IQA_MS_SSIM_lossVal, IQA_LPIPSvgg_lossVal, PIQ_SSIM_lossVal, PIQ_FSIM_lossVal,
                        PIQ_MS_SSIM_lossVal,
                        PIQ_GMSD_lossVal, PIQ_MS_GMSD_lossVal, PIQ_VSI_lossVal, PIQ_HaarPSI_lossVal, PIQ_DISTS_lossVal,
                        PIQ_VIF_lossVal, PIQ_MDSI_lossVal, outMean, targetMean]
        fileOpen = open(result_file_path, "a")
        for element in writeOutData:
            fileOpen.write(str(element) + ",")
        fileOpen.write("\n")
        k += 1
        print(IQA_SSIM_lossVal)
        print('finish one image')
        fileOpen.close()
