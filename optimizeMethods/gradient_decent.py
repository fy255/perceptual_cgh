import sys
import torch
import torch.nn as nn
import torch.optim as optim
import utils
import propMethods.asm as asm
import IQA_pytorch as IQA
from torch.utils.tensorboard import SummaryWriter
import piq as PIQ


class gradient_decent(nn.Module):
    def __init__(self, camera, slm, HMatrix, dSize, camModel, prop_dist, wavelength, pix_pitch, device, trans_f=None,
                 trans_b=None, num_iters=20, lossName=nn.MSELoss(), prop_method='ASM', lr=0.01, lr_s=0.003, s0=1.0):
        super(gradient_decent, self).__init__()
        self.prop_dist = prop_dist
        self.wavelength = wavelength
        self.pix_pitch = pix_pitch
        self.device = device
        self.prop_method = prop_method
        self.trans_f = trans_f
        self.trans_b = trans_b

        self.num_iters = num_iters
        self.loss = lossFuncs(lossName, device)
        self.lr = lr
        self.lr_s = lr_s
        self.s0 = s0

        self.camera = camera
        self.slm = slm
        self.HMatrix = HMatrix
        self.dSize = dSize
        self.camModel = camModel
        self.writerName = 'runs/' + lossName

    def forward(self, target_amp, initial_angle):
        MSELoss = nn.MSELoss().to(self.device)
        SSIMLoss = IQA.SSIM(channels=1).to(device=self.device, dtype=torch.double)  # Init a ssim loss
        writer = SummaryWriter(self.writerName)  # Init a summary writer
        target_amp = target_amp.clone().detach().squeeze().double().to(self.device)
        self.s0 = self.s0.to(self.device).requires_grad_(True)
        slm_angle = initial_angle.to(self.device).requires_grad_(True)
        optvars = [{'params': slm_angle}]
        if self.lr_s > 0:
            optvars += [{'params': self.s0, 'lr': self.lr_s}]
        optimizer = optim.Adam(optvars, lr=self.lr)
        init_loss = nn.MSELoss().to(self.device)
        # Iterations based on the selected loss
        for k in range(self.num_iters):
            optimizer.zero_grad()
            slm_amp = torch.ones(slm_angle.shape, dtype=torch.double, device=self.device)
            slm_field = utils.polar2cart(slm_amp, slm_angle)
            recon_field = asm.ASM_prop(slm_field, linear_conv=True, tensor_trans=self.trans_f)
            out_amp = self.s0 * recon_field.abs()
            if k <= 15:
                lossValue = init_loss(out_amp, target_amp)
            else:
                lossValue = self.loss(out_amp, target_amp)
                MSELossValue = MSELoss(out_amp, target_amp)
                writer.add_scalar(self.loss.lossName, lossValue, k)
                writer.add_scalar(f'MSE loss ', MSELossValue, k)
                iterNum = '%.4f' % (k / self.num_iters * 100)
                print('finished: ', iterNum, '%')
            lossValue.backward()
            optimizer.step()

        optimized_angle = slm_angle.clone().detach()
        loss = SSIMLoss(self.s0 * out_amp.unsqueeze(0).unsqueeze(0).float(),
                        target_amp.unsqueeze(0).unsqueeze(0).float(), as_loss=False)
        return optimized_angle, MSELossValue.clone().detach().cpu().numpy(), utils.torch2np(loss), utils.torch2np(
            out_amp * 255), utils.torch2np(target_amp * 255)


class lossFuncs(nn.Module):
    def __init__(self, lossName, device):
        super(lossFuncs, self).__init__()
        self.lossName = lossName

        # TORCH ERROR VISIBILITY METHODS
        if lossName == 'MSE' or lossName is None:
            self.loss = nn.MSELoss().to(device)
        elif lossName == 'MAE' or lossName is None:
            self.loss = nn.L1Loss().to(device)

        # IQA LIBRARY
        # IQA ERROR VISIBILITY METHODS
        elif lossName == 'NLPD':
            self.loss = IQA.NLPD(channels=1).to(device=device, dtype=torch.float)

        # IQA STRUCTURAL SIMILARITY METHODS
        elif lossName == 'SSIM':
            self.loss = IQA.SSIM(channels=1).to(device=device, dtype=torch.double)
        elif lossName == 'MS_SSIM':
            self.loss = IQA.MS_SSIM(channels=1).to(device=device, dtype=torch.double)
        elif lossName == 'GMSD':
            self.loss = IQA.GMSD(channels=1).to(device=device, dtype=torch.float)

        # IQA INFORMATION THEORETICAL METHODS
        elif lossName == 'VIF':
            self.loss = IQA.VIF(channels=1).to(device=device, dtype=torch.double)
        elif lossName == 'VIFs':
            self.loss = IQA.VIFs(channels=1).to(device=device, dtype=torch.double)
        elif lossName == 'VSI':
            self.loss = IQA.VSI().to(device=device, dtype=torch.float)

        # IQA LEARNING BASED METHODS
        elif lossName == 'LPIPSvgg':
            self.loss = IQA.LPIPSvgg().to(device=device, dtype=torch.double)
        elif lossName == 'DISTS':
            self.loss = IQA.DISTS().to(device=device, dtype=torch.double)

        # PIQ LIBRARY
        # PIQ SSIM METHODS
        elif lossName == 'PIQ_SSIM':
            self.loss = PIQ.SSIMLoss(data_range=1.).to(device=device, dtype=torch.double)
        elif lossName == 'PIQ_FSIM':
            self.loss = PIQ.FSIMLoss(data_range=1., reduction='none')
        elif lossName == 'PIQ_MS_SSIM':
            self.loss: torch.Tensor = PIQ.MultiScaleSSIMLoss(data_range=1., reduction='none')
        elif lossName == 'PIQ_GMSD':
            self.loss: torch.Tensor = PIQ.GMSDLoss(data_range=1., reduction='none')
        elif lossName == 'PIQ_MS_GMSD':
            self.loss: torch.Tensor = PIQ.MultiScaleGMSDLoss(
                chromatic=False, data_range=1., reduction='none')
        elif lossName == 'PIQ_VSI':
            self.loss: torch.Tensor = PIQ.VSILoss(data_range=1.)
        elif lossName == 'PIQ_HaarPSI':
            self.loss: torch.Tensor = PIQ.HaarPSILoss(reduction='none')

        # PIQ INFORMATION THEORETICAL METHODS
        elif lossName == 'PIQ_DISTS':
            self.loss: torch.Tensor = PIQ.DISTS(reduction='none')
        elif lossName == 'PIQ_VIF':
            self.loss: torch.Tensor = PIQ.VIFLoss(sigma_n_sq=2.0, data_range=1.)
        else:
            sys.exit('Selected Loss is invalid')

    def forward(self, outAmp, targetAmp):
        if self.lossName == 'MSE' or self.lossName == 'MAE':
            return self.loss(outAmp, targetAmp)
        elif self.lossName == 'SSIM':
            outAmpNorm = outAmp.unsqueeze(0).unsqueeze(0).float()
            targetAmpNorm = targetAmp.unsqueeze(0).unsqueeze(0).float()
            output: torch.Tensor = self.loss(outAmpNorm, targetAmpNorm)
            return output
        elif self.lossName == 'DISTS':
            rgb_out_amp = torch.stack([outAmp, outAmp, outAmp], dim=0).unsqueeze(0).float()
            rgb_target_amp = torch.stack([targetAmp, targetAmp, targetAmp], dim=0).unsqueeze(0).float()
            return self.loss(rgb_out_amp, rgb_target_amp, as_loss=True)
        elif self.lossName == 'VSI':  # WORK PERFECTLY!
            rgb_out_amp = torch.stack([outAmp, outAmp, outAmp], dim=0).unsqueeze(0).float()
            rgb_target_amp = torch.stack([targetAmp, targetAmp, targetAmp], dim=0).unsqueeze(0).float()
            return self.loss(rgb_out_amp, rgb_target_amp, as_loss=True)
        elif self.lossName == 'PIQ_FSIM':
            outAmpNorm = utils.torch_img_normalize_01(outAmp)
            targetAmpNorm = utils.torch_img_normalize_01(targetAmp)
            rgb_out_amp = torch.stack([outAmpNorm, outAmpNorm, outAmpNorm], dim=0).unsqueeze(0).float()
            rgb_target_amp = torch.stack([targetAmpNorm, targetAmpNorm, targetAmpNorm], dim=0).unsqueeze(0).float()
            output: torch.Tensor = self.loss(rgb_out_amp, rgb_target_amp)
            return output
        elif 'PIQ' in self.lossName:
            # outAmpNorm = torch.tanh(outAmp).unsqueeze(0).unsqueeze(0)
            # outAmpNorm = torch.sigmoid(outAmp).unsqueeze(0).unsqueeze(0)
            outAmpNorm = utils.torch_img_normalize_01(outAmp).unsqueeze(0).unsqueeze(0)
            # targetAmpNorm = torch.sigmoid(targetAmp).unsqueeze(0).unsqueeze(0)
            targetAmpNorm = targetAmp.unsqueeze(0).unsqueeze(0)
            output: torch.Tensor = self.loss(outAmpNorm, targetAmpNorm)
            return output

        else:
            return self.loss(outAmp.unsqueeze(0).unsqueeze(0).float(), targetAmp.unsqueeze(0).unsqueeze(0).float(),
                             as_loss=True)
