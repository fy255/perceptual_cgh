import logging
import torch
import torch.nn as nn
import propMethods.asm as asm
import utils
import IQA_pytorch as IQA


class gerchberg_saxton(nn.Module):
    def __init__(self, prop_dist, wavelength, feature_size, device, phase_path=None,
                 trans_f=None, trans_b=None, num_iters=50, loss=nn.MSELoss(), prop_method='ASM'):
        super(gerchberg_saxton, self).__init__()
        self.prop_dist = prop_dist
        self.wavelength = wavelength
        self.feature_size = feature_size
        self.device = device
        self.phase_path = phase_path
        self.prop_method = prop_method
        self.trans_f = trans_f
        self.trans_b = trans_b

        self.num_iters = num_iters

    def forward(self, target_amp, initial_angle):
        MSELoss = nn.MSELoss().to(self.device)
        SSIMLoss = IQA.SSIM(channels=1).to(device=self.device, dtype=torch.double)  # Init a ssim loss

        slm_angle = initial_angle.to(self.device)
        slm_amp = torch.ones(slm_angle.shape, dtype=torch.double, device=self.device)
        slm_field = utils.polar2cart(slm_amp, slm_angle)
        target_amp = target_amp.clone().detach().squeeze().double().to(self.device)
        logging.info(f'Using gerchberg-saxton method to optimize initial phase...')
        for k in range(self.num_iters):
            recon_field = asm.ASM_prop(slm_field, linear_conv=True, tensor_trans=self.trans_f)
            recon_field = utils.replace_amp(recon_field, target_amp)
            slm_field = asm.ASM_prop(recon_field, linear_conv=True, tensor_trans=self.trans_b)
            slm_field = utils.replace_amp(slm_field, torch.ones_like(slm_field, dtype=torch.double).to(self.device))
        optimized_angle = slm_field.angle().clone().detach()
        recon_field = asm.ASM_prop(slm_field, linear_conv=True, tensor_trans=self.trans_f)
        out_amp = recon_field.abs()

        MSELossValue = MSELoss(out_amp, target_amp)

        loss = SSIMLoss(out_amp.unsqueeze(0).unsqueeze(0).float(),
                        target_amp.unsqueeze(0).unsqueeze(0).float(), as_loss=False)
        return optimized_angle, MSELossValue.clone().detach().cpu().numpy(), utils.torch2np(loss), utils.torch2np(
            out_amp * 255), utils.torch2np(target_amp * 255)
