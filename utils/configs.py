import os
import argparse

cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9


##Project setup
def get_args():
    parser = argparse.ArgumentParser(description='Perceptual CGH model Implementation')

    # Project settings
    parser.add_argument('--loss', type=list, default=['MS_SSIM'], help='type of perceptual losses')

    parser.add_argument('--prop_method', type=str, default='asm', help='type of propagation method')
    parser.add_argument('--optimize_method', type=str, default='gradient_decent', help='type of optimization method')
    parser.add_argument('--image_path', type=str, default='./data/image', help='directory of target image')
    parser.add_argument('--result_path', type=str, default='./data/temp', help='directory of saved data')
    parser.add_argument('--camera_model', default=2, type=int, help='Hikivison:0, Canon:1, no camera:2')

    # CGH parameters
    parser.add_argument('--channel', default=1, type=int, help='red:0, green:1, blue:2, rgb:3')
    parser.add_argument('--prop_dist', type=tuple, default=(15 * cm, 15 * cm, 15 * cm), help='propagation distance')
    parser.add_argument('--pix_pitch', type=tuple, default=(6.4 * um, 6.4 * um), help='pixel pitch x and y')
    parser.add_argument('--wavelength', type=tuple, default=(638 * nm, 532 * nm, 450 * nm), help='RGB wavelength')
    parser.add_argument('--slm_res', type=tuple, default=(1080, 1920), help='SLM resolution')
    parser.add_argument('--image_res', type=tuple, default=(1080, 1920), help='target image resolution')
    parser.add_argument('--homo_res', type=tuple, default=(880, 1600),
                        help='homography resolution for camera calibration')

    # Algorithm parameters
    parser.add_argument('--num_iters', type=int, default=500, help='iteration number for SGD')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate for phase optimization')
    parser.add_argument('--s0', type=float, default=1., help='initial image scale')
    parser.add_argument('--lr_s', type=float, default=0.05, help='learning rate for image scale')

    return parser.parse_args()


##Parameter display
def print_params(args, device):
    prop_dist = args.prop_dist[args.channel]  # single wave propagation distance
    wavelength = args.wavelength[args.channel]  # single wave wavelength
    chan_str = ('red', 'green', 'blue')[args.channel]

    # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print(f'Using device {device}')
    print(f'Parameters:\n'
          f'\tselected wavelength: {wavelength} \n'
          f'\tprop method: {args.prop_method} \n'
          f'\toptimize method:{args.optimize_method} \n'
          f'\tselected wavelength: {wavelength} \n'
          f'\tpropagating distance: {prop_dist} \n'
          f'\tselected colour: {chan_str} \n'
          f'\tpixel pitch: {args.pix_pitch}\n ')
