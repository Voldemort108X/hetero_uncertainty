import os
import argparse
import glob

# third party
import numpy as np
import nibabel as nib
import torch
from scipy.io import loadmat, savemat
import nrrd
from helper import *

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['TSM_BACKEND'] = 'pytorch'
import transmorph as tsm  # nopep8
from transmorph.torch.layers import SpatialTransformer
from tqdm import tqdm


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='Name of the dataset')
parser.add_argument('--test-dir', required=True, help='path to all test files')
parser.add_argument('--result-dir', required=True, help='where to save the result')
parser.add_argument('--model-motion', required=True, help='pytorch model for nonlinear registration')
parser.add_argument('--model-variance', required=True, help='pytorch model for variance estimation')
parser.add_argument('--inshape', required=True, nargs='+', type=int, help='input shape of the network e.g. (64, 64, 64) or (128, 128)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

assert args.dataset in ['Echo', 'CAMUS', 'ACDC']
inshape = tuple(args.inshape)
ndims = len(inshape)

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)



device = 'cuda'

test_files = glob.glob(os.path.join(args.test_dir, '*.mat'))
assert len(test_files) > 0

# load moving and fixed images
add_feat_axis = not args.multichannel



if ndims == 3:
    model_motion = tsm.networks.TransMorph.load(args.model_motion, device) # size = (64,64,64) inshape inside Vxm: (128, 128, 128)
elif ndims == 2:
    model_motion = tsm.networks_2D.TransMorph.load(args.model_motion, device)

model_motion.to(device)
model_motion.eval()


model_variance = tsm.networks.VarianceEstimator(inshape=inshape).load(args.model_variance, device)
model_variance.to(device)
model_variance.eval()



# print(model)

if ndims == 2:
    warp_layer = SpatialTransformer(inshape).to('cuda') # size=(128,128,128)


# for file_path in test_files:
for file_path in tqdm(test_files):
    if args.dataset == 'Echo':
        im_ED, im_ES, ED_myo, ES_myo, ED_epi, ES_epi, ED_endo, ES_endo = tsm.py.utils_echo.load_volfile_mat(file_path, add_batch_axis=True, add_feat_axis=True, registration=True)
    if args.dataset == 'CAMUS' or args.dataset == 'ACDC':
        im_ED, im_ES, ED_myo, ES_myo = tsm.py.utils_2D.load_imfile_mat(file_path, add_batch_axis=True, add_feat_axis=True, registration=True)

    if ndims == 3:
        input_ED = torch.from_numpy(im_ED).to(device).float().permute(0, 4, 1, 2, 3)
        input_ES = torch.from_numpy(im_ES).to(device).float().permute(0, 4, 1, 2, 3)
        input_ED_myo = torch.from_numpy(ED_myo).to(device).float().permute(0, 4, 1, 2, 3)


    elif ndims == 2:
        input_ED = torch.from_numpy(im_ED).to(device).float().permute(0, 3, 1, 2)
        input_ES = torch.from_numpy(im_ES).to(device).float().permute(0, 3, 1, 2)
        input_ED_myo = torch.from_numpy(ED_myo).to(device).float().permute(0, 3, 1, 2)



    im_ES_pred, dvf, mask_bgd = model_motion(input_ED, input_ES, registration=True)

    input_bank = torch.cat((im_ES_pred, input_ES), dim=1)
    logsigma_image = model_variance(input_bank)

    image_weight = tsm.losses.compute_SNR_from_variance(input_ES, logsigma_image, mask_bgd, gamma = 0.5)

    if args.dataset == 'Echo':
        # get target shape from test_noresize
        test_noresize_path = file_path.replace('test', 'test_noresize')
        file_noresize = loadmat(test_noresize_path)
        tgt_shape = [file_noresize['imshape'][0,j] for j in range(3)]

        # resize and rescale dvf
        dvf = dvf_resize(dvf.detach().cpu().numpy()[0], tgt_shape)

        # get im_ED, im_ES, ED_myo, ES_myo in original shape
        im_ED, im_ES, ED_myo, ES_myo = load_test_echo_noresize(file_noresize)

        # warp the im_ED and ED_myo to get the warped images
        warp_layer = SpatialTransformer(tgt_shape)
        input_ED_myo = torch.tensor(ED_myo, dtype=torch.get_default_dtype())[None,None,:,:,:]
        input_ED = torch.tensor(im_ED, dtype=torch.get_default_dtype())[None,None,:,:,:]
        input_dvf = torch.tensor(dvf, dtype=torch.get_default_dtype())[None,:,:,:,:]
        ES_myo_pred = warp_layer(input_ED_myo, input_dvf).detach().numpy()[0,0,:,:,:]
        im_ES_pred = warp_layer(input_ED, input_dvf).detach().numpy()[0,0,:,:,:]

        mask_bgd = (im_ED!=0) + (im_ES!=0)

    elif args.dataset == 'ACDC' or args.dataset == 'CAMUS':
        ES_myo_pred = warp_layer(input_ED_myo, dvf.to(device))
    
    if args.dataset == 'Echo':
        save_file = {
                "im_ED": im_ED, 
                "im_ES": im_ES, 
                "im_ES_pred": im_ES_pred, 
                "ED_myo": ED_myo, 
                "ES_myo":ES_myo, 
                "ES_myo_pred": ES_myo_pred, 
                "dvf": dvf, 
                "mask_bgd": mask_bgd,
                "logsigma_image": logsigma_image.detach().cpu().numpy()[0,0],
                "image_weight": image_weight.detach().cpu().numpy()[0,0]
                }
        
    elif ndims == 2:
        save_file = {
                "im_ED": im_ED[0, :, :, 0], 
                "im_ES": im_ES[0, :, :, 0], 
                "im_ES_pred": im_ES_pred.detach().cpu().numpy()[0,0], 
                "ED_myo": ED_myo[0, :, :, 0], 
                "ES_myo":ES_myo[0, :, :, 0], 
                "ES_myo_pred": ES_myo_pred.detach().cpu().numpy()[0, 0], 
                "dvf": dvf.detach().cpu().numpy()[0], 
                "mask_bgd": mask_bgd.detach().cpu().numpy()[0,0],
                "logsigma_image": logsigma_image.detach().cpu().numpy()[0,0],
                "image_weight": image_weight.detach().cpu().numpy()[0,0]
                }

    save_name = file_path.split('/')[-1]
    savemat(os.path.join(args.result_dir, save_name) ,save_file)

