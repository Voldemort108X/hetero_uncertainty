#!/usr/bin/env python


import os
import random
import argparse
import time
import numpy as np
import torch
import glob


# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8

# wandb for tracking experiments
import wandb

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
# parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
parser.add_argument('--dataset', required=True, help='Name of the dataset')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', required=True,
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')


# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of training epochs (default: 150)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model-motion', help='optional motion model file to initialize with')
parser.add_argument('--load-model-variance', help='optional variance model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')
parser.add_argument('--save_freq', type=int, default=20, 
                    help='save frequency')
parser.add_argument('--warm_start_epoch', type=int, default=10, 
                    help='number of epochs to warm start the network')


# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')


# loss selection
parser.add_argument('--motion-loss-type', required=True,
                    help='image reconstruction loss - can be mse or ncc or gc or adp (default: mse)')
parser.add_argument('--variance-loss-type', required=True,
                    help='image reconstruction loss - can be mse or ncc or gc or adp (default: mse)')


# loss hyperparameters
parser.add_argument('--beta', default=0.5, type=float, help='interpolation parameter for negative log likelihood (NLL)')
parser.add_argument('--gamma', default=0.5, type=float, help='weighting exponential of the snr-like map')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')

# gradient accumulation
parser.add_argument('--accumulation_steps', type=int, default=1, help='number of steps before backward and optimizer step')

# wandb run name
parser.add_argument('--use_wandb', action='store_true', help='use wandb for tracking experiments')
parser.add_argument('--wandb-name', type=str, default=None, help='name of wandb run (optional)')
parser.add_argument('--wandb-project', type=str, default=None, help='name of wandb project (optional)')

# flags for ablation experiments
parser.add_argument('--warm_start', action='store_true', help='warm start the network by first training x epochs for motion esimator and then x epochs for variance estimator')
parser.add_argument('--flag_laplace', action='store_true', help='Default is false, whether to use Laplacian assumption for noise')

args = parser.parse_args()

bidir = args.bidir

assert args.dataset in ['Echo', 'CAMUS', 'ACDC']
print(os.listdir('../../'))
train_files = glob.glob(os.path.join('../../Dataset/', args.dataset, 'train/*.mat')) + glob.glob(os.path.join('../../Dataset', args.dataset, 'val/*.mat'))
assert len(train_files) > 0, 'Could not find any training data.'



# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

# compute the real batch size needed
reduced_batch_size = int(args.batch_size / args.accumulation_steps)


if args.dataset == 'Echo':
    generator = vxm.generators_echo.scan_to_scan_echo(
        train_files, batch_size=reduced_batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)
if args.dataset == 'ACDC' or args.dataset == 'CAMUS':
    generator = vxm.generators_2D.scan_to_scan_2D(
        train_files, batch_size=reduced_batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)

# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]
ndims = len(inshape)

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model_motion:
    # load initial model (if specified)
    model_motion = vxm.networks.VxmDense.load(args.load_model, device)
else:
    # otherwise configure new model
    model_motion = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )

if args.load_model_variance:
    model_variance = vxm.networks.VarianceEstimator.load(args.load_model_variance, device)
else:
    model_variance = vxm.networks.VarianceEstimator(inshape, nb_unet_features=[enc_nf, dec_nf])


# prepare the model for training and send to device
model_motion.to(device)
model_motion.train()

model_variance.to(device)
model_variance.train()

# set optimizer
optimizer_motion = torch.optim.Adam(model_motion.parameters(), lr=args.lr)
optimizer_variance = torch.optim.Adam(model_variance.parameters(), lr=args.lr)

# prepare the motion loss
if args.motion_loss_type == 'mse':
    loss_func_motion = vxm.losses.MSE().loss
elif args.motion_loss_type == 'wmse':
    loss_func_motion = vxm.losses.WeightedMSE().loss
else:
    raise ValueError('Image loss should be "mse" or "wmse", but found "%s"' % args.motion_loss_type)

# prepare the variance loss
if args.variance_loss_type == 'beta-NLL':
    # loss_func_variance = vxm.losses.betaNLL(args.beta).loss
    loss_func_variance = vxm.losses.VarianceNLL(beta=args.beta).loss

# need two image loss functions if bidirectional
if bidir:
    losses_motion = [loss_func_motion, loss_func_motion]
    weights_motion = [0.5, 0.5]

    losses_variance = [loss_func_variance, loss_func_variance]
    weights_variance = [0.5, 0.5]
else:
    losses_motion = [loss_func_motion]
    weights_motion = [1]

    losses_variance = [loss_func_variance]
    weights_variance = [1]

# prepare deformation loss
if ndims == 3:
    losses_motion += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
elif ndims == 2:
    losses_motion += [vxm.losses.Grad_2d('l2', loss_mult=args.int_downsize).loss]
else:
    raise NotImplementedError('Only 2D and 3D supported')

weights_motion += [args.weight]

warm_start_epoch = args.warm_start_epoch if args.warm_start == True else 0

# wandb tracking experiments
if args.use_wandb:
    run = wandb.init(
        # set the wandb project where this run will be logged
        project=args.wandb_project,
        dir='../../Logs',
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "architecture": "UNet",
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "bidir": args.bidir,
        "enc_nf": enc_nf,
        "dec_nf": dec_nf,
        
        "motion-loss": args.motion_loss_type,
        "variance-loss": args.variance_loss_type,
        "motion_loss_weights": weights_motion,
        "accumulation_steps": args.accumulation_steps,
        "warm_start": args.warm_start
        },

        # have a run name
        name = args.wandb_name
    )


# training loops
for epoch in range(args.initial_epoch, args.epochs):

    # warm start flags for motion and variance 
    if args.warm_start == True:
        if epoch < warm_start_epoch:
            flag_motion = True
            flag_variance = False
        elif epoch >= warm_start_epoch and epoch < 2 * warm_start_epoch:
            flag_motion = False
            flag_variance = True
        else:
            flag_motion = True
            flag_variance = True
    else:
        flag_motion = True
        flag_variance = True

    # save model checkpoint
    if epoch % args.save_freq == 0:
        # wandb.save(os.path.join(wandb.run.dir, '%04d.pt' % epoch))
        model_motion.save(os.path.join(model_dir, 'motion_%04d.pt' % epoch))
        model_variance.save(os.path.join(model_dir, 'variance_%04d.pt' % epoch))


    epoch_motion_loss, epoch_variance_loss = [], []
    epoch_motion_total_loss, epoch_variance_total_loss = [], []
    epoch_step_time = []

    # track gpu memory for batch accumulation experiment

    for step in range(args.steps_per_epoch):

        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true = next(generator)

        if ndims == 3:
            inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
            y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
        elif ndims == 2:
            inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in inputs]
            y_true = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in y_true]

       
        #########################################################
        # run the forward pass of motion estimator from t to t+1
        #########################################################
        if flag_motion == True:

            # run motion estimator forward pass at time t
            y_pred = model_motion(*inputs)

            # compute motion loss at time t
            motion_loss = 0
            motion_loss_list = []

            image_weight_vis = [] # check the visual of image_weight
            for n, loss_function in enumerate(losses_motion):

                # when computing the data term for the motion loss, use the predicted variance
                if n < len(losses_motion) - 1:

                    # run forward pass from predictive variance model
                    input_bank = torch.cat((y_pred[n], y_true[n]), dim=1)
                    logsigma_image = model_variance(input_bank)
                    # print(logsigma_image.shape) # B x 1 x H x W x D

                    image_weight = vxm.losses.compute_SNR_from_variance(y_true[n], logsigma_image, y_pred[-1]) # （y_true, logsigma_image, mask_bgd）

                    image_weight_vis.append(image_weight)
  
                
                    if flag_variance == False:
                        # first few epochs the variance estimator is not working
                        image_weight = torch.ones_like(image_weight)
                        # print('image weight',image_weight.shape) # B x 1 x H x W x D



                curr_loss = loss_function(y_true[n], y_pred[n], image_weight, y_pred[-1]) * weights_motion[n]
                
                motion_loss_list.append(curr_loss.item())
                motion_loss += curr_loss

            # backpropagate and optimize t -> t+1
            motion_loss.backward()

            if (step + 1) % args.accumulation_steps == 0:
                optimizer_motion.step()
                optimizer_motion.zero_grad()

            epoch_motion_loss.append(motion_loss_list)
            epoch_motion_total_loss.append(motion_loss.item())

        #########################################################
        # run the forward pass of variance estimator from t to t+1
        #########################################################
        if flag_variance == True:

            # run motion estimator at t+1
            y_pred = model_motion(*inputs)

            # compute variance loss
            variance_loss = 0
            variance_loss_list = []

            logsigma_image_vis = [] # save the forward and backward pass of logsigma_image for visualization
            for n, loss_function in enumerate(losses_variance):
                input_bank = torch.cat((y_pred[n], y_true[n]), dim=1)
                logsigma_image = model_variance(input_bank)

                logsigma_image_vis.append(logsigma_image)

                curr_loss = loss_function(y_true[n], y_pred[n], logsigma_image, y_pred[-1]) * weights_motion[n]

                variance_loss_list.append(curr_loss.item())
                variance_loss += curr_loss
            
            # backpropagate and optimize t -> t+1
            variance_loss.backward()

            if (step + 1) % args.accumulation_steps == 0:
                optimizer_variance.step()
                optimizer_variance.zero_grad()

            epoch_variance_loss.append(variance_loss_list)
            epoch_variance_total_loss.append(variance_loss.item())

         # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    # track gpu memory
    memory_allocated = torch.cuda.memory_allocated(torch.cuda.current_device())
    memory_reserved = torch.cuda.memory_reserved(torch.cuda.current_device())


    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    memory_info = 'GPU Memory Allocated: %.2f MB, Reserved: %.2f MB' % (memory_allocated / (1024**2), memory_reserved / (1024**2))

    # for motion info
    if flag_motion:
        losses_info_motion = ', '.join(['%.4e' % f for f in np.mean(epoch_motion_loss, axis=0)])
        loss_info_motion = 'motion loss: %.4e  (%s)' % (np.mean(epoch_motion_total_loss), losses_info_motion)
        print(' - '.join((epoch_info, time_info, loss_info_motion, memory_info)), flush=True)

    # for variance info
    if flag_variance:
        losses_info_variance = ', '.join(['%.4e' % f for f in np.mean(epoch_variance_loss, axis=0)])
        loss_info_variance = 'variance loss: %.4e  (%s)' % (np.mean(epoch_variance_total_loss), losses_info_variance)
        print(' - '.join((epoch_info, time_info, loss_info_variance, memory_info)), flush=True)
        # print(f"GPU Memory Allocated: {memory_allocated / (1024**2):.2f} MB")
        # print(f"GPU Memory Reserved: {memory_reserved / (1024**2):.2f} MB")

    

    if flag_motion and flag_variance and args.use_wandb:
        # wandb visualization 

        if ndims == 3:
            z_idx = 32

            src_im = wandb.Image(inputs[0].cpu().detach().numpy()[0, 0][:, :, z_idx])
            tgt_im = wandb.Image(inputs[1].cpu().detach().numpy()[0, 0][:, :, z_idx])
            pred_tgt = wandb.Image(y_pred[0].cpu().detach().numpy()[0, 0][:, :, z_idx])

            logsigma_image_fwd = wandb.Image(logsigma_image_vis[0].cpu().detach().numpy()[0, 0][:, :, z_idx])
            image_weight_fwd = wandb.Image(image_weight_vis[0].cpu().detach().numpy()[0, 0][:, :, z_idx])


            variance_image_fwd = wandb.Image(np.multiply(np.exp(logsigma_image_vis[0].cpu().detach().numpy()[0, 0]), y_pred[-1].cpu().detach().numpy()[0, 0])[:, :, z_idx])
            
            if args.bidir:
                pred_src = wandb.Image(y_pred[1].cpu().detach().numpy()[0, 0,][:, :, z_idx])
                logsigma_image_bwd = wandb.Image(logsigma_image_vis[1].cpu().detach().numpy()[0, 0][:, :, z_idx])
                image_weight_bwd = wandb.Image(image_weight_vis[1].cpu().detach().numpy()[0, 0][:, :, z_idx])


                variance_image_bwd = np.multiply(np.exp(logsigma_image_vis[1].cpu().detach().numpy()[0, 0]), y_pred[-1].cpu().detach().numpy()[0, 0])
                # print(variance_image_bwd.shape)
                variance_image_bwd = wandb.Image(variance_image_bwd[:, :, z_idx])

            mask_bgd = wandb.Image(y_pred[-1].cpu().detach().numpy()[0, 0][:, :, z_idx])
        

        if ndims == 2:
            src_im = wandb.Image(inputs[0].cpu().detach().numpy()[0, 0])
            tgt_im = wandb.Image(inputs[1].cpu().detach().numpy()[0, 0])
            pred_tgt = wandb.Image(y_pred[0].cpu().detach().numpy()[0, 0])
            
            logsigma_image_fwd = wandb.Image(logsigma_image_vis[0].cpu().detach().numpy()[0, 0])
            image_weight_fwd = wandb.Image(image_weight_vis[0].cpu().detach().numpy()[0, 0])

            # weight_sigma_fwd = wandb.Image(weight_sigma_vis[0].cpu().detach().numpy()[0, 0])
            # weight_I_fwd = wandb.Image(weight_I_vis[0].cpu().detach().numpy()[0, 0])

            variance_image_fwd = wandb.Image(np.multiply(np.exp(logsigma_image_vis[0].cpu().detach().numpy()[0, 0]), y_pred[-1].cpu().detach().numpy()[0, 0]))


            if args.bidir:
                pred_src = wandb.Image(y_pred[1].cpu().detach().numpy()[0, 0,])
                logsigma_image_bwd = wandb.Image(logsigma_image_vis[1].cpu().detach().numpy()[0, 0])
                image_weight_bwd = wandb.Image(image_weight_vis[1].cpu().detach().numpy()[0, 0])

                variance_image_bwd = wandb.Image(np.multiply(np.exp(logsigma_image_vis[1].cpu().detach().numpy()[0, 0]), y_pred[-1].cpu().detach().numpy()[0, 0]))

            mask_bgd = wandb.Image(y_pred[-1].cpu().detach().numpy()[0, 0])



        # track using wandb
        log_dict = {"motion_loss": np.mean(epoch_motion_total_loss), 
                    "variance_loss": np.mean(epoch_variance_total_loss),
                    "motion_fwd": np.mean(epoch_motion_loss, axis=0)[0], 
                    "variance_fwd": np.mean(epoch_variance_loss, axis=0)[0], 
                    "reg": np.mean(epoch_motion_loss, axis=0)[-1], 
                    "src_im":src_im, 
                    "tgt_im": tgt_im, 
                    "pred_tgt": pred_tgt, 
                    "logsigma_image_fwd": logsigma_image_fwd,
                    "logsigma_image_fwd_mean": np.mean(logsigma_image_vis[0].cpu().detach().numpy()[0, 0]),
                    "image_weight_fwd": image_weight_fwd,
                    "image_weight_fwd_mean": np.mean(image_weight_vis[0].cpu().detach().numpy()[0, 0]),
                    "mask_bgd": mask_bgd
                    }

        if args.bidir:
            log_dict['motion_bwd'] = np.mean(epoch_motion_loss, axis=0)[1]
            log_dict["variance_bwd"] = np.mean(epoch_variance_loss, axis=0)[1]
            log_dict['pred_src'] = pred_src
            log_dict['logsigma_image_bwd'] = logsigma_image_bwd
            log_dict['logsigma_image_bwd_mean'] = np.mean(logsigma_image_vis[1].cpu().detach().numpy()[0, 0])
            log_dict['image_weight_bwd'] = image_weight_bwd
            log_dict['image_weight_bwd_mean'] = np.mean(image_weight_vis[1].cpu().detach().numpy()[0, 0])

        wandb.log(log_dict)


# save everything for the last epoch
model_motion.save(os.path.join(model_dir, 'vxm_{}_motion.pt'.format(str(args.dataset))))
model_variance.save(os.path.join(model_dir, 'vxm_{}_variance.pt'.format(str(args.dataset))))





