import torch
import torch.nn.functional as F
import numpy as np
import math



class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred, mask_bgd):
        y_true = torch.mul(y_true, mask_bgd)
        y_pred = torch.mul(y_pred, mask_bgd)
        return torch.mean((y_true - y_pred) ** 2)



class WeightedMSE:
    """
    Weighted mean squared error loss.
    """

    def __init__(self, loss_mult=None):
        self.loss_mult = loss_mult

    def loss(self, y_true, y_pred, image_weight, mask_bgd):

        image_weight = image_weight.detach()

        y_true = torch.mul(y_true, mask_bgd)
        y_pred = torch.mul(y_pred, mask_bgd)
        image_weight = torch.mul(image_weight, mask_bgd)


        return torch.mean(torch.mul((y_true - y_pred) ** 2, image_weight))



class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred, logsigma_image, mask_bgd): # log_var should not be used for flow regularization
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult # to compensate the 
        return grad

class Grad_2d:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred, logsigma_image, mask_bgd): # log_var should not be used for flow regularization
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult # to compensate the 
        return grad
    

class VarianceNLL:
    """N-D adaptive loss B x C x H x W x (D)
    NLL: https://proceedings.neurips.cc/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf
    beta-NLL: https://github.com/martius-lab/beta-nll#citation
    """
    def __init__(self, beta,  flag_laplace=False):
        self.beta = beta
        self.flag_laplace = flag_laplace

    def loss(self, y_true, y_pred, logsigma_image, mask_bgd):
        assert y_true.size() == y_pred.size() and y_true.size() == logsigma_image.size()


        y_true = y_true.detach()
        y_pred = y_pred.detach()

        y_true = torch.mul(y_true, mask_bgd)
        y_pred = torch.mul(y_pred, mask_bgd)
        logsigma_image = torch.mul(logsigma_image, mask_bgd)

        if self.flag_laplace == True:
            error = 0.5 * torch.mul(torch.abs(y_true - y_pred), torch.exp(-logsigma_image)) + 0.5 * logsigma_image
        else:
            error = 0.5 * torch.mul(torch.square(y_true - y_pred), torch.exp(-logsigma_image)) + 0.5 * logsigma_image

        if self.beta > 0:
            error = error * (torch.exp(logsigma_image.detach()) ** self.beta)

        # normalization = np.prod([y_true.size(dim=i) for i in range(y_true.dim())])

        return torch.mean(error) #torch.sum(error) / normalization



def compute_SNR_from_variance(y_true, logsigma_image, mask_bgd, gamma = 0.5):
    assert y_true.size() == logsigma_image.size()

    y_true = torch.mul(y_true, mask_bgd)
    logsigma_image = torch.mul(logsigma_image, mask_bgd)
    
    rel_snr = y_true / torch.exp(logsigma_image)
    rel_snr = rel_snr ** gamma

    rel_snr = (rel_snr - torch.min(rel_snr))/(torch.max(rel_snr) - torch.min(rel_snr))
    rel_snr = torch.sigmoid(rel_snr)

    return rel_snr



###############################################
# NLL loss for both motion and variance losses
###############################################
class NLLloss:
    """N-D adaptive loss B x C x H x W x (D)
    NLL: https://proceedings.neurips.cc/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf
    beta-NLL: https://github.com/martius-lab/beta-nll#citation
    """
    def __init__(self, beta,  flag_laplace=False):
        self.beta = beta
        self.flag_laplace = flag_laplace

    def loss(self, y_true, y_pred, logsigma_image, mask_bgd):
        assert y_true.size() == y_pred.size() and y_true.size() == logsigma_image.size()


        y_true = torch.mul(y_true, mask_bgd)
        y_pred = torch.mul(y_pred, mask_bgd)
        logsigma_image = torch.mul(logsigma_image, mask_bgd)

        if self.flag_laplace == True:
            error = 0.5 * torch.mul(torch.abs(y_true - y_pred), torch.exp(-logsigma_image)) + 0.5 * logsigma_image
        else:
            error = 0.5 * torch.mul(torch.square(y_true - y_pred), torch.exp(-logsigma_image)) + 0.5 * logsigma_image

        if self.beta > 0:
            error = error * (torch.exp(logsigma_image.detach()) ** self.beta)

        # normalization = np.prod([y_true.size(dim=i) for i in range(y_true.dim())])

        return torch.mean(error) #torch.sum(error) / normalization
