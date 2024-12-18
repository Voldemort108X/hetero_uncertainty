# uncertainty_metrics = ["abs_rel", "rmse", "a1"]
import numpy as np
from medpy import metric

uncertainty_metrics = ["mse"]

def compute_eigen_errors_motion(gt, pred, metrics=uncertainty_metrics, mask=None, reduce_mean=False):
    results = []

    if mask is not None:
        # print(pred['im_pred'].shape)
        # print('mask',mask.shape)
        pred = pred[mask]
        gt = gt[mask]

    # else:
    #     pred_im = pred['im_pred']
    #     gt_im = gt['im_gt']
    #     pred_seg = pred['seg_pred']
    #     gt_seg = gt['seg_gt']

    if "mse" in metrics:
        mse = (pred - gt) ** 2
        if reduce_mean:
            mse = mse.mean()
        results.append(mse)

    return results


def compute_aucs(gt, pred, uncert, intervals=50):
    """Computation of auc metrics
    """
    
    # results dictionaries
    # AUSE = {"abs_rel":0, "rmse":0, "a1":0}
    # AURG = {"abs_rel":0, "rmse":0, "a1":0}
    AUSE = {"mse":0}
    AURG = {"mse":0}

    # revert order (high uncertainty first)
    uncert = -uncert
    true_uncert = compute_eigen_errors_motion(gt,pred)
    true_uncert = {"mse":-true_uncert[0]}

    # prepare subsets for sampling and for area computation
    quants = [100./intervals*t for t in range(0,intervals)]
    plotx = [1./intervals*t for t in range(0,intervals+1)]

    # get percentiles for sampling and corresponding subsets
    thresholds = [np.percentile(uncert, q) for q in quants]
    subs = [(uncert >= t) for t in thresholds]

    # compute sparsification curves for each metric (add 0 for final sampling)
    # print('test shape before',pred['im_pred'].shape)
    # for m in uncertainty_metrics:
    #     print(m)
    # print(len(subs))
    # for sub in subs:
    #     print(sub.shape)
    sparse_curve = {m:[compute_eigen_errors_motion(gt,pred,metrics=[m],mask=sub,reduce_mean=True)[0] for sub in subs]+[0] for m in uncertainty_metrics }

    # print(sparse_curve['mse'].shape)

    # human-readable call
    '''
    sparse_curve =  {"rmse":[compute_eigen_errors_v2(gt,pred,metrics=["rmse"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0], 
                     "a1":[compute_eigen_errors_v2(gt,pred,metrics=["a1"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0],
                     "abs_rel":[compute_eigen_errors_v2(gt,pred,metrics=["abs_rel"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0]}
    '''
    
    # get percentiles for optimal sampling and corresponding subsets
    opt_thresholds = {m:[np.percentile(true_uncert[m], q) for q in quants] for m in uncertainty_metrics}
    opt_subs = {m:[(true_uncert[m] >= o) for o in opt_thresholds[m]] for m in uncertainty_metrics}

    # compute sparsification curves for optimal sampling (add 0 for final sampling)
    opt_curve = {m:[compute_eigen_errors_motion(gt,pred,metrics=[m],mask=opt_sub,reduce_mean=True)[0] for opt_sub in opt_subs[m]]+[0] for m in uncertainty_metrics}

    # compute metrics for random sampling (equal for each sampling)
    rnd_curve = {m:[compute_eigen_errors_motion(gt,pred,metrics=[m],mask=None,reduce_mean=True)[0] for t in range(intervals+1)] for m in uncertainty_metrics}    

    # compute error and gain metrics
    for m in uncertainty_metrics:

        # error: subtract from method sparsification (first term) the oracle sparsification (second term)
        AUSE[m] = np.trapz(sparse_curve[m], x=plotx) - np.trapz(opt_curve[m], x=plotx)
        
        # gain: subtract from random sparsification (first term) the method sparsification (second term)
        AURG[m] = rnd_curve[m][0] - np.trapz(sparse_curve[m], x=plotx)

    # returns a dictionary with AUSE and AURG for each metric
    return {m:[AUSE[m], AURG[m]] for m in uncertainty_metrics}, sparse_curve, opt_curve, plotx