from medpy import metric
from scipy.io import loadmat
import os

def func_computeSegMetrics3D(pred, gt):

    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd


def compute_metric(test_dir):
    dice_list, jc_list, hd_list, asd_list = [], [], [], []

    for file_name in os.listdir(test_dir):
        file = loadmat(os.path.join(test_dir, file_name))
        try:
            dice, jc, hd, asd = func_computeSegMetrics3D(file['ES_myo'], file['ES_myo_pred'])

            dice_list.append(dice), jc_list.append(jc), hd_list.append(hd), asd_list.append(asd)
        except:
            continue

    return dice_list, jc_list, hd_list, asd_list





