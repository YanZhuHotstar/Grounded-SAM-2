import numpy as np
import os
import argparse
from tqdm import tqdm

from metrics_utils import evaluate_scenes


def evaluate(result_dir, gt_dir, n_frames_miss_tolerance, verbose=False):
    ### F1 score
    p, r, f1, n, tp, fp, fn = 0, 0, 0, 0, 0, 0, 0
    # skip per-frame prediction files
    pred_files = [x for x in os.listdir(result_dir) if (not x.endswith('frame.txt')) and (x.endswith('.txt'))]
    # pred file could contain suffix start with underscore, remove suffix to obtain ground truth file name
    gt_files = ['_'.join(x.split('_')[:-1]) + '.txt' for x in pred_files]
    # filter out files without ground truth labels
    good_files = [(_pred, _gt) for _pred, _gt in zip(pred_files, gt_files)
                  if os.path.exists(os.path.join(gt_dir, _gt))]
    if len(good_files) != len(pred_files):
        print(
            f'WARNING: {len(pred_files) - len(good_files)}/{len(pred_files)} files do not have ground truth labels, skipping them in evaluation. ')
    for _pred, _gt in tqdm(good_files):
        pred_scenes = np.loadtxt(os.path.join(result_dir, _pred), dtype=np.int32, ndmin=2)
        gt_scenes = np.loadtxt(os.path.join(gt_dir, _gt), dtype=np.int32, ndmin=2)
        _p, _r, _f1, (_tp, _fp, _fn) = evaluate_scenes(gt_scenes, pred_scenes,
                                                       n_frames_miss_tolerance=n_frames_miss_tolerance)
        p += _p
        r += _r
        f1 += _f1
        n += 1
        tp += _tp
        fp += _fp
        fn += _fn
        if verbose:
            print(_pred,
                  ' - p={:.4f}, r={:.4f}, f1={:.4f}, tp={:d}, fp={:d}, fn={:d}'.format(_p, _r, _f1, _tp, _fp, _fn))
    overall_p = tp / (tp + fp)
    overall_r = tp / (tp + fn)
    overall_f1 = 2 * overall_r * overall_p / (overall_r + overall_p)
    print('Overall - p={:.4f}, r={:.4f}, f1={:.4f}, tp={:d}, fp={:d}, fn={:d}'.format(overall_p, overall_r, overall_f1,
                                                                                      tp, fp, fn))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate f1 score, assume ground truth file has the same name as the predition file.")
    parser.add_argument("--gt_dir", required=True, help="path to ground truth")
    parser.add_argument("--result_dir", required=True, help="path to predicted results")
    parser.add_argument("--n_frames_miss_tolerance", type=int, default=2, help="max number of frame shifting tolerance to be counted as true positive")
    parser.add_argument("--verbose", action='store_true', help="print result of every video")
    args = parser.parse_args()

    evaluate(**vars(args))
