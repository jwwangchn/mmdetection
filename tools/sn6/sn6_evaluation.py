import os
import cv2
import numpy as np
import mmcv
import wwtool
import pandas as pd
import argparse
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
import solaris as sol


def mask2segm(bbox_result, segm_result, heigh=900, width=900, score_thr=0.05):
    """convert mask to binary mask (semantic segmentation)
    
    Arguments:
        bbox_result {list} --  results of object detection
        segm_result {list} -- results of instance segmentation
        heigh {int} -- height of image
        width {int} -- width of image
    """
    binary_mask = wwtool.generate_image(heigh, width, 0)
    # bboxes = np.vstack(bbox_result)
    # segms = mmcv.concat_list(segm_result)
    inds = np.where(bbox_result[:, -1] > score_thr)[0]
    for i in inds:
        mask = maskUtils.decode(segm_result[i]).astype(np.bool)
        binary_mask[mask] = 255
    
    # wwtool.show_image(binary_mask)
    
    return binary_mask
    

def evaluation(truthcsv, predcsv):
    """
    Compares infered test data vector labels to ground truth.
    """
    minevalsize = 80

    evaluator = sol.eval.base.Evaluator(truthcsv)
    evaluator.load_proposal(predcsv, proposalCSV=True, conf_field_list=[])
    report = evaluator.eval_iou_spacenet_csv(miniou=0.5, min_area=minevalsize)

    tp = 0
    fp = 0
    fn = 0
    for entry in report:
        tp += entry['TruePos']
        fp += entry['FalsePos']
        fn += entry['FalseNeg']
    f1score = (2*tp) / (2*tp + fp + fn)
    print('Vector F1: {}'.format(f1score))

def parse_args():
    parser = argparse.ArgumentParser(description='SN6 Evaluation')
    parser.add_argument('--dataset', default='sn6', help='dataset name')
    parser.add_argument('--dataset_version', default='v1', help='dataset name')
    parser.add_argument('--config_version', help='version of experiments (DATASET_V#NUM)')
    parser.add_argument('--imageset', default='test', help='imageset of evaluation')
    parser.add_argument('--image_source', default='SAR-Intensity', help='PS-RGB or SAR-Intensity')
    parser.add_argument('--evaluation', default=True, help='evaluation flag')
    parser.add_argument('--evaluation_only', action='store_true', help='evaluation only flag')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # config and result files
    config_file = './configs/{}/{}.py'.format(args.dataset, args.config_version)
    pkl_file = 'results/{}/{}/coco_results.pkl'.format(args.dataset, args.config_version)
    annopath = '/data/sn6/{}/coco/annotations/sn6_{}_{}_{}.json'.format(args.dataset_version, args.imageset, args.dataset_version, args.image_source)

    # csv file
    predcsv_file = './results/{}/{}/SN6_Test_Public_AOI_11_Rotterdam_Buildings.csv'.format(args.dataset, args.config_version)
    truthcsv_file = './data/sn6/v1/{}/labels/SN6_Train_AOI_11_Rotterdam_Buildings.csv'.format(args.imageset)

    # load COCO
    coco=COCO(annopath)

    catIds = coco.getCatIds(catNms=[''])
    imgIds = coco.getImgIds(catIds=catIds)

    prog_bar = mmcv.ProgressBar(len(imgIds))

    bboxes, segms = wwtool.load_coco_pkl_results(pkl_file)

    if not args.evaluation_only:
        firstfile = True
        for idx, imgId in enumerate(imgIds):
            img = coco.loadImgs(imgIds[idx])[0]
            img_name = img['file_name']

            bbox_result, segm_result = bboxes[idx], segms[idx]
            binary_mask = mask2segm(bbox_result, segm_result)

            vectordata = sol.vector.mask.mask_to_poly_geojson(
                binary_mask,
                output_type='csv',
                min_area=0,
                bg_threshold=128,
                do_transform=False,
                simplify=True
            )

            #Add to the cumulative inference CSV file
            tilename = '_'.join(os.path.splitext(img_name)[0].split('_')[-4:])
            csvaddition = pd.DataFrame({'ImageId': tilename,
                                        'BuildingId': 0,
                                        'PolygonWKT_Pix': vectordata['geometry'],
                                        'Confidence': 1
            })
            csvaddition['BuildingId'] = range(len(csvaddition))
            if firstfile:
                proposalcsv = csvaddition
                firstfile = False
            else:
                proposalcsv = proposalcsv.append(csvaddition)

            prog_bar.update()
        
        # save csv file
        proposalcsv.to_csv(predcsv_file, index=False)
    
    # evaluation
    if args.evaluation:
        evaluation(truthcsv_file, predcsv_file)