import os
import cv2
import numpy as np
import mmcv
import wwtool
import pandas as pd
import argparse
import pycocotools.mask as maskUtils
import solaris as sol

from mmdet.apis import init_detector, inference_detector, show_result


def mask2segm(bbox_result, segm_result, heigh=900, width=900, score_thr=0.05, show_flag=False):
    """convert mask to binary mask (semantic segmentation)
    
    Arguments:
        bbox_result {list} --  results of object detection
        segm_result {list} -- results of instance segmentation
        heigh {int} -- height of image
        width {int} -- width of image
    """
    binary_mask = wwtool.generate_image(heigh, width, 0)
    bboxes = np.vstack(bbox_result)
    segms = mmcv.concat_list(segm_result)
    inds = np.where(bboxes[:, -1] > score_thr)[0]
    for i in inds:
        mask = maskUtils.decode(segms[i]).astype(np.bool)
        binary_mask[mask] = 255

    if show_flag:
        wwtool.show_image(binary_mask)
    
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
    parser = parser = argparse.ArgumentParser(description='SN6 Testing')
    parser.add_argument('--dataset', default='sn6', help='dataset name')
    parser.add_argument('--dataset_version', default='v1', help='dataset name')
    parser.add_argument('--config_version', default='sn6_v102', help='version of experiments (DATASET_V#NUM)')
    parser.add_argument('--imageset', default='test', help='imageset of evaluation')
    parser.add_argument('--image_source', default='SAR-Intensity', help='PS-RGB or SAR-Intensity')
    parser.add_argument('--epoch', default=12, help='epoch')
    parser.add_argument('--evaluation', default=False, help='evaluation flag')
    parser.add_argument('--show', action='store_true', help='show flag')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # config and result files
    config_file = './configs/{}/{}.py'.format(args.dataset, args.config_version)
    checkpoint_file = './work_dirs/{}/epoch_{}.pth'.format(args.config_version, args.epoch)
    img_dir = './data/{}/v1/{}/{}'.format(args.dataset, args.imageset, args.image_source)

    # csv file
    predcsv_file = './results/{}/{}/SN6_Test_Public_AOI_11_Rotterdam_Buildings.csv'.format(args.dataset, args.config_version)
    truthcsv_file = './data/sn6/v1/{}/labels/SN6_Train_AOI_11_Rotterdam_Buildings.csv'.format(args.imageset)

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img_list = os.listdir(img_dir)
    prog_bar = mmcv.ProgressBar(len(img_list))

    firstfile = True
    for img_name in img_list:
        img_file = os.path.join(img_dir, img_name)
        img = cv2.imread(img_file)
        if args.show:
            wwtool.show_image(img, win_name='original')
        bbox_result, segm_result = inference_detector(model, img_file)
        # show_result(img, (bbox_result, segm_result), model.CLASSES, score_thr=0.5)
        # wwtool.imshow_bboxes(img, bbox_result[0][:, 0:-1], show=True)
        binary_mask = mask2segm(bbox_result, segm_result, show_flag=args.show)

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
