import os 
import torch
import numpy as np
import cv2
import json


def calculate_mean_pixel_accuracy(true_labels, predicted_labels):
    assert true_labels.shape == predicted_labels.shape

    accuracy_class_1 = torch.sum((predicted_labels == 1) & (true_labels == 1)).float() / torch.sum(true_labels == 1).float()
    accuracy_class_0 = torch.sum((predicted_labels == 0) & (true_labels == 0)).float() / torch.sum(true_labels == 0).float()

    accuracy_class_1 = accuracy_class_1 if torch.sum(true_labels == 1) > 0 else torch.tensor(0.)
    accuracy_class_0 = accuracy_class_0 if torch.sum(true_labels == 0) > 0 else torch.tensor(0.)

    mPA = (accuracy_class_1 + accuracy_class_0) / 2
    return mPA


def calculate_mean_precision(true_labels, predicted_labels):
    assert true_labels.shape == predicted_labels.shape

    precision_class_1 = torch.sum((predicted_labels == 1) & (true_labels == 1)).float() / torch.sum(predicted_labels == 1).float()
    precision_class_0 = torch.sum((predicted_labels == 0) & (true_labels == 0)).float() / torch.sum(predicted_labels == 0).float()

    mP = (precision_class_1 + precision_class_0) / 2
    return mP


def m360(scene_name, eval_data_root, saving_root):
    """
        Evaluate the mask of m360, using annotations from the GOI paper.
    """
    gt_root = os.path.join(eval_data_root, scene_name)
    queries = os.listdir(gt_root)
    iou_list, mpa_list, mp_list = [], [], []

    for prompt in queries:
        gt_masks = os.listdir(os.path.join(gt_root, prompt, 'masks'))
        ious, mpas, mps = [], [], []
        for gt_mask in gt_masks:
            img_name = gt_mask.split('.')[0]
            gt_mask_p = os.path.join(gt_root, prompt, 'masks', gt_mask)
            rendered_mask_p = os.path.join(saving_root, scene_name, prompt, img_name + '.png')
            if not os.path.exists(rendered_mask_p):
                print(rendered_mask_p)
                continue
            gt_mask = cv2.imread(gt_mask_p, cv2.IMREAD_GRAYSCALE)
            rendered_mask = cv2.imread(rendered_mask_p, cv2.IMREAD_GRAYSCALE)
            rendered_mask = cv2.resize(rendered_mask, (gt_mask.shape[1], gt_mask.shape[0]))
            gt = gt_mask > 0
            pred = rendered_mask > 0
            iou = np.sum(gt & pred) / np.sum(gt | pred)
            ious.append(iou)
            mpas.append(calculate_mean_pixel_accuracy(torch.tensor(gt), torch.tensor(pred)).item())
            mps.append(calculate_mean_precision(torch.tensor(gt), torch.tensor(pred)).item())
        iou_list.append(np.mean(ious))
        mpa_list.append(np.mean(mpas))
        mp_list.append(np.mean(mps))
    print(f'{scene_name} metrics, (iou, mpa, mp): {np.mean(iou_list), np.mean(mpa_list), np.mean(mp_list)}')
    return np.mean(iou_list), np.mean(mpa_list), np.mean(mp_list)

def main_m360(scene_list, data_root, saving_root):
    ious, mpas, mps = [], [], []
    for scene in scene_list:
        iou, mpa, mp = m360(scene, data_root, saving_root)
        ious.append(iou)
        mpas.append(mpa)
        mps.append(mp)
    print(f'Overall metrics, (iou, mpa, mp): {np.mean(ious), np.mean(mpas), np.mean(mps)}')
        

def replica_top7(scene_name, data_root, saving_root):
    """
        Using the top 7 prompt list to evaluate the mask of replica.
    """
    gt_root = f'{data_root}/{scene_name}/test/sem'
    gt_masks = os.listdir(gt_root)

    with open(f"{data_root}/{scene_name}/test/top_list.json") as f:
        data = json.load(f)

    scene_iou, scene_mpa, scene_mp = [], [], []
    for gt in gt_masks:
        img_name = gt.split('.')[0]
        gt_mask_p = os.path.join(gt_root, gt)
        gt_all_mask = cv2.imread(gt_mask_p, cv2.IMREAD_GRAYSCALE)

        prompt_list = data[img_name + '.png']
        img_iou, img_mpa, img_mp = [], [], []
        for i in range(len(prompt_list)):
            prompt, id = prompt_list[i]['class_name'], prompt_list[i]['id']
            mask = gt_all_mask == id
            rendered_mask_p = os.path.join(saving_root, scene_name, prompt, 'rgb_' + img_name.split('_')[1] + '.png')
            if not os.path.exists(rendered_mask_p):
                print(rendered_mask_p)
                continue
            rendered_mask = cv2.imread(rendered_mask_p, cv2.IMREAD_GRAYSCALE)
            rendered_mask = cv2.resize(rendered_mask, (gt_all_mask.shape[1], gt_all_mask.shape[0]))
            rendered_mask = rendered_mask > 0
            iou = np.sum(mask & rendered_mask) / np.sum(mask | rendered_mask)
            mpa = calculate_mean_pixel_accuracy(torch.tensor(mask), torch.tensor(rendered_mask)).item()
            mp = calculate_mean_precision(torch.tensor(mask), torch.tensor(rendered_mask)).item()

            img_iou.append(iou)
            img_mpa.append(mpa)
            img_mp.append(mp)
        scene_iou.append(np.mean(img_iou))
        scene_mpa.append(np.mean(img_mpa))
        scene_mp.append(np.mean(img_mp))
    print(f'{scene_name} miou, mpa, mp: {np.mean(scene_iou), np.mean(scene_mpa), np.mean(scene_mp)}')
    return np.mean(scene_iou), np.mean(scene_mpa), np.mean(scene_mp)

def main_replica(scene_list, data_root, saving_root):
    iou, mpa, mp = [], [], []
    for scene in scene_list:
        _iou, _mpa, _mp = replica_top7(scene, data_root, saving_root)
        iou.append(_iou)
        mpa.append(_mpa)
        mp.append(_mp)
    print(f'Overall metrics, iou, mpa, mp: {np.mean(iou), np.mean(mpa), np.mean(mp)}')



            
if __name__ == "__main__":
    print('Please adjust the mask path based on where you save them. Change #L45 for m360 and #L95 for replica.\n')
    from argparse import ArgumentParser
    parser = ArgumentParser('Evaluate the mask of GOI')
    parser.add_argument('--eval_root', '-e', type=str)
    parser.add_argument('--saving_root', '-s', type=str)
    parser.add_argument('--scene_list', nargs='+', default=['room'])
    parser.add_argument('--dataset', '-d', type=str, default='m360')
    args = parser.parse_args()
    if args.dataset == 'm360':
        main_m360(args.scene_list, args.eval_root, args.saving_root)
    elif args.dataset == 'replica':
        main_replica(args.scene_list, args.eval_root, args.saving_root)
    else:
        raise ValueError('Unknown dataset')
