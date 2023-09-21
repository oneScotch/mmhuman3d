import os
import os.path as osp
import shutil
from argparse import ArgumentParser

import mmcv
from mmcv.runner.checkpoint import _load_checkpoint, load_state_dict

import numpy as np
import torch
from collections import OrderedDict

from mmhuman3d.apis import (
    feature_extract,
    inference_image_based_model,
    init_model,
)

from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.utils.demo_utils import (
    prepare_frames,
    process_mmdet_results,
)
from mmhuman3d.utils.ffmpeg_utils import array_to_images
from mmhuman3d.utils.transforms import rotmat_to_aa

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False



def get_detection_result(args, frames_iter, mesh_model, extractor):
    person_det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    frame_id_list = []
    result_list = []
    for i, frame in enumerate(mmcv.track_iter_progress(frames_iter)):
        mmdet_results = inference_detector(person_det_model, frame)
        # keep the person class bounding boxes.
        results = process_mmdet_results(
            mmdet_results, cat_id=args.det_cat_id, bbox_thr=args.bbox_thr)
        # extract features from the input video or image sequences
        if mesh_model.cfg.model.type == 'VideoBodyModelEstimator' \
                and extractor is not None:
            results = feature_extract(
                extractor, frame, results, args.bbox_thr, format='xyxy')
        # drop the frame with no detected results
        if results == []:
            continue
        # vis bboxes
        if args.draw_bbox:
            bboxes = [res['bbox'] for res in results]
            bboxes = np.vstack(bboxes)
            mmcv.imshow_bboxes(
                frame, bboxes, top_k=-1, thickness=2, show=False)
        frame_id_list.append(i)
        result_list.append(results)

    return frame_id_list, result_list


def single_person_with_mmdet(args, frames_iter):
    """Estimate smplx parameters with mmdetection
    Args:
        args (object):  object of argparse.Namespace.
        frames_iter (np.ndarray,): prepared frames

    """
    mesh_model, extractor = init_model(
        args.mesh_reg_config,
        args.mesh_reg_checkpoint,
        device=args.device.lower())
    smplx_results = dict(
        global_orient=[],
        body_pose=[],
        betas=[],
        left_hand_pose=[],
        right_hand_pose=[],
        jaw_pose=[],
        expression=[])
    pred_cams, bboxes_xyxy = [], []

    frame_id_list, result_list = get_detection_result(args, frames_iter,
                                                      mesh_model, extractor)

    frame_num = len(frame_id_list)

    smplerx_config = mmcv.Config.fromfile(args.mesh_reg_config)
    print("here")
    smplerx_config.model['device'] = args.device
    model, _ = init_model(
        smplerx_config, device=args.device.lower())
    ckpt = torch.load(args.mesh_reg_checkpoint)
    new_state_dict = OrderedDict()
    for k, v in ckpt['network'].items():
        if 'module' not in k:
            k = 'module.' + k
        k = k.replace('module.backbone', 'module.encoder').replace('body_rotation_net', 'body_regressor').replace(
            'hand_rotation_net', 'hand_regressor')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)

    targets = {}
    meta_info = {}
    for i, result in enumerate(mmcv.track_iter_progress(result_list)):
        # mesh recovery
        print(result)
        with torch.no_grad():
            out = model.forward_test(result, targets, meta_info)
        mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]


    #     ## save single person param
    #     smplx_pred = {}
    #     smplx_pred['global_orient'] = out['smplx_root_pose'].reshape(-1,3).cpu().numpy()
    #     smplx_pred['body_pose'] = out['smplx_body_pose'].reshape(-1,3).cpu().numpy()
    #     smplx_pred['left_hand_pose'] = out['smplx_lhand_pose'].reshape(-1,3).cpu().numpy()
    #     smplx_pred['right_hand_pose'] = out['smplx_rhand_pose'].reshape(-1,3).cpu().numpy()
    #     smplx_pred['jaw_pose'] = out['smplx_jaw_pose'].reshape(-1,3).cpu().numpy()
    #     smplx_pred['leye_pose'] = np.zeros((1, 3))
    #     smplx_pred['reye_pose'] = np.zeros((1, 3))
    #     smplx_pred['betas'] = out['smplx_shape'].reshape(-1,10).cpu().numpy()
    #     smplx_pred['expression'] = out['smplx_expr'].reshape(-1,10).cpu().numpy()
    #     smplx_pred['transl'] =  out['cam_trans'].reshape(-1,3).cpu().numpy()

    #         if args.output is not None:
    #     os.makedirs(args.output, exist_ok=True)
    #     human_data = HumanData()
    #     smplx = {}
    #     smplx['fullpose'] = fullpose
    #     smplx['betas'] = smplx_results['betas']
    #     human_data['smplx'] = smplx
    #     human_data['pred_cams'] = pred_cams
    #     human_data.dump(osp.join(args.output, 'inference_result.npz'))

    #     save_path_smplx = os.path.join(args.output_folder, 'smplx')
    #     os.makedirs(save_path_smplx, exist_ok= True)

    #     npz_path = os.path.join(save_path_smplx, f'{frame:05}_{bbox_id}.npz')
    #     np.savez(npz_path, **smplx_pred)

    #     ## render single person mesh
    #     focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
    #     princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
    #     vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt},
    #                             mesh_as_vertices=args.show_verts)
    #     if args.show_bbox:
    #         vis_img = cv2.rectangle(vis_img, start_point, end_point, (255, 0, 0), 2)

    #     ## save single person meta
    #     meta = {'focal': focal,
    #             'princpt': princpt,
    #             'bbox': bbox.tolist(),
    #             'bbox_mmdet': mmdet_box_xywh.tolist(),
    #             'bbox_id': bbox_id,
    #             'img_path': img_path}
    #     json_object = json.dumps(meta, indent=4)

    #     save_path_meta = os.path.join(args.output_folder, 'meta')
    #     os.makedirs(save_path_meta, exist_ok= True)
    #     with open(os.path.join(save_path_meta, f'{frame:05}_{bbox_id}.json'), "w") as outfile:
    #         outfile.write(json_object)
    #     frame_name = img_path.split('/')[-1]
    #     save_path_img = os.path.join(args.output, 'img')
    #     os.makedirs(save_path_img, exist_ok= True)
    #     cv2.imwrite(os.path.join(save_path_img, f'{frame_name}'), vis_img[:, :, ::-1])


    # del model
    # del extractor
    # torch.cuda.empty_cache()


def main(args):

    # prepare input
    frames_iter = prepare_frames(args.input_path)

    single_person_with_mmdet(args, frames_iter)



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '--mesh_reg_config',
        type=str,
        default='configs/expose/expose.py',
        help='Config file for mesh regression')
    parser.add_argument(
        '--mesh_reg_checkpoint',
        type=str,
        default='data/pretrained_models/expose-d9d5dbf7_20220708.pth',
        help='Checkpoint file for mesh regression')
    parser.add_argument(
        '--single_person_demo',
        action='store_true',
        help='Single person demo with MMDetection')
    parser.add_argument(
        '--det_config',
        default='demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
        help='Config file for detection')
    parser.add_argument(
        '--det_checkpoint',
        default='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
        'faster_rcnn_r50_fpn_1x_coco/'
        'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        help='Checkpoint file for detection')
    parser.add_argument(
        '--tracking_config',
        default='demo/mmtracking_cfg/'
        'deepsort_faster-rcnn_fpn_4e_mot17-private-half.py',
        help='Config file for tracking')
    parser.add_argument(
        '--det_cat_id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')

    parser.add_argument(
        '--body_model_dir',
        type=str,
        default='data/body_models/',
        help='Body models file path')
    parser.add_argument(
        '--input_path',
        type=str,
        default='demo/resources/single_person_demo.mp4',
        help='Input path')
    parser.add_argument(
        '--output',
        type=str,
        default='demo_result',
        help='directory to save output result file')
    parser.add_argument(
        '--show_path',
        type=str,
        default='demo_result',
        help='directory to save rendered images or video')
    parser.add_argument(
        '--render_choice',
        type=str,
        default='hq',
        help='Render choice parameters')
    parser.add_argument(
        '--bbox_thr',
        type=float,
        default=0.99,
        help='Bounding box score threshold')
    parser.add_argument(
        '--draw_bbox',
        action='store_true',
        help='Draw a bbox for each detected instance')
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()

    if args.single_person_demo:
        assert has_mmdet, 'Please install mmdet to run the demo.'
        assert args.det_config is not None
        assert args.det_checkpoint is not None
    main(args)
