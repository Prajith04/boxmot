# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path

import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model, is_yolox_model)

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.
    """
    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)
    predictor.trackers = trackers

@torch.no_grad()
def run(args):
    if args.imgsz is None:
        args.imgsz = default_imgsz(args.yolo_model)

    # Initialize YOLO with the specified model (should be a pose model)
    yolo = YOLO(
        args.yolo_model if is_ultralytics_model(args.yolo_model)
        else 'yolov8n.pt',
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if not is_ultralytics_model(args.yolo_model):
        m = get_yolo_inferer(args.yolo_model)
        yolo_model = m(model=args.yolo_model, device=yolo.predictor.device,
                       args=yolo.predictor.args)
        yolo.predictor.model = yolo_model
        if is_yolox_model(args.yolo_model):
            yolo.add_callback("on_predict_batch_start",
                              lambda p: yolo_model.update_im_paths(p))
            yolo.predictor.preprocess = (
                lambda imgs: yolo_model.preprocess(im=imgs))
            yolo.predictor.postprocess = (
                lambda preds, im, im0s:
                yolo_model.postprocess(preds=preds, im=im, im0s=im0s))

    yolo.predictor.custom_args = args
    
    for r in results:
        # Plot the detections and keypoints (if pose model is used)
        img = r.plot()

        # Process keypoints for posture detection
        if r.keypoints is not None:  # Check if keypoints are available (pose model required)
            for i in range(len(r.boxes)):
                box = r.boxes[i].xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                kpts = r.keypoints.xy[i].cpu().numpy()  # (num_keypoints, 2) - x, y coordinates
                confs = r.keypoints.conf[i].cpu().numpy()  # (num_keypoints,) - confidence scores

                # COCO keypoint indices: 5 (left shoulder), 6 (right shoulder), 11 (left hip), 12 (right hip)
                ls_idx, rs_idx, lh_idx, rh_idx = 5, 6, 11, 12
                
                # Ensure all required keypoints are detected with sufficient confidence
                if all(confs[idx] > 0.5 for idx in [ls_idx, rs_idx, lh_idx, rh_idx]):
                    ls = kpts[ls_idx]  # Left shoulder
                    rs = kpts[rs_idx]  # Right shoulder
                    lh = kpts[lh_idx]  # Left hip
                    rh = kpts[rh_idx]  # Right hip

                    # Calculate neck (midpoint of shoulders) and hip (midpoint of hips)
                    neck = (ls + rs) / 2
                    hip = (lh + rh) / 2

                    # Compute horizontal and vertical distances
                    dx = abs(neck[0] - hip[0])
                    dy = hip[1] - neck[1]  # y increases downwards in image coordinates

                    # Assess posture
                    if dy > 0:  # Ensure hip is below neck (upright person)
                        ratio = dx / dy
                        if ratio < 0.2:  # Threshold for vertical alignment
                            posture = "Good Posture"
                            color = (0, 255, 0)  # Green
                        else:
                            posture = "Bad Posture"
                            color = (0, 0, 255)  # Red
                    else:
                        posture = "Unknown"
                        color = (255, 255, 0)  # Yellow
                else:
                    posture = "Unknown"
                    color = (255, 255, 0)  # Yellow (insufficient keypoint confidence)

                # Add posture text above the bounding box
                text_position = (int(box[0]), int(box[1] +10))
                cv2.putText(img, posture, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the annotated frame
        if args.show is True:
            cv2.imshow('BoxMOT', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path (use yolov8n-pose.pt for posture detection)')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=None,
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0 for persons')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='show confidences')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    run(opt)