import os
import logging
import boto3
import io
import json


from mmdet.apis import init_detector, inference_detector

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, \
    get_single_tag_keys, DATA_UNDEFINED_NAME
from label_studio_tools.core.utils.io import get_data_dir
from botocore.exceptions import ClientError
from urllib.parse import urlparse


####################################################
# DETECTRON2 imports
####################################################
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

####################################################
# CUSTOM imports
####################################################
import cv2
import math
import numpy as np

def get_bbox8(cx, cy, w, h):
    """
    Get the 4x2 bounding box coordinates for a rotated rectangle
    wrt the center of the rectangle.

    """
    
    # get the 4x2 bounding box coordinates
    bbox8 = np.array([[-w / 2, -h / 2],
                      [w / 2, -h / 2],
                      [w / 2, h / 2],
                      [-w / 2, h / 2]])

    bbox8[:, 0] += cx
    bbox8[:, 1] += cy

    return bbox8

def rotate_bbox8(bbox8, center_coord, center_angle):
    """
    Rotate the 4x2 bounding box coordinates wrt the center coord.

    """
    bbox8 = bbox8.copy()

    # move the 4x2 bounding box coordinates to the origin
    bbox8[:, 0] -= center_coord[0]
    bbox8[:, 1] -= center_coord[1]


    # get the center angle in radians
    theta = math.radians(center_angle)


    # get the rotation matrix
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)]])

    # rotate the 4x2 bounding box coordinates
    bbox8 = np.matmul(bbox8, R)

    # move the 4x2 bounding box coordinates back to the center coord
    bbox8[:, 0] += center_coord[0]
    bbox8[:, 1] += center_coord[1]

    return bbox8

def draw_bbox8(name, bbox8):
    # get the top-left corner coordinates
    x1 = bbox8[0, 0]
    y1 = bbox8[0, 1]

    # get the top-right corner
    x2 = bbox8[1, 0]
    y2 = bbox8[1, 1]

    # get the bottom-right corner
    x3 = bbox8[2, 0]
    y3 = bbox8[2, 1]

    # get the bottom-left corner
    x4 = bbox8[3, 0]
    y4 = bbox8[3, 1]

    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.circle(img, (int(x1), int(y1)), 10, (0, 0, 255), -1)
    cv2.circle(img, (int(x2), int(y2)), 10, (0, 0, 255), -1)
    cv2.circle(img, (int(x3), int(y3)), 10, (0, 0, 255), -1)
    cv2.circle(img, (int(x4), int(y4)), 10, (0, 0, 255), -1)

    cv2.putText(img, "1", (int(x1 + 10), int(y1 + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(img, "2", (int(x2 + 10), int(y2 + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(img, "3", (int(x3 + 10), int(y3 + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(img, "4", (int(x4 + 10), int(y4 + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imwrite(name, img)


def get_rotated_bbox_coords(bbox8):
    """
    Get the top-left corner coordinates and rotation angle
    of a rotated rectangle from its 4x2 bounding box coordinates.

    """
    bbox8 = bbox8.copy()

    # get the top-left corner coordinates
    x1 = bbox8[0, 0]
    y1 = bbox8[0, 1]


    # get the top-right corner
    x2 = bbox8[1, 0]
    y2 = bbox8[1, 1]

    # get the bottom-right corner
    x3 = bbox8[2, 0]
    y3 = bbox8[2, 1]

    # get the bottom-left corner
    x4 = bbox8[3, 0]
    y4 = bbox8[3, 1]

    # get the width and height
    w = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    h = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)


    # after we move the coordinate system to the top-left
    # the top right corner will be tell us the angle
    cos_angle = (x2 - x1) / w
    sin_angle = (y2 - y1) / w


    angle1 = math.acos(cos_angle)
    angle2 = math.asin(sin_angle)

    # convert to degrees
    angle1 = math.degrees(angle1)
    angle2 = math.degrees(angle2)
    # get the rotation angle wrt the top-left corner

    theta = angle1 if angle2 > 0 else -angle1
    return x1, y1, w, h, theta
    

logger = logging.getLogger(__name__)


class Detectron2(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection"""

    def __init__(
        self,
        image_dir=None,
        config_file=None,
        checkpoint_file=None,
        labels_file=None, 
        score_threshold=0.01, 
        device='cuda:0', 
        **kwargs
    ):
        """
        Load MMDetection model from config and checkpoint into memory.
        (Check https://mmdetection.readthedocs.io/en/v1.2.0/GETTING_STARTED.html#high-level-apis-for-testing-images)

        Optionally set mappings from COCO classes to target labels
        :param config_file: Absolute path to MMDetection config file (e.g. /home/user/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x.py)
        :param checkpoint_file: Absolute path MMDetection checkpoint file (e.g. /home/user/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth)
        :param image_dir: Directory where images are stored (should be used only in case you use direct file upload into Label Studio instead of URLs)
        :param labels_file: file with mappings from COCO labels to custom labels {"airplane": "Boeing"}
        :param score_threshold: score threshold to wipe out noisy results
        :param device: device (cpu, cuda:0, cuda:1, ...)
        :param kwargs: can contain endpoint_url in case of non amazon s3
        """
        super(Detectron2, self).__init__(**kwargs)
        
        config_file = config_file or os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "workdir/rotated_bbox_config.yaml")

        checkpoint_file = checkpoint_file or os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "workdir/synced-rtx/training-output/debug0809/mapper_OldMapper/crop_images_False/num_paste_0/model_0038999.pth")

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.merge_from_file(config_file)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # Preiction confidence threshold,
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.05
        cfg.MODEL.WEIGHTS = checkpoint_file
        cfg.MODEL.DEVICE = device
        
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(f'{self.__class__.__name__} reads images from {self.image_dir}')
        self.label_map = open(labels_file).read().splitlines()
        self.label_map = {i: label for i, label in enumerate(self.label_map)}
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.label_map)
        

        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)

        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name

        print('Load new model from: ', config_file, checkpoint_file)
        self.predictor = DefaultPredictor(cfg)
        print(self.predictor.model)

    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3', endpoint_url=self.endpoint_url)
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )
            except ClientError as exc:
                logger.warning(f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
        return image_url

    def predict(self, tasks, **kwargs):
        assert len(tasks) == 1
        task = tasks[0]
        
        image_url = self._get_image_url(task)
        # image_path = self.get_local_path(image_url)
        image_path = image_url.replace("/data/local-files/?d=", "./datadir/")

        assert os.path.exists(image_path), f'Image not found: {image_path}'
        # image = read_image(image_path, format="BGR")
        image = cv2.imread(image_path)

        model_results = self.predictor(image)["instances"]
        
        results = []
        all_scores = []
        img_width, img_height = get_image_size(image_path)
        # model_results is a dict contiaining detectron2.structures.instances.Instances:
        #   - pred_boxes: Boxes object storing N instances of predicted object bounding boxes
        #   - scores: Tensor of N confidence scores for the predicted object class
        #   - pred_classes: Tensor of N labels (in integer format) for predicted object classes
        #   - pred_masks: Tensor of N predicted masks for each instance
        #   - pred_keypoints: Tensor of N predicted keypoints for each instance
        #   - pred_keypoint_heatmaps: Tensor of N predicted keypoint heatmaps for each instance

        # we convert the results into Label Studio format
        for instance_idx in range(len(model_results)):
            # check format of bbox
            instance = model_results[instance_idx]
            cx, cy, w, h, center_angle = instance.pred_boxes.tensor.tolist()[0]

            # get the coordinates of the rotated bbox
            bbox8 = get_bbox8(cx, cy, w, h)
            
            rot_bbox8 = rotate_bbox8(bbox8, (cx, cy), center_angle)
            
            xmin, ymin, w, h, top_left_angle = get_rotated_bbox_coords(rot_bbox8)

            # get the confidence score
            score = instance.scores.tolist()[0]
            # get the label
            label = self.label_map[instance.pred_classes.tolist()[0]]
            all_scores.append(score)
            # convert to Label Studio format
            results.append({
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'rectanglelabels',
                'value': {
                    'rectanglelabels': [label],
                    'x': xmin / img_width * 100,
                    'y': ymin / img_height * 100,
                    'width': w / img_width * 100,
                    'height': h / img_height * 100,
                    'rotation': top_left_angle,
                },
                "score": score,
            })
            # results.append({
            #     'from_name': self.from_name,
            #     'to_name': self.to_name,
            #     'type': 'rectanglelabels',
            #     'value': {
            #         'rectanglelabels': [label],
            #         'x': xmin / img_width * 100,
            #         'y': ymin / img_height * 100,
            #         'width': (xmax - xmin) / img_width * 100,
            #         'height': (ymax - ymin) / img_height * 100,
            #         'rotation': -ang,
            #     },
            #     "score": score,
            # })
        
        # sort results by score
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        avg_score = sum(all_scores) / len(all_scores)
        return [{
            'result': results,
            'score': avg_score,
        }]
        

        # for bboxes, label in zip(model_results, self.model.CLASSES):
        #     output_label = self.label_map.get(label, label)

        #     if output_label not in self.labels_in_config:
        #         print(output_label + ' label not found in project config.')
        #         continue
        #     for bbox in bboxes:
        #         bbox = list(bbox)
        #         if not bbox:
        #             continue
        #         score = float(bbox[-1])
        #         if score < self.score_thresh:
        #             continue
        #         x, y, xmax, ymax = bbox[:4]
        #         results.append({
        #             'from_name': self.from_name,
        #             'to_name': self.to_name,
        #             'type': 'rectanglelabels',
        #             'value': {
        #                 'rectanglelabels': [output_label],
        #                 'x': x / img_width * 100,
        #                 'y': y / img_height * 100,
        #                 'width': (xmax - x) / img_width * 100,
        #                 'height': (ymax - y) / img_height * 100
        #             },
        #             'score': score
        #         })
        #         all_scores.append(score)
        # avg_score = sum(all_scores) / max(len(all_scores), 1)
        # return [{
        #     'result': results,
        #     'score': avg_score
        # }]


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data
