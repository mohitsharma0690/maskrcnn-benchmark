from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import cv2
import pdb

# config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
config_file = "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"	

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
# coco_demo Uses image in BGR format.
image = cv2.imread('/home/mohit/Datasets/cutting/cucumber_1.jpeg')
pdb.set_trace()
predictions = coco_demo.compute_prediction(image)
top_predictions = coco_demo.select_top_predictions(predictions)
print(predictions)
visual_predictions = coco_demo.run_on_opencv_image(image)
