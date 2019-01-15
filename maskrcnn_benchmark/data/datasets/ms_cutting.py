import torch
import torchvision

import json
from PIL import Image
import os

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

class MSCuttingDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, root, remove_images_without_annotations=True,
                 transforms=None):
        super(MSCuttingDataset, self).__init__()

        self.json_root = ann_file
        self.data_root = root

        json_data = json.load(open(self.json_root, 'r'))
        num_images = len(json_data)
        img_id, category_id = 0, 0
        self.img_id_to_path, self.img_id_to_anno= {}, {}
        self.json_category_id_to_contiguous_id = {}
        self.continuous_category_id_to_json_id = {}
        for i, annot_data in enumerate(json_data):
            if annot_data['class'] == 'image':
                self.img_id_to_path[img_id] = annot_data['filename'] 
                self.img_id_to_anno[img_id] = annot_data
                img_id = img_id + 1
                
                for anno in annot_data['annotations']:
                    if self.json_category_id_to_contiguous_id.get(
                            anno['class']) is None:
                        self.json_category_id_to_contiguous_id[anno["class"]] =\
                                category_id
                        self.continuous_category_id_to_json_id[category_id] =\
                                anno['class'] 
                        category_id = category_id + 1

        
        self.num_images = img_id
        print("Total images loaded: {}, total classes: {}".format(
            self.num_images, len(self.continuous_category_id_to_json_id)))

        self.transforms = transforms

    def get_bounding_box_for_scrot_annotation_dict(self, anno_dict, mode="xywh"):
        if mode == "xywh":
            return (anno_dict['x'], anno_dict['y'], anno_dict['width'],
                    anno_dict['height'])
        elif mode == "xyxy":
            return (anno_dict['x'], anno_dict['y'],
                    anno_dict['x'] + anno_dict['width'],
                    anno_dict['y'] + anno_dict['height'])
        else:
            raise ValueError("Incorrect mode {}".format(mode))

    def __getitem__(self, idx):
        img_path, anno = self.img_id_to_path[idx], self.img_id_to_anno[idx]

        img = Image.open(os.path.join(self.data_root, img_path)).convert('RGB')
        '''
        if self.transforms is not None:
            img = self.transforms(img)
        '''

        boxes = [self.get_bounding_box_for_scrot_annotation_dict(obj) 
                    for obj in anno['annotations']]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["class"] for obj in anno['annotations']]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        # We don't have any segmentation masks

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def __len__(self):
        return self.num_images

    def get_img_info(self, index):
        img_path = self.img_id_to_path[index]
        img = Image.open(os.path.join(self.data_root, img_path)).convert('RGB')
        img_info = {"height": img.height, "width": img.width}
        return img_info
