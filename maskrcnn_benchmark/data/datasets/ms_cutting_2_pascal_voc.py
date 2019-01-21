import torch
import torchvision

import json
from PIL import Image
import os
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class MSCuttingDatasetVOC(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "tool",
        "cucumber",
        "cucumber_slice",
    )

    def __init__(self, root, remove_images_without_annotations, transforms=None):
        super(MSCuttingDataset, self).__init__()

        self.data_root = root

        self.img_with_anno_list = self.get_all_images_with_annotations(root)
	self.num_images = len(self.img_with_anno_list)

        self.img_id_to_path, self.img_id_to_anno = {}, {}
        for i in range(self.num_images):
            self.img_id_to_path[i] = self.img_with_anno_list[i][0]
            self.img_id_to_anno[i] = self.img_with_anno_list[i][1]

 	cls = MSCuttingDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def is_valid_image_file(filepath):
        return filepath.endswith('.jpg') or filepath.endswith('.jpeg') or \
            filepath.endswith('.png') or filepath.endswith('.bmp')

    def is_valid_annotation_file(filepath):
        return filepath.endswith('.xml')

    def parse_voc_xml(self, node):
	    voc_dict = {}
	    children = list(node)
	    if children:
		def_dic = collections.defaultdict(list)
		for dc in map(self.parse_voc_xml, children):
		    for ind, v in dc.items():
			def_dic[ind].append(v)
		voc_dict = {
		    node.tag:
		    {ind: v[0] if len(v) == 1 else v
		     for ind, v in def_dic.items()}
		}
	    if node.text:
		text = node.text.strip()
		if not children:
		    voc_dict[node.tag] = text
	    return voc_dict

    def get_all_images_with_annotations(self, data_root):
        '''Get all images within every folder of data_root which has an
           annotation available.
        '''
        img_path_to_anno_dict = {}
        for d in os.listdir(data_root):
            d_path = os.path.join(data_root, d)
            if os.path.isfile(d) and self.is_valid_annotation_file(d_path):
                img_path_no_suffix = d_path[:-4]
                if os.path.exists(img_path_no_suffix + "jpg"):
                    img_path_to_anno_dict[img_path_no_suffix + "jpg"] = \
                            d_path
                elif os.path.exists(img_path_no_suffix + "jpeg"):
                    img_path_to_anno_dict[img_path_no_suffix + "jpeg"] = \
                            d_path
                elif os.path.exists(img_path_no_suffix + "png"):
                    img_path_to_anno_dict[img_path_no_suffix + "png"] = \
                            d_path
                elif os.path.exists(img_path_no_suffix + "bmp"):
                    img_path_to_anno_dict[img_path_no_suffix + "bmp"] = \
                            d_path
                else:
                    print("Cannot find image for annotation: {}".format(d_path))
        
        img_with_anno_list = []
        for f in sorted(img_path_to_anno_dict.keys())
            img_with_anno_list.append((f, img_path_to_anno_dict[f])) 

        return img_with_anno_list

    def __getitem__(self, idx):
        img, anno = super
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        img_path = self.img_id_to_path[index]
        img = Image.open(img_path).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index


    def get_groundtruth(self, index):
        annopath = self.img_id_to_anno[index]
        anno = ET.parse(annopath).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1
        
        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text, 
                bb.find("ymin").text, 
                bb.find("xmax").text, 
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        annopath = self.img_id_to_anno[index]
        anno = ET.parse(annopath).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}
