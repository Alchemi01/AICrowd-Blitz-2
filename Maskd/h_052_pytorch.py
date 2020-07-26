# -*- coding:utf-8 -*-
import cv2
import time

import argparse
import numpy as np
from PIL import Image
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.pytorch_loader import load_pytorch_model, pytorch_inference

model = load_pytorch_model('models/model360.pth');
# anchor configuration
#feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}


def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)

    image_transposed = image_exp.transpose((0, 3, 1, 2))

    y_bboxes_output, y_cls_output = pytorch_inference(model, image_transposed)
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]

        # margin for thumb img
        margin_time = 0.2
        t_x1, t_y1, t_x2, t_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        t_w = t_x2 - t_x1
        t_h = t_y2 - t_y1

        t_margin_x = margin_time * t_w / 2
        t_margin_y = margin_time * t_h / 2

        t_x1 -= t_margin_x
        t_y1 -= t_margin_y
        t_x2 += t_margin_x
        t_y2 += t_margin_y

        # x_dis = int(np.maximum(x - margin_x, 0))
        # y_dis = int(np.maximum(y - margin_y, 0))
        # x_dis2 = int(np.minimum(x2 + margin_x, img_size[1]))
        # y_dis2 = int(np.minimum(y2 + margin_y, img_size[0]))

        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(t_x1 * width))
        ymin = max(0, int(t_y1 * height))
        xmax = min(int(t_x2 * width), width)
        ymax = min(int(t_y2 * height), height)

        if draw_result:
            if class_id == 0:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        output_info.append([xmin, ymin, xmax, ymax, conf, class_id])

    if show_result:
        Image.fromarray(image).show()
    return output_info



import cv2
import sys
import numpy as np
import datetime
import os
import glob
import time
import json

from tqdm import tqdm


FOLDER_PATH = './AICrowd/maskd/dataset'
TEST_FOLDER = os.path.join(FOLDER_PATH, 'test_images')

with open(FOLDER_PATH + '/test.json') as json_file:
    test_data = json.load(json_file)
test_images_map_id = {}
for x in test_data["images"]:
    test_images_map_id[x["file_name"]] = x["id"]

def post_process(filename, dets, img, is_show_box=False):
    global test_images_map_id
    this_res = []
    #print('dets: ', dets)
    for i in range(len(dets)):
        #print('score', dets[i][4])
        face = dets[i]
        box = face[0:4]
        detect_score = float(face[4])
        #print(i,box,mask_score)

        # margin = 20
        # x1 = max(int(box[0])-margin, 0)
        # y1 = max(int(box[1])-margin, 0)
        # x2 = min(int(box[2])+margin, width)
        # y2 = min(int(box[3])+margin, height)

        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3]        )
        w = x2 - x1
        h = y2 - y1

        # min_size = 10
        # if w <= min_size or h <= min_size:
        #     continue

        # box_img = img[y1:y2, x1:x2]
        class_id = face[5]
        is_mask = 1 if class_id == 0 else 2

        if is_show_box:
            # draw bbox
            color = (0,0,255) if is_mask == 1 else (0,255,0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # if ql_label == 'bad' and ql_scores[0] > 0.95:
        #     # skip this bad detect
        #     continue

        # is_mask = 1 if ql_label == 'mask' else 2          # 1: mask, 2: no-mask

        _result = {}
        # filename = os.path.basename(filepath)
        _result["image_id"] = test_images_map_id[filename]
        # _result["bbox"] = [int(box[0]), int(box[1]), int(box[2])-int(box[0]), int(box[3])-int(box[1])]
        _result["bbox"] = [x1, y1, w, h]
        _result["score"] = detect_score
        _result["category_id"] = is_mask
        this_res.append(_result)
    return this_res


list_result = []
files = os.listdir(TEST_FOLDER)
for filename in tqdm(files, total=len(files)):
    filepath = os.path.join(TEST_FOLDER, filename)
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = inference(img,
              conf_thresh=0.1,
              iou_thresh=0.4,
              target_shape=(360, 360),
              draw_result=False,
              show_result=False)

    is_show_box = False
    res = post_process(filename, dets, img, is_show_box=is_show_box)
    list_result.extend(res)

    if is_show_box:
        # show image
        cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # cv2.imshow('box_img', box_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        #break
        #continue
    #break

ts = int(time.time())
fp = open('submission_' + str(ts) + '.json', "w")

print("Writing JSON...")
fp.write(json.dumps(list_result))
fp.close()

