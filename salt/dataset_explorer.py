from pycocotools import mask
from skimage import measure
import json
import shutil
import itertools
import numpy as np
from simplification.cutil import simplify_coords_vwp
import os, cv2, copy
from distinctipy import distinctipy
from datetime import datetime
import copy
import math


def init_coco(dataset_folder, image_names, categories, coco_json_path):
    coco_json = {
        "info": {
            "description": "SAM Dataset",
            "url": "",
            "version": "1.0",
            "contributor": "Sam",
            "date_created": datetime.today().strftime("%Y-%m-%d"),
        },
        "images": [],
        "annotations": [],
        "categories": [],
    }
    for category_index, category in enumerate(categories):
        category_index += 1
        coco_json["categories"].append(
            {"id": category_index, "name": category, "supercategory": category}
        )
    for image_index, image_path in enumerate(image_names):
        image_index += 1
        im = cv2.imread(os.path.join(dataset_folder, image_path))
        image_name = os.path.basename(image_path)
        coco_json["images"].append(
            {
                "id": image_index,
                "file_name": image_name,
                "width": im.shape[1],
                "height": im.shape[0],
            }
        )
    with open(coco_json_path, "w") as f:
        json.dump(coco_json, f)


def bunch_coords(coords):
    coords_trans = []
    for i in range(0, len(coords) // 2):
        coords_trans.append([coords[2 * i], coords[2 * i + 1]])
    return coords_trans


def unbunch_coords(coords):
    return list(itertools.chain(*coords))


def bounding_box_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = []
    for contour in contours:
        all_contours.extend(contour)
    convex_hull = cv2.convexHull(np.array(all_contours))
    x, y, w, h = cv2.boundingRect(convex_hull)
    return x, y, w, h


def rot_bounding_box_from_mask(mask, arrow_pos):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = []
    for contour in contours:
        all_contours.extend(contour)

    rotated_rect = cv2.minAreaRect(np.array(all_contours))
    (center, size, angle) = rotated_rect

    box_points = cv2.boxPoints(rotated_rect)
    # sort points by Y axis
    sorted_points = sorted(box_points, key=lambda p: p[1], reverse=False)
    # find two lowest points
    lower_1 = sorted_points[0]
    lower_2 = sorted_points[1]
    # find lowest left and lowest right points
    if lower_1[0] > lower_2[0]:
        lower_left, lower_right = lower_2, lower_1
    else:
        lower_left, lower_right = lower_1, lower_2
    # calculate deltas
    dx = lower_right[0] - lower_left[0]
    dy = lower_right[1] - lower_left[1]
    # calcualte angle beetwen X axis and two lowest points
    angle_rad = np.arctan2(dy, dx)
    angle_deg = float(np.degrees(angle_rad))
    angle_deg = np.round(angle_deg, 2)

    width = size[0]
    height = size[1]

    if angle_deg <= 0:
        angle_deg += 360
        width = size[1]
        height = size[0]
    # angle need to be: [0, 360)
    if angle_deg == 360:
        angle_deg = 0

    if arrow_pos[0] != None and arrow_pos[1] != None:
        accurate_angle_deg = calculate_angle(
            (arrow_pos[0][0], arrow_pos[0][1]), (arrow_pos[1][0], arrow_pos[1][1])
        )

        all_contours = np.vstack(contours)

        M = cv2.moments(all_contours)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), accurate_angle_deg, 1.0)
        rotated_contour = cv2.transform(all_contours, rotation_matrix)

        x, y, width, height = cv2.boundingRect(rotated_contour)
        angle_deg = accurate_angle_deg


    not_rotated_left_top = (center[0] - width / 2, center[1] - height / 2)


    return not_rotated_left_top[0], not_rotated_left_top[1], width, height, angle_deg


def calculate_angle(start_point, end_point):
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad) % 360

    if dx > 0 and dy > 0:
        # angle from 90 to 180
        angle_deg = 90 + abs(angle_deg)
    if dx < 0 and dy > 0:
        # angle from 180 to 270
        angle_deg = 90 + abs(angle_deg)
    if dx <= 0 and dy < 0:
        # angle from 270 to 360
        angle_deg = 270 + abs(180 - angle_deg)
    if dx > 0 and dy < 0:
        # angle from 0 to 90
        angle_deg = 90 - abs(360 - angle_deg)

    if dx == 0 and dy <= 0:
        angle_deg = 0
    if dx >= 0 and dy == 0:
        angle_deg = 90
    if dx == 0 and dy >= 0:
        angle_deg = 180
    if dx <= 0 and dy == 0:
        angle_deg = 270

    return angle_deg


def parse_mask_to_coco(
    image_id, anno_id, image_mask, category_id, annotation_type, arrow_pos
):
    start_anno_id = anno_id
    if annotation_type == "rot-bbox":
        x, y, width, height, rotation = rot_bounding_box_from_mask(
            image_mask, arrow_pos
        )
    else:
        x, y, width, height = bounding_box_from_mask(image_mask)
        rotation = 0
    contours, _ = cv2.findContours(
        image_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    annotation = {
        "id": start_anno_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(x), float(y), float(width), float(height)],
        "area": float(width * height),
        "iscrowd": 0,
        "segmentation": [],
        "attributes": {"occluded": False, "rotation": rotation},
    }
    for contour in contours:
        sc = simplify_coords_vwp(contour[:, 0, :], 2).ravel().tolist()
        tol = 1e-3
        cleaned = []
        for x, y in zip(sc[::2], sc[1::2]):
            if x > tol and y > tol:
                cleaned.append(x)
                cleaned.append(y)
        annotation["segmentation"].append(cleaned)
    annotation["segmentation"] = [list(itertools.chain(*annotation["segmentation"]))]
    return annotation


class DatasetExplorer:
    def __init__(self, dataset_folder, categories=None, coco_json_path=None):
        self.dataset_folder = dataset_folder
        self.image_names = os.listdir(os.path.join(self.dataset_folder, "images"))
        self.image_names = [
            os.path.split(name)[1]
            for name in self.image_names
            if name.endswith(".jpg") or name.endswith(".png")
        ]
        self.coco_json_path = coco_json_path
        if not os.path.exists(coco_json_path):
            self.__init_coco_json(categories)
        with open(coco_json_path, "r") as f:
            self.coco_json = json.load(f)

        self.categories = [
            category["name"] for category in self.coco_json["categories"]
        ]
        self.annotations_by_image_id = {}
        for annotation in self.coco_json["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in self.annotations_by_image_id:
                self.annotations_by_image_id[image_id] = []
            self.annotations_by_image_id[image_id].append(annotation)

        try:
            self.global_annotation_id = (
                max(self.coco_json["annotations"], key=lambda x: x["id"])["id"] + 1
            )
        except:
            self.global_annotation_id = 1
        category_colors = distinctipy.get_colors(len(self.categories))
        category_colors = [
            tuple([int(255 * c) for c in color]) for color in category_colors
        ]
        self.category_colors = {i + 1: value for i, value in enumerate(category_colors)}

    def __init_coco_json(self, categories):
        appended_image_names = [
            os.path.join("images", name) for name in self.image_names
        ]
        init_coco(
            self.dataset_folder, appended_image_names, categories, self.coco_json_path
        )

    def get_colors(self, category_id):
        return self.category_colors[category_id]

    def get_categories(self, get_colors=False):
        if get_colors:
            return self.categories, self.category_colors
        return self.categories

    def get_num_images(self):
        return len(self.image_names)

    def get_image_data(self, image_id):
        image_name = self.coco_json["images"][image_id - 1]["file_name"]
        image_path = os.path.join(
            self.dataset_folder, os.path.join("images", image_name)
        )
        embedding_path = os.path.join(
            self.dataset_folder,
            "embeddings",
            os.path.splitext(os.path.split(image_name)[1])[0] + ".npy",
        )
        image = cv2.imread(image_path)
        image_bgr = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_embedding = np.load(embedding_path)
        return image, image_bgr, image_embedding

    def __add_to_our_annotation_dict(self, annotation):
        image_id = annotation["image_id"]
        if image_id not in self.annotations_by_image_id:
            self.annotations_by_image_id[image_id] = []
        self.annotations_by_image_id[image_id].append(annotation)

    def get_annotations(self, image_id, return_colors=False):
        if image_id not in self.annotations_by_image_id:
            return [], []
        cats = [a["category_id"] for a in self.annotations_by_image_id[image_id]]
        colors = [self.category_colors[c] for c in cats]
        if return_colors:
            return self.annotations_by_image_id[image_id], colors
        return self.annotations_by_image_id[image_id]

    def delete_annotations(self, image_id, annotation_id):
        for annotation in self.coco_json["annotations"]:
            if annotation["image_id"] == image_id and annotation["id"] == annotation_id:
                self.coco_json["annotations"].remove(annotation)
                break
        for annotation in self.annotations_by_image_id[image_id]:
            if annotation["id"] == annotation_id:
                self.annotations_by_image_id[image_id].remove(annotation)
                break

    def add_annotation(
        self, image_id, category_id, mask, annotation_type, arrow_pos
    ):
        if mask is None or not mask.any():
            return
        annotation = parse_mask_to_coco(
            image_id,
            self.global_annotation_id,
            mask,
            category_id,
            annotation_type,
            arrow_pos,
        )
        self.__add_to_our_annotation_dict(annotation)
        self.coco_json["annotations"].append(annotation)
        self.global_annotation_id += 1

    def save_annotation(self, annotation_type):
        # save global file which is needed for the system to work
        with open(self.coco_json_path, "w") as file:
            json.dump(self.coco_json, file)

        # save file for cvat with bbox annotations only
        data_to_save = copy.deepcopy(self.coco_json)
        if annotation_type == "bbox" or annotation_type == "rot-bbox":
            for annotation in data_to_save["annotations"]:
                annotation["segmentation"] = []

        base_name = os.path.splitext(os.path.basename(self.coco_json_path))[0]
        new_file_name = f"{base_name}_{annotation_type}.json"
        new_file_path = os.path.join(
            os.path.dirname(self.coco_json_path), new_file_name
        )

        with open(new_file_path, "w") as file:
            json.dump(data_to_save, file)
