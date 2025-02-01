import cv2
import numpy as np
from pycocotools import mask as coco_mask


class DisplayUtils:
    def __init__(self):
        self.transparency = 0
        self.box_width = 2

    def increase_transparency(self):
        self.transparency = min(1.0, self.transparency + 0.05)

    def decrease_transparency(self):
        self.transparency = max(0.0, self.transparency - 0.05)

    def overlay_mask_on_image(self, image, mask, color=(255, 0, 0)):
        gray_mask = mask.astype(np.uint8) * 255
        gray_mask = cv2.merge([gray_mask, gray_mask, gray_mask])
        color_mask = cv2.bitwise_and(gray_mask, color)
        masked_image = cv2.bitwise_and(image.copy(), color_mask)
        overlay_on_masked_image = cv2.addWeighted(
            masked_image, self.transparency, color_mask, 1 - self.transparency, 0
        )
        background = cv2.bitwise_and(image.copy(), cv2.bitwise_not(color_mask))
        image = cv2.add(background, overlay_on_masked_image)
        return image

    def __convert_ann_to_mask(self, ann, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        poly = ann["segmentation"]

        poly = [np.array(p, dtype=np.float32).flatten() for p in poly]

        for i in range(len(poly)):
            if len(poly[i]) == 4:
                poly[i] = np.append(poly[i], poly[i][-2:])

        rles = coco_mask.frPyObjects(poly, height, width)
        rle = coco_mask.merge(rles)
        mask_instance = coco_mask.decode(rle)
        mask_instance = np.logical_not(mask_instance)
        mask = np.logical_or(mask, mask_instance)
        mask = np.logical_not(mask)
        return mask

    def draw_box_on_image(self, image, ann, color):
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        if color == (0, 0, 0):
            image = cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
        else:
            image = cv2.rectangle(image, (x, y), (x + w, y + h), color, self.box_width)
        image = cv2.putText(
            image,
            "id: " + str(ann["id"]),
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            4,
        )
        return image

    def draw_rot_box_on_image(self, image, ann, color):
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        rotation = ann["attributes"]['rotation']

        rect_points = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32)

        # calculate rotation angle to format required by getRotationMatrix2D
        rotation = np.round(rotation, 2)
        rotation = -rotation

        center = (x + w / 2, y + h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        rotated_points = cv2.transform(np.array([rect_points]), rotation_matrix)[0]

        if color == (0, 0, 0):
            image = cv2.fillPoly(image, [np.int32(rotated_points)], color)
        else:
            image = cv2.polylines(image, [np.int32(rotated_points)], isClosed=True, color=color,
                                  thickness=self.box_width)

        image = cv2.putText(
            image,
            "id: " + str(ann["id"]),
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            4,
        )
        return image

    def draw_annotations(self, image, annotations, colors, annotation_type):
        for ann, color in zip(annotations, colors):
            if annotation_type == 'rot-bbox':
                image = self.draw_rot_box_on_image(image, ann, color)
            else:
                image = self.draw_box_on_image(image, ann, color)

            mask = self.__convert_ann_to_mask(ann, image.shape[0], image.shape[1])
            image = self.overlay_mask_on_image(image, mask, color)
        return image

    def draw_points(
        self, image, points, labels, colors={1: (0, 255, 0), 0: (0, 0, 255)}, radius=5
    ):
        for i in range(points.shape[0]):
            point = points[i, :]
            label = labels[i]
            color = colors[label]
            image = cv2.circle(image, tuple(point), radius, color, -1)
        return image
