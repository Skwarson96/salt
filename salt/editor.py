import os, copy
import numpy as np
from salt.onnx_model import OnnxModels
from salt.dataset_explorer import DatasetExplorer
from salt.display_utils import DisplayUtils


class CurrentCapturedInputs:
    def __init__(self):
        self.input_point = np.array([])
        self.input_box = np.array([])
        self.input_label = np.array([])
        self.low_res_logits = None
        self.curr_mask = None

    def reset_inputs(self):
        self.input_point = np.array([])
        self.input_box = np.array([])
        self.input_label = np.array([])
        self.low_res_logits = None
        self.curr_mask = None

    def set_mask(self, mask):
        self.curr_mask = mask

    def add_input_click(self, input_data, input_label, prompt_type):
        # self.input_box = None
        if prompt_type == "point":
            if len(self.input_point) == 0:
                self.input_point = np.array([input_data])
            else:
                self.input_point = np.vstack([self.input_point, np.array([input_data])])
            self.input_label = np.append(self.input_label, input_label)
        if prompt_type == "box":
            self.input_box = np.array([input_data])
            self.input_label = np.append(self.input_label, input_label)

    def set_low_res_logits(self, low_res_logits):
        self.low_res_logits = low_res_logits


class Editor:
    def __init__(
        self, onnx_models_path, dataset_path, annotation_type, categories=None, coco_json_path=None
    ):
        self.dataset_path = dataset_path
        self.coco_json_path = coco_json_path
        self.annotation_type = annotation_type

        if categories is None and not os.path.exists(coco_json_path):
            raise ValueError("categories must be provided if coco_json_path is None")
        if self.coco_json_path is None:
            self.coco_json_path = os.path.join(self.dataset_path, "annotations.json")
        self.dataset_explorer = DatasetExplorer(
            self.dataset_path, categories=categories, coco_json_path=self.coco_json_path
        )
        self.curr_inputs = CurrentCapturedInputs()
        self.categories, self.category_colors = self.dataset_explorer.get_categories(
            get_colors=True
        )
        self.image_id = 1
        self.category_id = 1
        self.show_other_anns = True
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        self.onnx_helper = OnnxModels(
            onnx_models_path,
            image_width=self.image.shape[1],
            image_height=self.image.shape[0],
        )
        self.du = DisplayUtils()
        self.reset()
        self.prompt_type = "point"
        self.arrow_exist = False
        self.drawing_arrow = False
        self.arrow_pos = [None, None] # [start point, end point]

    def list_annotations(self):
        anns, colors = self.dataset_explorer.get_annotations(
            self.image_id, return_colors=True
        )
        return anns, colors

    def delete_annotations(self, annotation_id):
        self.dataset_explorer.delete_annotations(self.image_id, annotation_id)

    def __draw_known_annotations(self, selected_annotations=[]):
        anns, colors = self.dataset_explorer.get_annotations(
            self.image_id, return_colors=True
        )
        for i, (ann, color) in enumerate(zip(anns, colors)):
            for selected_ann in selected_annotations:
                if ann["id"] == selected_ann:
                    colors[i] = (0, 0, 0)
        # Use this to list the annotations
        self.display = self.du.draw_annotations(self.display, anns, colors, self.annotation_type)

    def __draw(self, selected_annotations=[]):
        self.display = self.image_bgr.copy()
        if self.curr_inputs.curr_mask is not None:
            if self.prompt_type == "point":
                self.display = self.du.draw_points(
                    self.display,
                    self.curr_inputs.input_point,
                    self.curr_inputs.input_label,
                )
            self.display = self.du.overlay_mask_on_image(
                self.display, self.curr_inputs.curr_mask
            )
        if self.show_other_anns:
            self.__draw_known_annotations(selected_annotations)

    def add_click(self, new_pt, new_label, selected_annotations=[]):
        self.curr_inputs.add_input_click(new_pt, new_label, self.prompt_type)
        masks, low_res_logits = self.onnx_helper.call(
            self.image,
            self.image_embedding,
            self.curr_inputs.input_point,
            self.curr_inputs.input_box,
            self.curr_inputs.input_label,
            low_res_logits=self.curr_inputs.low_res_logits,
        )
        if masks is not None and low_res_logits is not None:
            self.curr_inputs.set_mask(masks[0, 0, :, :])
            self.curr_inputs.set_low_res_logits(low_res_logits)
            self.__draw(selected_annotations)

    def remove_click(self, new_pt):
        print("ran remove click")

    def reset(self, hard=True, selected_annotations=[]):
        self.curr_inputs.reset_inputs()
        self.arrow_pos = [None, None]
        self.__draw(selected_annotations)

    def toggle(self, selected_annotations=[]):
        self.show_other_anns = not self.show_other_anns
        self.__draw(selected_annotations)

    def step_up_transparency(self, selected_annotations=[]):
        self.display = self.image_bgr.copy()
        self.du.increase_transparency()
        self.__draw(selected_annotations)

    def step_down_transparency(self, selected_annotations=[]):
        self.display = self.image_bgr.copy()
        self.du.decrease_transparency()
        self.__draw(selected_annotations)

    def draw_selected_annotations(self, selected_annotations=[]):
        self.__draw(selected_annotations)

    def save_ann(self):
        # self.arrow_pos = [None, None]
        self.dataset_explorer.add_annotation(
            self.image_id, self.category_id, self.curr_inputs.curr_mask, self.annotation_type, self.arrow_pos
        )

    def save(self):
        self.dataset_explorer.save_annotation(self.annotation_type)

    def change_prompt_type(self):
        if self.prompt_type == "point":
            self.prompt_type = "box"
        else:
            self.prompt_type = "point"
        print(f"Prompt type: {self.prompt_type}")

    def next_image(self):
        if self.image_id == self.dataset_explorer.get_num_images():
            return
        self.image_id += 1
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        self.onnx_helper.set_image_resolution(self.image.shape[1], self.image.shape[0])
        self.reset()

    def prev_image(self):
        if self.image_id == 1:
            return
        self.image_id -= 1
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        self.onnx_helper.set_image_resolution(self.image.shape[1], self.image.shape[0])
        self.reset()

    def next_category(self):
        if self.category_id == len(self.categories):
            self.category_id = 1
            return
        self.category_id += 1

    def prev_category(self):
        if self.category_id == 1:
            self.category_id = len(self.categories)
            return
        self.category_id -= 1

    def get_categories(self, get_colors=False):
        if get_colors:
            return self.categories, self.category_colors
        return self.categories

    def select_category(self, category_name):
        category_id = self.categories.index(category_name) + 1
        self.category_id = category_id
