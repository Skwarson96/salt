import cv2
from PyQt5.QtWidgets import (
    QScrollArea,
    QWidget,
    QVBoxLayout,
    QLabel,
    QGraphicsView,
    QGraphicsScene,
    QApplication,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent
from PyQt5.QtCore import Qt, QRectF, QPoint
from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QRadioButton,
)

selected_annotations = []


class CustomGraphicsView(QGraphicsView):
    def __init__(self, editor):
        super(CustomGraphicsView, self).__init__()

        self.editor = editor
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.TextAntialiasing)

        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setInteractive(True)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.start_point = QPoint()
        self.end_point = QPoint()
        self.drawing = False
        self.display_image = None
        self.curr_image = None
        self.is_box_added = True

    def set_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        if self.display_image:
            self.display_image.setPixmap(pixmap)
            self.setSceneRect(QRectF(pixmap.rect()))
        else:
            self.display_image = self.scene.addPixmap(pixmap)
            self.setSceneRect(QRectF(pixmap.rect()))
        # self.curr_image = self.editor.image.copy()

    def wheelEvent(self, event: QWheelEvent):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            adj = (event.angleDelta().y() / 120) * 0.1
            self.scale(1 + adj, 1 + adj)
        else:
            delta_y = event.angleDelta().y()
            delta_x = event.angleDelta().x()
            x = self.horizontalScrollBar().value()
            self.horizontalScrollBar().setValue(x - delta_x)
            y = self.verticalScrollBar().value()
            self.verticalScrollBar().setValue(y - delta_y)

    def imshow(self, img):
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(
            img.data, width, height, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        self.set_image(q_img)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # FUTURE USE OF RIGHT CLICK EVENT IN THIS AREA
        modifiers = QApplication.keyboardModifiers()
        label = None
        if modifiers == Qt.ControlModifier:
            print("Control/ Command key pressed during a mouse click")
            # self.editor.remove_click([int(x), int(y)])
        else:
            pos = event.pos()
            pos_in_item = self.mapToScene(pos) - self.display_image.pos()
            x, y = pos_in_item.x(), pos_in_item.y()
            if event.button() == Qt.LeftButton:
                label = 1
                if self.editor.prompt_type == "box" and self.is_box_added:
                    self.drawing = True
                    self.start_point = event.pos()
                    self.end_point = event.pos()
                    self.is_box_added = False
            elif event.button() == Qt.RightButton:
                label = 0
            if label is not None:
                if self.editor.prompt_type == "point":
                    self.editor.add_click([int(x), int(y)], label, selected_annotations)
        self.imshow(self.editor.display)

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            temp_image = self.editor.display.copy()
            cv2.rectangle(
                temp_image,
                (self.start_point.x(), self.start_point.y()),
                (self.end_point.x(), self.end_point.y()),
                (0, 255, 0),
                2,
            )
            self.imshow(temp_image)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.editor.prompt_type == "box" and self.drawing:
                self.drawing = False
                self.end_point = event.pos()
                cv2.rectangle(
                    self.editor.display,
                    (self.start_point.x(), self.start_point.y()),
                    (self.end_point.x(), self.end_point.y()),
                    (0, 255, 0),
                    2,
                )
                label = 1
                self.editor.add_click(
                    [
                        self.start_point.x(),
                        self.start_point.y(),
                        self.end_point.x(),
                        self.end_point.y(),
                    ],
                    label,
                    selected_annotations,
                )
                self.imshow(self.editor.display)


class ApplicationInterface(QWidget):
    def __init__(self, app, editor, panel_size=(1920, 1080)):
        super(ApplicationInterface, self).__init__()
        self.app = app
        self.editor = editor
        self.panel_size = panel_size

        self.layout = QVBoxLayout()

        self.top_bar = self.get_top_bar()
        self.layout.addWidget(self.top_bar)

        self.main_window = QHBoxLayout()

        self.graphics_view = CustomGraphicsView(editor)
        self.main_window.addWidget(self.graphics_view)

        self.panel = self.get_side_panel()
        self.panel_annotations = QListWidget()
        self.panel_annotations.setFixedWidth(200)
        self.panel_annotations.setSelectionMode(QAbstractItemView.MultiSelection)
        self.panel_annotations.itemClicked.connect(self.annotation_list_item_clicked)
        self.get_side_panel_annotations()
        self.main_window.addWidget(self.panel)
        self.main_window.addWidget(self.panel_annotations)

        self.layout.addLayout(self.main_window)

        self.setLayout(self.layout)

        self.graphics_view.imshow(self.editor.display)

    def reset(self):
        global selected_annotations
        self.editor.reset(selected_annotations)
        self.graphics_view.is_box_added = True
        self.graphics_view.imshow(self.editor.display)

    def add(self):
        global selected_annotations
        self.editor.save_ann()
        self.editor.reset(selected_annotations)
        self.graphics_view.is_box_added = True
        self.graphics_view.imshow(self.editor.display)

    def next_image(self):
        global selected_annotations
        self.editor.next_image()
        selected_annotations = []
        self.graphics_view.imshow(self.editor.display)

    def prev_image(self):
        global selected_annotations
        self.editor.prev_image()
        selected_annotations = []
        self.graphics_view.imshow(self.editor.display)

    def toggle(self):
        global selected_annotations
        self.editor.toggle(selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def transparency_up(self):
        global selected_annotations
        self.editor.step_up_transparency(selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def transparency_down(self):
        self.editor.step_down_transparency(selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def save_all(self):
        self.editor.save()

    def change_prompt_type(self):
        self.editor.change_prompt_type()

    def get_top_bar(self):
        top_bar = QWidget()
        button_layout = QHBoxLayout(top_bar)
        self.layout.addLayout(button_layout)
        buttons = [
            ("Add", lambda: self.add()),
            ("Reset", lambda: self.reset()),
            ("Prev", lambda: self.prev_image()),
            ("Next", lambda: self.next_image()),
            ("Toggle", lambda: self.toggle()),
            ("Transparency Up", lambda: self.transparency_up()),
            ("Transparency Down", lambda: self.transparency_down()),
            ("Save", lambda: self.save_all()),
            ("Change prompt type", lambda: self.change_prompt_type()),
            (
                "Remove Selected Annotations",
                lambda: self.delete_annotations(),
            ),
        ]
        for button, lmb in buttons:
            bt = QPushButton(button)
            bt.clicked.connect(lmb)
            button_layout.addWidget(bt)

        return top_bar

    def get_side_panel(self):
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        categories, colors = self.editor.get_categories(get_colors=True)
        label_array = []

        for category_index, category in enumerate(categories):
            label_array.append(QRadioButton(category))
            label_array[category_index].clicked.connect(
                lambda state, x=category: self.editor.select_category(x)
            )

            color = colors[category_index + 1]
            label_array[category_index].setStyleSheet(
                "background-color: rgba({},{},{},0.6)".format(*color[::-1])
            )
            panel_layout.addWidget(label_array[category_index])

        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(panel)
        scroll.setFixedWidth(200)
        return scroll

    def get_side_panel_annotations(self):
        anns, colors = self.editor.list_annotations()
        list_widget = self.panel_annotations
        list_widget.clear()
        categories = self.editor.get_categories(get_colors=False)
        for i, ann in enumerate(anns):
            listWidgetItem = QListWidgetItem(
                str(ann["id"]) + " - " + (categories[ann["category_id"]-1])
            )
            list_widget.addItem(listWidgetItem)
        return list_widget

    def delete_annotations(self):
        global selected_annotations
        for annotation in selected_annotations:
            self.editor.delete_annotations(annotation)
        self.get_side_panel_annotations()
        selected_annotations = []
        self.reset()

    def annotation_list_item_clicked(self, item):
        global selected_annotations
        if item.isSelected():
            selected_annotations.append(int(item.text().split(" ")[0]))
            self.editor.draw_selected_annotations(selected_annotations)
        else:
            selected_annotations.remove(int(item.text().split(" ")[0]))
            self.editor.draw_selected_annotations(selected_annotations)
        self.graphics_view.imshow(self.editor.display)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.app.quit()
        if event.key() == Qt.Key_A:
            self.prev_image()
            self.get_side_panel_annotations()
        if event.key() == Qt.Key_D:
            self.next_image()
            self.get_side_panel_annotations()
        if event.key() == Qt.Key_K:
            self.transparency_down()
        if event.key() == Qt.Key_L:
            self.transparency_up()
        if event.key() == Qt.Key_N:
            self.add()
            self.get_side_panel_annotations()
        if event.key() == Qt.Key_R:
            self.reset()
        if event.key() == Qt.Key_T:
            self.toggle()
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
            self.save_all()
        elif event.key() == Qt.Key_Space:
            print("Space pressed")
            # self.clear_annotations(selected_annotations)
            # Do something if the space bar is pressed
            # pass
