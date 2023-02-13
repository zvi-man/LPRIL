import os
import tkinter as tk
import tkinter.ttk as ttk
import pandas as pd
from PIL import Image, ImageTk

# Constants
DEFAULT_CSV_PATH = r"images.csv"
DEFAULT_IMG_DIR = r"D:\עבודה צבי\VehicleColorClassification\DataSets\HomeMade1\raanana\car_images"
IMAGE_NAME_COL = 'file_name'
IMAGE_COLOR_COL = "color"
IMAGE_MAKE_COL = "make"
IMAGE_TEXT_COL = 'image_string'
TAG_COL = "tag"
UNLABELED_STR = 'UN'
GOOD_STR = "Good"
WRONG_STR = "Wrong"


class App:
    def __init__(self, master, csv_path: str = DEFAULT_CSV_PATH, image_dir: str = DEFAULT_IMG_DIR):
        self.master = master
        self.csv_file = csv_path
        self.image_dir = image_dir
        self.current_index = 0
        self.data = pd.DataFrame()
        self.load_data()

        # Create widgets
        self.image_name_widget = ttk.Label(master, text="")
        self.picture_widget = ttk.Label(master)
        self.predicted_label_widget = ttk.Label(master, text="")
        self.prev_button = tk.Button(master, text="Previous", command=self.prev_image)
        self.next_button = tk.Button(master, text="Next", command=self.next_image)
        self.jump_label = ttk.Label(master, text="Jump to:")
        self.jump_box_val = tk.StringVar(self.master)
        self.jump_box = tk.Spinbox(master, from_=0, to=len(self.data) - 1,
                                   textvariable=self.jump_box_val, command=self.jump_to)
        self.good_button = tk.Button(master, text="Good!", command=self.tag_good, fg='green')
        self.un_label_button = tk.Button(master, text="UnLabel", command=self.tag_unlabeled)
        self.wrong_button = tk.Button(master, text="Wrong!", command=self.tag_wrong, fg='red')

        self.image_name_widget.grid(row=0, column=0, columnspan=2)
        self.predicted_label_widget.grid(row=0, column=2, columnspan=2)
        self.picture_widget.grid(row=2, column=0, columnspan=4)
        self.prev_button.grid(row=3, column=0, sticky=tk.E + tk.W)
        self.jump_label.grid(row=3, column=1)
        self.jump_box.grid(row=3, column=2)
        self.next_button.grid(row=3, column=3, sticky=tk.E + tk.W)
        self.good_button.grid(row=4, column=0, sticky=tk.E + tk.W)
        self.wrong_button.grid(row=4, column=1, sticky=tk.E + tk.W)
        self.un_label_button.grid(row=4, column=2, columnspan=2, sticky=tk.E + tk.W)
        self.update_image()

    def load_data(self):
        self.data = pd.read_csv(self.csv_file)
        self.data[IMAGE_TEXT_COL] = self.data[IMAGE_COLOR_COL] + "_" + self.data[IMAGE_MAKE_COL]
        if TAG_COL not in self.data.columns:
            self.data[TAG_COL] = UNLABELED_STR

    def update_image(self):
        if self.current_index >= len(self.data):
            self.current_index = 0
        elif self.current_index < 0:
            self.current_index = len(self.data) - 1
        image_string = self.data.iloc[self.current_index][IMAGE_TEXT_COL]
        image_name = self.data.iloc[self.current_index][IMAGE_NAME_COL]
        image_tag =  self.data.iloc[self.current_index][TAG_COL]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path)
        image = image.resize((500, 500))
        image = ImageTk.PhotoImage(image)

        self.picture_widget.config(image=image)
        self.picture_widget.image = image

        self.predicted_label_widget.config(text=image_string)
        self.image_name_widget.config(text=image_name)
        self.jump_box_val.set(self.current_index)
        self.reset_tag_buttons()

        if image_tag == UNLABELED_STR:
            self.highlight_button(self.un_label_button)
        if image_tag == GOOD_STR:
            self.highlight_button(self.good_button)
        if image_tag == WRONG_STR:
            self.highlight_button(self.wrong_button)

    @staticmethod
    def highlight_button(button: tk.Widget) -> None:
        button['borderwidth'] = 8

    def reset_tag_buttons(self):
        self.good_button['borderwidth'] = 2
        self.wrong_button['borderwidth'] = 2
        self.un_label_button['borderwidth'] = 2

    def next_image(self):
        self.current_index += 1
        self.update_image()

    def prev_image(self):
        self.current_index -= 1
        self.update_image()

    def jump_to(self):
        self.current_index = int(self.jump_box.get())
        self.update_image()

    def tag_good(self) -> None:
        self.reset_tag_buttons()
        self.highlight_button(self.good_button)
        self.save_tag(GOOD_STR)
        self.next_image()

    def tag_wrong(self) -> None:
        self.reset_tag_buttons()
        self.highlight_button(self.wrong_button)
        self.save_tag(WRONG_STR)
        self.next_image()

    def tag_unlabeled(self) -> None:
        self.reset_tag_buttons()
        self.highlight_button(self.un_label_button)
        self.save_tag(UNLABELED_STR)
        self.next_image()

    def save_tag(self, tag: str) -> None:
        self.data.loc[self.current_index, TAG_COL] = tag
        self.data.to_csv(self.csv_file, index=False)


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
