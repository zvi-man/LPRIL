import os
import shutil
from PIL import Image
import streamlit as st


def load_images_from_dir(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            images.append(image)
    return images


def copy_images(selected_images, target_dir):
    for image_path in selected_images:
        shutil.copy2(image_path, target_dir)


def main():
    st.title("Image View -> Select -> Copy")

    # Get directory path from user
    directory = st.text_input("Enter the directory path:")

    # Get number of images to display in each row
    num_images_per_row = st.number_input("Number of Images per Row", min_value=1, value=5, step=1)

    checkbox_num = 0

    # Display images from the directory
    if directory:
        images = load_images_from_dir(directory)
        if images:
            selected_images = []
            num_images = len(images)
            num_rows = -(-num_images // num_images_per_row)  # Ceiling division

            for row in range(num_rows):
                cols = st.columns(num_images_per_row)
                for col_idx, col in enumerate(cols):
                    image_idx = row * num_images_per_row + col_idx
                    if image_idx < num_images:
                        col.image(images[image_idx], use_column_width=True)

                        # Check if image is selected
                        checkbox_num += 1
                        if col.checkbox("Select", key=checkbox_num):
                            selected_images.append(os.path.join(directory, images[image_idx].filename))

            # Select target directory for copying
            target_dir = st.text_input("Enter the target directory path:")

            # Copy selected images to target directory
            if st.button("Copy Selected Images"):
                copy_images(selected_images, target_dir)
                st.success("Selected images copied successfully!")
        else:
            st.write("No images found in the directory.")


if __name__ == "__main__":
    main()
