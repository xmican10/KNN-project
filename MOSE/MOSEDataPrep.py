import os
from PIL import Image
import numpy as np
import argparse

import shutil

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def getColorsOfImg(img_path):
    # Load the image
    original_image = Image.open(img_path).convert("RGBA")

    # Convert the image into a numpy array
    image_array = np.array(original_image)

    # Find unique colors (masks) in the image by their RGBA values
    # Exclude pure black with full opacity
    unique_colors = {tuple(color) for row in image_array for color in row if tuple(color) != (0, 0, 0, 255)}

    return unique_colors

def getMaskFromColor(img_path, color):
    original_image = Image.open(img_path).convert("RGBA")

    image_array = np.array(original_image)

    mask = np.all(image_array == color, axis=-1)

    new_image_array = np.where(mask[:, :, None], 255, 0).astype(np.uint8)

    new_image = Image.fromarray(new_image_array[:, :, 0], 'L')

    return new_image

def generate_data(root_dir):
    
    # Prepare by creating new dirs
    mose_dir = os.path.join(root_dir, "MOSE")
    create_dir_if_not_exists(mose_dir)
    
    mose_dir_annotation = os.path.join(mose_dir, "Annotations")
    create_dir_if_not_exists(mose_dir_annotation)

    mose_dir_JPEGImages = os.path.join(mose_dir, "JPEGImages")
    create_dir_if_not_exists(mose_dir_JPEGImages)

    mose_dir_imageSets = os.path.join(mose_dir, "ImageSets")
    create_dir_if_not_exists(mose_dir_imageSets)

    # existedFile
    exist_annotation_train_dir = os.path.join(root_dir, "train/train/Annotations")
    exist_JPEG_train_dir = os.path.join(root_dir, "train/train/JPEGImages")

    sequencies = [seq for seq in os.listdir(exist_annotation_train_dir) if not seq.startswith('.')]
    
    file_train_path = os.path.join(mose_dir_imageSets, "train.txt")
    with open(file_train_path, "w") as file_train:
        for seq in sequencies:

            # create output direction
            mose_dir_annotation_seq = os.path.join(mose_dir_annotation, seq)
            create_dir_if_not_exists(mose_dir_annotation_seq)
            mose_dir_JPEGImages_seq = os.path.join(mose_dir_JPEGImages, seq)
            create_dir_if_not_exists(mose_dir_JPEGImages_seq)

            exist_annotation_train_dir_seq = os.path.join(exist_annotation_train_dir, seq)
            exist_real_train_dir_seq = os.path.join(exist_JPEG_train_dir, seq)
            
            frames_names_real = sorted([fn for fn in os.listdir(os.path.join(exist_JPEG_train_dir, seq)) if not fn.startswith('.')])
            frames_names_annotation = sorted([fn for fn in os.listdir(exist_annotation_train_dir_seq) if not fn.startswith('.')])

            print(frames_names_real)

            first_annotation_file = os.path.join(exist_annotation_train_dir_seq, frames_names_annotation[0])
            unique_color = getColorsOfImg(first_annotation_file)

            # copy the real frames into the results directory
            for real_frame in frames_names_real:
                src_path = os.path.join(exist_real_train_dir_seq, real_frame)
                dst_path = os.path.join(mose_dir_JPEGImages_seq, real_frame)
                shutil.copy(src_path, dst_path)

            for idx, color in enumerate(unique_color):
                for idx_frame, frame_name in enumerate(frames_names_annotation):
                    analyze_frame = os.path.join(exist_annotation_train_dir_seq, frame_name)
                    print(analyze_frame)
                    new_mask_img = getMaskFromColor(analyze_frame, color)
                    result_name = f'{(frame_name.split("."))[0]}_obj{idx}.png'
                    result_path = os.path.join(mose_dir_annotation_seq, result_name)
                
                    new_mask_img.save(result_path)
                    file_train.write(f"/JPEGImages/{seq}/{frames_names_real[idx_frame]} /Annotations/{seq}/{result_name}\n")


                    print(result_path)

    return


def generate_valid_data(root_dir):
    
    # Prepare by creating new dirs
    mose_dir = os.path.join(root_dir, "MOSE_valid")
    create_dir_if_not_exists(mose_dir)
    
    mose_dir_annotation = os.path.join(mose_dir, "Annotations")
    create_dir_if_not_exists(mose_dir_annotation)

    mose_dir_JPEGImages = os.path.join(mose_dir, "JPEGImages")
    create_dir_if_not_exists(mose_dir_JPEGImages)

    mose_dir_imageSets = os.path.join(mose_dir, "ImageSets")
    create_dir_if_not_exists(mose_dir_imageSets)

    # existedFile
    exist_annotation_train_dir = os.path.join(root_dir, "valid/Annotations")
    exist_JPEG_train_dir = os.path.join(root_dir, "valid/JPEGImages")

    sequencies = [seq for seq in os.listdir(exist_annotation_train_dir) if not seq.startswith('.')]
    
    file_train_path = os.path.join(mose_dir_imageSets, "valid.txt")
    with open(file_train_path, "w") as file_train:
        for seq in sequencies:

            # create output direction
            mose_dir_annotation_seq = os.path.join(mose_dir_annotation, seq)
            create_dir_if_not_exists(mose_dir_annotation_seq)
            mose_dir_JPEGImages_seq = os.path.join(mose_dir_JPEGImages, seq)
            create_dir_if_not_exists(mose_dir_JPEGImages_seq)

            exist_annotation_train_dir_seq = os.path.join(exist_annotation_train_dir, seq)
            exist_real_train_dir_seq = os.path.join(exist_JPEG_train_dir, seq)
            
            frames_names_real = sorted([fn for fn in os.listdir(os.path.join(exist_JPEG_train_dir, seq)) if not fn.startswith('.')])
            frames_names_annotation = sorted([fn for fn in os.listdir(exist_annotation_train_dir_seq) if not fn.startswith('.')])

            print(frames_names_real)

            first_annotation_file = os.path.join(exist_annotation_train_dir_seq, frames_names_annotation[0])
            unique_color = getColorsOfImg(first_annotation_file)

            # copy the real frames into the results directory
            for real_frame in frames_names_real:
                src_path = os.path.join(exist_real_train_dir_seq, real_frame)
                dst_path = os.path.join(mose_dir_JPEGImages_seq, real_frame)
                shutil.copy(src_path, dst_path)

            for idx, color in enumerate(unique_color):
                for idx_frame, frame_name in enumerate(frames_names_annotation):
                    analyze_frame = os.path.join(exist_annotation_train_dir_seq, frame_name)
                    print(analyze_frame)
                    new_mask_img = getMaskFromColor(analyze_frame, color)
                    result_name = f'{(frame_name.split("."))[0]}_obj{idx}.png'
                    result_path = os.path.join(mose_dir_annotation_seq, result_name)
                
                    new_mask_img.save(result_path)
                    file_train.write(f"/JPEGImages/{seq} /Annotations/{seq}/{result_name}\n")


                    print(result_path)
        

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--directory', type=str)

args = parser.parse_args()

if args.directory:
    generate_valid_data(args.directory)
