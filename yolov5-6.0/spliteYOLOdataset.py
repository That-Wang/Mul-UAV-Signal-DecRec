import os
import shutil
import re

def copy_files_by_class(source_image_dir, source_label_dir, target_image_dir, target_label_dir, target_classes):
    # 确保目标文件夹存在
    os.makedirs(target_image_dir, exist_ok=True)
    os.makedirs(target_label_dir, exist_ok=True)

    # 遍历源图像文件夹中的所有文件
    for filename in os.listdir(source_image_dir):
        if filename.endswith('.png'):
            # 提取类别数字
            match = re.search(r'_([0-9]+)_snr', filename)
            if match:
                class_number = int(match.group(1))
                if class_number in target_classes:
                    # 构建完整的文件路径
                    image_path = os.path.join(source_image_dir, filename)
                    label_filename = filename.replace('.png', '.txt')
                    label_path = os.path.join(source_label_dir, label_filename)

                    # 复制文件到目标文件夹
                    shutil.copy(image_path, target_image_dir)
                    shutil.copy(label_path, target_label_dir)
                    print(f"Copied {filename} and {label_filename}")

                    # 修改目标文件夹中的标签文件
                    # modify_label_file(target_label_dir)

def modify_label_file(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()

    with open(label_path, 'w') as file:
        for line in lines:
            parts = line.strip().split()
            if parts:
                # 将第一列标签减去5
                parts[0] = str(int(parts[0]) - 5)
                file.write(' '.join(parts) + '\n')

def modify_label_files_in_dir(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            label_path = os.path.join(directory, filename)
            modify_label_file(label_path)


def rename_files(directory, suffix):
    for filename in os.listdir(directory):
        if filename.endswith(f"{suffix}.txt") or filename.endswith(f"{suffix}.png"):
            base_name, ext = os.path.splitext(filename)
            new_filename = base_name
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed {filename} to {new_filename}")


def restore_files_images(directory):
    for filename in os.listdir(directory):
        if not filename.endswith('.png'):
            # 处理 .png 文件
            new_filename_png = f"{filename}.png"
            old_path_png = os.path.join(directory, filename)
            new_path_png = os.path.join(directory, new_filename_png)
            if os.path.exists(new_path_png):
                print(f"File {new_filename_png} already exists, skipping.")
                continue
            os.rename(old_path_png, new_path_png)
            print(f"Restored {filename} to {new_filename_png}")


def restore_files_labels(directory):
    for filename in os.listdir(directory):
        if not filename.endswith('.txt'):
            new_filename = f"{filename}.txt"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            if os.path.exists(new_path):
                print(f"File {new_filename} already exists, skipping.")
                continue
            os.rename(old_path, new_path)
            print(f"Restored {filename} to {new_filename}")



if __name__ == "__main__":
    source_image_dir = 'D:\\download\\UAVDataset\\rectrain_yolo\\images'
    source_label_dir = 'D:\\download\\UAVDataset\\rectrain_yolo\\labels'
    target_image_dir = 'D:\\download\\UAVDataset\\rectrain_yolo\\images_part'
    target_label_dir = 'D:\\download\\UAVDataset\\rectrain_yolo\\labels_part'
    target_classes = {5, 6, 7, 8, 9}

    # 修改标签文件
    # modify_label_files_in_dir(target_label_dir)
    # 复制文件
    # copy_files_by_class(source_image_dir, source_label_dir, target_image_dir, target_label_dir, target_classes)


    # suffixes_to_rename = 'DA4'
    # # 重命名文件
    # rename_files(source_image_dir, suffixes_to_rename)
    # rename_files(source_label_dir, suffixes_to_rename)
    # 恢复文件名
    restore_files_images(source_image_dir)
    restore_files_labels(source_label_dir)

