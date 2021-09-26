import os
import shutil

import cv2
import numpy as np
from PIL import Image

from skimage.draw import ellipse

# Dinh nghia bien
data_path = "dataset"

# Tap map cho cac anh san pham khong bi loi
def make_map_normal(data_path):
    # Loop through normal folder
    for folder in os.listdir(data_path):
        if (not folder.endswith("def")) and (not folder.startswith(".")) and (not folder.endswith("mask")):
            print("*" * 10, folder)
            # Make mask folder
            mask_folder = folder + "_mask"
            mask_folder = os.path.join(data_path, mask_folder)
            try:
                shutil.rmtree(mask_folder)
            except:
                pass

            os.mkdir(mask_folder)

            # Loop through file in current folder:
            current_folder = os.path.join(data_path, folder)
            for file in os.listdir(current_folder):
                if file.endswith("png"):
                    print(file)
                    # Read image file
                    current_file = os.path.join(current_folder, file)
                    image = cv2.imread(current_file)
                    w, h = image.shape[0], image.shape[1]

                    # Make mask file for normal product - it's blank image, no defect
                    mask_image = np.zeros((w, h), dtype=np.uint8)
                    mask_image = Image.fromarray(mask_image)

                    # Save the file
                    mask_image.save(os.path.join(mask_folder, file))


# Ve vung loi len tren anh map mau den
def draw_defect(file, labels, w, h):
    # Lấy file id
    file_id = int(file.replace(".png", ""))

    # Lấy nhãn của file
    label = labels[file_id - 1]

    # Tách các thành phần trong nhãn
    label = label.replace("\t", "").replace("  ", " ").replace("  ", " ").replace("\n", "")
    label_array = label.split(" ")

    # Vẽ hình ellipse
    major, minor, angle, x_pos, y_pos = float(label_array[1]), float(label_array[2]), float(label_array[3]), float(
        label_array[4]), float(label_array[5])
    rr, cc = ellipse(y_pos, x_pos, r_radius=minor, c_radius=major, rotation=-angle)

    # Tạo ảnh màu đen
    mask_image = np.zeros((w, h), dtype=np.uint8)

    try:
        # Gán các điểm thuộc hình ellipse thành 1
        mask_image[rr, cc] = 1
    except:
        # Nếu lỗi chỉ gán các điểm trong ảnh
        rr_n = [min(511, rr[i]) for i in rr]
        cc_n = [min(511, cc[i]) for i in cc]
        mask_image[rr_n, cc_n] = 1
        # mask_image = Image.fromarray(mask_image)

    # Chuyển thành ảnh
    mask_image = np.array(mask_image, dtype=np.uint8)
    mask_image = Image.fromarray(mask_image)

    return mask_image

# Tao map cho cac anh san pham bi loi
def make_map_defect(data_path):
    # Loop through defect folder
    for folder in os.listdir(data_path):
        if (folder.endswith("def")) and (not folder.startswith(".")):
            print("*" * 10, folder)

            # Make mask folder
            mask_folder = folder + "_mask"
            mask_folder = os.path.join(data_path, mask_folder)
            try:
                shutil.rmtree(mask_folder)
            except:
                pass

            os.mkdir(mask_folder)

            # Loop through file in current folder:
            current_folder = os.path.join(data_path, folder)

            # Load txt file
            f = open(os.path.join(current_folder, 'labels.txt'))
            labels = f.readlines()
            f.close()

            for file in os.listdir(current_folder):
                if file.find("(") > -1:
                    # Xoá file nếu bị trùng (do đặc thù dữ liệu)
                    os.remove(os.path.join(current_folder, file))
                    continue

                if file.endswith("png"):
                    print(file)
                    # Read image file
                    current_file = os.path.join(current_folder, file)
                    image = cv2.imread(current_file)
                    w, h = image.shape[0], image.shape[1]

                    # Make mask file for defect product - it's blank image with defect
                    mask_image = draw_defect(file, labels, w, h)

                    # Save the file
                    mask_image.save(os.path.join(mask_folder, file))
