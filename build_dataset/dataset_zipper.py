from argparse import ArgumentParser
import os
import glob
import numpy as np
import pydicom as dicom
import cv2

ROOT_FOLDER = '/data_new3/username/DL/PLA_data_bak/denoised/train'
ROOT_FOLDER2 = '/data_new3/username/DL/PLA_data/denoised/test'
MASK_PATH = '/data_new3/username/DL/scripts/result.tif'
SUB_FOLDER = ['100kv', '140kv']
START_SLICE = [10, 120, 150, 70, 15, 35, 35, 20, 15, 75, 35, 53, 45, 10, 90, 25]

def apply_mask(image, mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    masked_image = image.copy()
    masked_image[mask == 0] = 0
    return masked_image.astype(np.float32)

def save_patient_data(root_folder, save_path="/data_new3/username/DualEnergyCTSynthesis/dataset", dataset_type='train'):
    print(root_folder)
    patient_names = sorted([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])
    patient_id=0
    #print(patient_names)

    for i, patient_name in enumerate(patient_names):
        patient_folder = os.path.join(root_folder, patient_name)
        data_files = glob.glob(os.path.join(patient_folder, SUB_FOLDER[0])+ '/*.IMA')
        label_files = glob.glob(os.path.join(patient_folder, SUB_FOLDER[1])+ '/*.IMA')
        data_files.sort()
        label_files.sort()
        #print(data_files)

        if len(data_files) != len(label_files):
            raise RuntimeError("Unequal number between data files and label files!")
        patient_id = i
        if dataset_type == 'train' and patient_id >=13:
            dataset_type = 'valid'
        for j, (data_file, label_file) in enumerate(zip(data_files, label_files)):
            if j < START_SLICE[i] and dataset_type == 'train':
                continue
            data_dcm = dicom.read_file(data_file)
            label_dcm = dicom.read_file(label_file)
            data = apply_mask(data_dcm.pixel_array, MASK_PATH)
            label = apply_mask(label_dcm.pixel_array, MASK_PATH)

            file_name = f"{dataset_type}_{patient_id:02d}_{j + 1:03d}.npy"
            print(f"Saving {file_name}...")
            np.save(os.path.join(save_path, file_name), {'data': data, 'label': label, 'patient_id': patient_id, 'image_id': j + 1})


def main_func(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_patient_data(ROOT_FOLDER,save_path, 'train')
    save_patient_data(ROOT_FOLDER2,save_path, 'test')

    print("Data saved.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-path", type=str, default="/data_new3/username/DualEnergyCTSynthesis/dataset",
                        help="Path to save npy files.")
    args = parser.parse_args()
    main_func(**vars(args))
