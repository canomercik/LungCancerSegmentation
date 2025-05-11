import os
import pydicom
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from skimage.draw import polygon
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
import random

def create_directories():
    """Train/Val klasörlerini oluştur"""
    os.makedirs('train/images', exist_ok=True)
    os.makedirs('train/masks', exist_ok=True)
    os.makedirs('val/images', exist_ok=True)
    os.makedirs('val/masks', exist_ok=True)


def find_all_dicom_dirs(base_dir):
    """Tüm hasta dizinlerini bul"""
    dicom_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if any(f.endswith('.dcm') for f in files) and any(f.endswith('.xml') for f in files):
            dicom_dirs.append(root)
    return dicom_dirs


def is_ct_and_axial(ds):
    """CT ve axial düzlem kontrolü"""
    is_ct = (ds.Modality == "CT") if hasattr(ds, "Modality") else False
    is_axial = np.allclose(
        ds.ImageOrientationPatient,
        [1, 0, 0, 0, 1, 0],
        atol=1e-1
    ) if hasattr(ds, "ImageOrientationPatient") else False
    return is_ct and is_axial


def load_and_preprocess_dicom(dicom_path):
    """DICOM'ı yükle ve HU penceresi uygula"""
    try:
        ds = pydicom.dcmread(dicom_path)
        if not is_ct_and_axial(ds):
            return None, None

        hu_image = ds.pixel_array.astype(np.float32)
        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
            hu_image = hu_image * ds.RescaleSlope + ds.RescaleIntercept

        hu_min, hu_max = -200, 200
        hu_image = np.clip(hu_image, hu_min, hu_max)
        hu_image = (hu_image - hu_min) / (hu_max - hu_min) * 255

        return hu_image.astype(np.uint8), ds.SOPInstanceUID
    except Exception as e:
        print(f"DICOM hatası: {dicom_path} - {str(e)}")
        return None, None


def parse_all_annotations(xml_path):
    """XML'den maskeleri oluştur"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        namespaces = {'ns': 'http://www.nih.gov'}
        annotation_dict = defaultdict(lambda: np.zeros((512, 512), dtype=np.uint8))

        for nodule in root.findall('.//ns:unblindedReadNodule', namespaces):
            for roi in nodule.findall('.//ns:roi', namespaces):
                sop_uid = roi.find('.//ns:imageSOP_UID', namespaces).text.strip()
                x_coords = [int(edge.find('.//ns:xCoord', namespaces).text) for edge in
                            roi.findall('.//ns:edgeMap', namespaces)]
                y_coords = [int(edge.find('.//ns:yCoord', namespaces).text) for edge in
                            roi.findall('.//ns:edgeMap', namespaces)]

                if len(x_coords) >= 3:
                    rr, cc = polygon(y_coords, x_coords, (512, 512))
                    annotation_dict[sop_uid][rr, cc] += 1

        return {sop_uid: (mask >= 1).astype(np.uint8) * 255 for sop_uid, mask in annotation_dict.items()}  # En az 1 etiket içerenler
    except Exception as e:
        print(f"XML hatası: {xml_path} - {str(e)}")
        return {}


def process_patient(patient_dir, split_type='train'):
    """Tek bir hastayı işle"""
    print(f"\n{split_type.upper()} işleniyor: {os.path.basename(patient_dir)}")

    xml_files = [f for f in os.listdir(patient_dir) if f.endswith('.xml')]
    dicom_files = [f for f in os.listdir(patient_dir) if f.endswith('.dcm')]

    for xml_file in xml_files:
        xml_path = os.path.join(patient_dir, xml_file)
        masks = parse_all_annotations(xml_path)

        for dicom_file in dicom_files:
            dicom_path = os.path.join(patient_dir, dicom_file)
            image, sop_uid = load_and_preprocess_dicom(dicom_path)

            if image is None:
                continue

            # Boşsa zeros, doluysa ilgili mask
            mask = masks.get(sop_uid, np.zeros_like(image, dtype=np.uint8))
            # Eğer maskede hiç segment yoksa (%100 sıfır):
            if np.count_nonzero(mask) == 0:
                # rand > 0.1 ise atla → %95 atlama, %5 kaydetme
                if random.random() > 0.05:
                    continue

            base_name = f"{os.path.basename(patient_dir)}_{dicom_file[:-4]}"
            cv2.imwrite(f'{split_type}/images/{base_name}.png', image)
            cv2.imwrite(f'{split_type}/masks/{base_name}_mask.png', mask)

            """
            if image is not None and sop_uid in masks:
                base_name = f"{os.path.basename(patient_dir)}_{dicom_file[:-4]}"
                cv2.imwrite(f'{split_type}/images/{base_name}.png', image)
                cv2.imwrite(f'{split_type}/masks/{base_name}_mask.png', masks[sop_uid])"""

if __name__ == "__main__":
    create_directories()
    base_dir = r"D:\LIDC\manifest-1600709154662\LIDC-IDRI"
    all_dirs = find_all_dicom_dirs(base_dir)

    # Hasta bazında bölme
    patient_ids = [os.path.basename(d).split('-')[-1] for d in all_dirs]
    gss = GroupShuffleSplit(test_size=0.1, random_state=42)
    train_idx, val_idx = next(gss.split(all_dirs, groups=patient_ids))

    train_dirs = [all_dirs[i] for i in train_idx]
    val_dirs = [all_dirs[i] for i in val_idx]

    print(f"Toplam hasta: {len(all_dirs)}")
    print(f"Train hastalar: {len(train_dirs)}")
    print(f"Validation hastalar: {len(val_dirs)}")

    # Paralel işleme
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Train hastalarını işle
        for patient_dir in train_dirs:
            executor.submit(process_patient, patient_dir, 'train')

        # Validation hastalarını işle
        for patient_dir in val_dirs:
            executor.submit(process_patient, patient_dir, 'val')
