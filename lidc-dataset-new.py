import os
import pydicom
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from skimage.draw import polygon
from skimage.measure import label, regionprops
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
import random
import csv


def create_directories():
    """Train/Val klasörlerini oluştur"""
    for split in ['train', 'val']:
        os.makedirs(f'{split}/images', exist_ok=True)
        os.makedirs(f'{split}/masks', exist_ok=True)


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


# Sabit pencere aralığı
HU_MIN, HU_MAX = -1000.0, 400.0

def load_and_preprocess_dicom(dicom_path):
    """DICOM'ı yükle, sabit HU window uygula ve CLAHE ile kontrast dengele."""
    try:
        ds = pydicom.dcmread(dicom_path)
        if not is_ct_and_axial(ds):
            return None, None

        # HU değerine dönüştür
        hu = ds.pixel_array.astype(np.float32)
        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
            hu = hu * ds.RescaleSlope + ds.RescaleIntercept

        # Sabit window
        hu = np.clip(hu, HU_MIN, HU_MAX)
        img = ((hu - HU_MIN) / (HU_MAX - HU_MIN) * 255.0).astype(np.uint8)

        # CLAHE ile kontrast dengeleme (opsiyonel, gerekirse kaldırabilirsiniz)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)

        return img, ds.SOPInstanceUID
    except Exception as e:
        print(f"DICOM hatası: {dicom_path} – {e}")
        return None, None


def parse_all_annotations(xml_path):
    """XML'den maskeleri oluştur"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {'ns': 'http://www.nih.gov'}
        annots = defaultdict(lambda: np.zeros((512, 512), dtype=np.uint8))

        for nodule in root.findall('.//ns:unblindedReadNodule', ns):
            for roi in nodule.findall('.//ns:roi', ns):
                sop = roi.find('.//ns:imageSOP_UID', ns).text.strip()
                x_coords = [int(e.find('.//ns:xCoord', ns).text) for e in roi.findall('.//ns:edgeMap', ns)]
                y_coords = [int(e.find('.//ns:yCoord', ns).text) for e in roi.findall('.//ns:edgeMap', ns)]
                if len(x_coords) >= 3:
                    rr, cc = polygon(y_coords, x_coords, (512, 512))
                    annots[sop][rr, cc] += 1

        # Final mask: consensus >=4
        return {sop: ((m >= 4).astype(np.uint8) * 255) for sop, m in annots.items()}
    except Exception as e:
        print(f"XML hatası: {xml_path} - {str(e)}")
        return {}


def process_patient(patient_dir, split, meta_list, intensity_thresh=20, min_area=50):
    """Tek hasta için dönüştürme ve kaydetme"""
    xmls = [f for f in os.listdir(patient_dir) if f.endswith('.xml')]
    dcmds = [f for f in os.listdir(patient_dir) if f.endswith('.dcm')]

    for xml in xmls:
        masks = parse_all_annotations(os.path.join(patient_dir, xml))
        for dcm in dcmds:
            img, sop = load_and_preprocess_dicom(os.path.join(patient_dir, dcm))
            if img is None or sop not in masks:
                continue

            m = masks[sop]
            # Maske ve görüntü boyutunu eşitle
            if m.shape != img.shape:
                m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # %0 maskeleri %90 drop
            if np.count_nonzero(m) == 0 and random.random() > 0.1:
                continue

            if split == 'train':
                # Sadece yeterli büyüklükteki segmentler
                if np.count_nonzero(m) < min_area:
                    continue

                # Crop ROI
                bin_mask = (m > 0).astype(np.uint8)
                labeled = label(bin_mask)
                props = regionprops(labeled)
                if not props:
                    continue
                largest = max(props, key=lambda x: x.area)
                minr, minc, maxr, maxc = largest.bbox
                pad = 10
                minr, minc = max(minr - pad, 0), max(minc - pad, 0)
                maxr, maxc = min(maxr + pad, img.shape[0]), min(maxc + pad, img.shape[1])
                img_c = img[minr:maxr, minc:maxc]
                mask_c = ((labeled == largest.label).astype(np.uint8) * 255)[minr:maxr, minc:maxc]

                # Görüntü yoğunluğunu doğrula
                mask_pixels = img_c[mask_c > 0]
                if mask_pixels.size == 0 or mask_pixels.mean() < intensity_thresh:
                    continue

                bbox = (minr, minc, maxr, maxc)
            else:
                img_c, mask_c = img, m
                bbox = (0, 0, img.shape[0], img.shape[1])

            base = f"{os.path.basename(patient_dir)}_{dcm[:-4]}"
            img_path = f"{split}/images/{base}.png"
            msk_path = f"{split}/masks/{base}_mask.png"
            cv2.imwrite(img_path, img_c)
            cv2.imwrite(msk_path, mask_c)

            meta_list.append({
                'patient': os.path.basename(patient_dir),
                'sop_uid': sop,
                'split': split,
                'bbox': bbox,
                'mask_area': int(np.sum(mask_c > 0)),
                'img_path': img_path,
                'mask_path': msk_path
            })

if __name__ == '__main__':
    random.seed(42)
    create_directories()
    base_dir = r"D:\LIDC\manifest-1600709154662\LIDC-IDRI"
    all_dirs = find_all_dicom_dirs(base_dir)

    patient_ids = [os.path.basename(d).split('-')[-1] for d in all_dirs]
    gss = GroupShuffleSplit(test_size=0.1, random_state=42)
    train_idx, val_idx = next(gss.split(all_dirs, groups=patient_ids))
    train_dirs = [all_dirs[i] for i in train_idx]
    val_dirs = [all_dirs[i] for i in val_idx]

    meta = []
    with ThreadPoolExecutor(max_workers=4) as exe:
        for d in train_dirs:
            exe.submit(process_patient, d, 'train', meta)
        for d in val_dirs:
            exe.submit(process_patient, d, 'val', meta)

    keys = ['patient', 'sop_uid', 'split', 'bbox', 'mask_area', 'img_path', 'mask_path']
    with open('dataset_metadata.csv', 'w', newline='') as f:
        dw = csv.DictWriter(f, fieldnames=keys)
        dw.writeheader()
        dw.writerows(meta)

    print('İşlem tamamlandı.')
