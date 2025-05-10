import os
import numpy as np
import pydicom
import xml.etree.ElementTree as ET
from collections import defaultdict
from skimage.draw import polygon
from tqdm import tqdm
import random
import shutil

# === Ayarlar ===
base_dir = r"D:\Dataset\manifest-1600709154662\LIDC-IDRI"
temp_dir = "temp_npz"
train_dir = "data_3d/train"
val_dir = "data_3d/val"
val_ratio = 0.1
min_slices = 5
min_nonzero_mask = 100

# === Sayaçlar ===
valid_count = 0
empty_count = 0
small_count = 0
total_files = 0

# === Klasörleri oluştur ===
for d in [temp_dir, train_dir, val_dir]:
    os.makedirs(d, exist_ok=True)

def is_ct_axial(ds):
    return hasattr(ds, "Modality") and ds.Modality == "CT" and \
           hasattr(ds, "ImageOrientationPatient") and \
           np.allclose(ds.ImageOrientationPatient, [1, 0, 0, 0, 1, 0], atol=1e-1)

def load_dicom(dcm_path):
    try:
        ds = pydicom.dcmread(dcm_path)
        if not is_ct_axial(ds): return None, None, None
        image = ds.pixel_array.astype(np.float32)
        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
            image = image * ds.RescaleSlope + ds.RescaleIntercept
        image = np.clip(image, -200, 200)
        image = ((image + 200) / 400) * 255
        image = image.astype(np.uint8)
        return image, ds.SOPInstanceUID, ds.InstanceNumber
    except:
        return None, None, None


def parse_annotations(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {'ns': 'http://www.nih.gov'}
        annots = defaultdict(lambda: np.zeros((512, 512), dtype=np.uint8))

        for nodule in root.findall('.//ns:unblindedReadNodule', ns):
            for roi in nodule.findall('.//ns:roi', ns):
                sop_uid = roi.find('.//ns:imageSOP_UID', ns).text.strip()
                xs = [int(e.find('.//ns:xCoord', ns).text) for e in roi.findall('.//ns:edgeMap', ns)]
                ys = [int(e.find('.//ns:yCoord', ns).text) for e in roi.findall('.//ns:edgeMap', ns)]
                if len(xs) >= 3:
                    rr, cc = polygon(ys, xs, (512, 512))
                    annots[sop_uid][rr, cc] += 1

        # En az 1 radyoloğun işaretlediği yerleri al
        return {uid: (mask / 4.0).astype(np.float32) for uid, mask in annots.items()}

    except Exception as e:
        print(f"Annotation parse hatası: {e}")
        return {}

def process_patient(patient_path):
    global valid_count, empty_count, small_count, total_files

    xml_files = [f for f in os.listdir(patient_path) if f.endswith(".xml")]
    if not xml_files:
        return None
    xml_path = os.path.join(patient_path, xml_files[0])
    mask_dict = parse_annotations(xml_path)

    slices = []
    masks = []

    for f in os.listdir(patient_path):
        if not f.endswith(".dcm"): continue
        dcm_path = os.path.join(patient_path, f)
        image, uid, instance = load_dicom(dcm_path)
        if image is None or uid not in mask_dict:
            continue
        slices.append((instance, image))
        masks.append((instance, mask_dict[uid] * 255))

    if len(slices) < min_slices:
        return None

    slices.sort()
    masks.sort()
    volume = np.stack([s[1] for s in slices], axis=0)
    mask = np.stack([m[1] for m in masks], axis=0)

    total_files += 1
    nonzero = np.count_nonzero(mask)

    if nonzero == 0:
        empty_count += 1
        return None
    elif nonzero < min_nonzero_mask:
        small_count += 1
        return None

    filename = os.path.basename(patient_path).replace('.', '_') + ".npz"
    save_path = os.path.join(temp_dir, filename)
    np.savez_compressed(save_path, image=volume, mask=mask)
    valid_count += 1
    return filename

# === .dcm ve .xml içeren tüm klasörleri bul ===
all_dirs = []
for root, dirs, files in os.walk(base_dir):
    if any(f.endswith('.dcm') for f in files) and any(f.endswith('.xml') for f in files):
        all_dirs.append(root)

print(f"\n📂 Toplam işlenecek klasör: {len(all_dirs)}")

# === .npz üretimi ===
created_files = []
for p in tqdm(all_dirs):
    result = process_patient(p)
    if result:
        created_files.append(result)

# === Train / Val ayır ===
random.shuffle(created_files)
val_count = int(len(created_files) * val_ratio)
val_files = created_files[:val_count]
train_files = created_files[val_count:]

# === Taşı ve logla ===
for f in train_files:
    shutil.copy(os.path.join(temp_dir, f), os.path.join(train_dir, f))
for f in val_files:
    shutil.copy(os.path.join(temp_dir, f), os.path.join(val_dir, f))

with open("train_files.txt", "w") as f:
    f.writelines([file + "\n" for file in train_files])
with open("val_files.txt", "w") as f:
    f.writelines([file + "\n" for file in val_files])

# === Özet Yazdır ===
print(f"\n✅ {len(train_files)} train | {len(val_files)} val dosyası ayrıldı.")
print("📄 Loglar: train_files.txt ve val_files.txt")
print(f"📁 Train klasörü: {train_dir}")
print(f"📁 Val klasörü:   {val_dir}")
print(f"\n📊 Filtreleme Özeti:")
print(f"Toplam XML + DICOM hasta klasörü: {len(all_dirs)}")
print(f"İşlenen .npz adedi              : {total_files}")
print(f"Geçerli (maskeli) .npz          : {valid_count}")
print(f"Boş maske bulunanlar            : {empty_count}")
print(f"Çok küçük maskeler (<100 px)    : {small_count}")
