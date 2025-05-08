import os
import pydicom
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from collections import defaultdict

dir = "LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/069.xml"
path_only = os.path.dirname(dir)
# XML'den tüm anotasyonları SOP_UID'ye göre gruplayan fonksiyon
def parse_xml_annotations(xml_path):
    namespaces = {'ns': 'http://www.nih.gov'}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations_dict = defaultdict(list)

    # Tüm nodülleri ve ROI'ları gez
    for nodule in root.findall('.//ns:unblindedReadNodule', namespaces):
        for roi in nodule.findall('.//ns:roi', namespaces):
            sop_uid = roi.find('.//ns:imageSOP_UID', namespaces).text.strip()
            points = []
            for edge in roi.findall('.//ns:edgeMap', namespaces):
                x = int(edge.find('.//ns:xCoord', namespaces).text)
                y = int(edge.find('.//ns:yCoord', namespaces).text)
                points.append((x, y))
            annotations_dict[sop_uid].append(points)

    return annotations_dict


# DICOM görüntüsünü ve ilgili anotasyonları göster
def show_dicom_with_annotations(dicom_path, annotations_dict):
    ds = pydicom.dcmread(dicom_path)
    sop_uid = ds.SOPInstanceUID

    plt.figure(figsize=(10, 10))
    plt.imshow(ds.pixel_array, cmap='gray')

    if sop_uid in annotations_dict:
        for polygon in annotations_dict[sop_uid]:
            x_coords = [p[0] for p in polygon]
            y_coords = [p[1] for p in polygon]
            # Çokgen çizimi
            plt.plot(x_coords, y_coords, linewidth=2, color='red')
            plt.scatter(x_coords, y_coords, s=30, c='yellow', marker='o', edgecolors='red')

    plt.title(f"DICOM: {os.path.basename(dicom_path)}\nSOP_UID: {sop_uid}")
    plt.axis('off')
    plt.show()


# Ana işlem
if __name__ == "__main__":
    xml_file = f"{dir}"  # XML dosyanızın adı
    dicom_dir = f"{path_only}"  # DICOM dosyalarınızın dizini

    # Anotasyonları SOP UID'ye göre grupla
    annotations = parse_xml_annotations(xml_file)

    # Dizin içindeki tüm DICOM'ları işle
    for filename in os.listdir(dicom_dir):
        if filename.endswith(".dcm"):
            dicom_path = os.path.join(dicom_dir, filename)
            try:
                show_dicom_with_annotations(dicom_path, annotations)
            except Exception as e:
                print(f"Hata: {filename} - {str(e)}")