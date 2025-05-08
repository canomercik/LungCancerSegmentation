import os
import pydicom
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


dir = "LIDC-IDRI/LIDC-IDRI-0002/01-01-2000-NA-NA-98329/3000522.000000-NA-04919"


# XML'den anotasyonları çeken fonksiyon
def parse_xml_annotations(xml_path):
    namespaces = {'ns': 'http://www.nih.gov/idri'}  # XML namespace tanımı
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []
    # Tüm okuma oturumlarını bul
    for session in root.findall('.//ns:CXRreadingSession', namespaces):
        x = int(session.find('.//ns:xCoord', namespaces).text)
        y = int(session.find('.//ns:yCoord', namespaces).text)
        annotations.append((x, y))
    return annotations


# DICOM dosyasını ve anotasyonları gösteren fonksiyon
def show_dicom_with_annotations(dicom_path, annotations):
    ds = pydicom.dcmread(dicom_path)
    plt.imshow(ds.pixel_array, cmap='gray')

    # Anotasyonları işaretle
    if annotations:
        x_coords = [coord[0] for coord in annotations]
        y_coords = [coord[1] for coord in annotations]
        plt.scatter(x_coords, y_coords, c='red', s=40, marker='x', label='Nodül')
        plt.legend()

    plt.title(f"DICOM Görüntüsü: {os.path.basename(dicom_path)}")
    plt.axis('off')
    plt.show()


# Ana işlem
if __name__ == "__main__":
    xml_file = f"{dir}/071.xml"  # XML dosya adı
    dicom_dir = dir  # DICOM dosyalarının bulunduğu dizin

    # XML'den anotasyonları al
    annotations = parse_xml_annotations(xml_file)

    # Dizin içindeki tüm DICOM dosyalarını işle
    for filename in os.listdir(dicom_dir):
        if filename.endswith(".dcm"):
            dicom_path = os.path.join(dicom_dir, filename)
            try:
                show_dicom_with_annotations(dicom_path, annotations)
            except Exception as e:
                print(f"Hata: {filename} işlenemedi - {str(e)}")