````markdown
# Lung Nodule Segmentation on LIDC-IDRI

An end-to-end PyTorch pipeline for 2D and 3D U-Net–based lung nodule segmentation, from raw DICOM to interactive inference UI.

---

## 📂 Branches

- **main**  
  2D U-Net code:  
  `dataset.py`, `lidc-dataset-ct-axial-split.py`, `model.py`, `train_new.py`, `utils.py`, `app.py`  
- **mmy**  
  3D U-Net scripts and volumetric preprocessing (e.g. `train_3d.py`, patch extraction, resampling)

---

## ⚙️ Installation

1. Clone and enter the repo  
   `git clone https://github.com/your-username/LungNoduleSegmentation.git && cd LungNoduleSegmentation`  
2. Create a virtual environment and install  
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
````

3. Download LIDC-IDRI DICOMs under `data/LIDC-IDRI/`

---

## 🚀 Usage

1. **2D Training (main branch)**

   ```bash
   python train_new.py \
     --data-dir data/LIDC-IDRI \
     --img-size 256 \
     --batch-size 8 \
     --epochs 100 \
     --loss tversky_focal \
     --norm groupnorm
   ```
2. **3D Training (mmy branch)**

   ```bash
   git checkout mmy
   python train_3d.py \
     --data-dir data/LIDC-IDRI \
     --patch-size 64 128 128 \
     --batch-size 2 \
     --epochs 30 \
     --loss tversky_focal
   ```
3. **Inference UI**
   From **main** branch:

   ```bash
   python app.py
   ```

   Open your browser at `http://localhost:7860`

---

## 📊 Results

| Model    | Dice | IoU  |
| -------- | ---- | ---- |
| 2D U-Net | 0.66 | 0.47 |
| 3D U-Net | 0.60 | 0.43 |

---

## 🗂 File Structure

```
main branch
├── .gitignore
├── dataset.py
├── lidc-dataset-ct-axial-split.py
├── model.py
├── train_new.py
├── utils.py
├── app.py
└── UNET_architecture.png

mmy branch
├── train_3d.py
├── patch_extraction.py
└── resample_and_crop.py
```

---

## 📸 Figures

* **Prediction Example**

![top_5](https://github.com/user-attachments/assets/b5b6e2a5-a7cf-421a-aba3-5a6948781c86)

* **Gradio interface**

![Ekran görüntüsü 2025-05-11 110758](https://github.com/user-attachments/assets/bd77cce3-1a89-43fb-9853-1f029a931723)


---

## 📜 License

This project is MIT-licensed. See [LICENSE](LICENSE) for details.

```
```
---

## 🙌 Contributors

Thanks to all these wonderful people:

- [@canomercik](https://github.com/canomercik) – Project author  
- [@mustafayngl](https://github.com/mustafayngl) – Data preprocessing & scripts  
- [@Sinestre](https://github.com/Sinestre) – 3D U-Net implementation  
