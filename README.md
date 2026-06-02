# CattleScan Pro — Streamlit Web App

Automated cattle body measurement from side-view and rear-view images.

---

## Folder structure

```
cattle_app/
├── app.py                      ← main application
├── requirements.txt
├── Resnet101_Rcnn_Model.pth    ← side-view model (you supply)
├── rear_best_v2.pt             ← rear-view YOLO model (you supply)
├── sam2_b.pt                   ← SAM2 weights (you supply)
└── rf_score_model.pkl          ← RF score model (you supply)
```

---

## Installation

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your model files in this directory (see Folder structure above)

# 4. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## Publish on GitHub and Streamlit Community Cloud

The trained models are not committed to GitHub. The app downloads missing
models from the configured public Google Drive links when they are first
needed.

Upload these files to a new GitHub repository:

```text
.gitignore
app.py
model_files.py
packages.txt
requirements.txt
README.md
```

Do not upload the `.pt`, `.pth`, or `.pkl` model files.

To deploy:

1. Open Streamlit Community Cloud and create a new app from the GitHub repo.
2. Select `app.py` as the main file.
3. Open **Advanced settings** and select a supported Python version. Python
   3.12 is recommended and the pinned CPU wheels also support Python 3.14.
4. Deploy the app.

The first sticker click or estimate can take longer while the required models
download and load. Later actions reuse the downloaded files and Streamlit's
resource cache.

`packages.txt` installs only `libgl1`, which provides `libGL.so.1` for the
OpenCV dependency installed by Ultralytics.

The requirements use the official PyTorch CPU wheel index with the matched
pair `torch==2.9.0+cpu` and `torchvision==0.24.0+cpu`. Pip selects the wheel
for the deployed Python runtime without installing unused CUDA packages.

---

## Usage

| Step | Action |
|------|--------|
| 1 | Enter the cattle **Tag / ID** |
| 2 | Upload or capture the **side-view** image |
| 3 | Click on the **QR sticker** in the side image to calibrate scale |
| 4 | Upload or capture the **rear-view** image |
| 5 | Click on the **QR sticker** in the rear image |
| 6 | Click **Estimate Trait Measurements** |
| 7 | View key traits on the **Measure** page |
| 8 | View labelled images on the **Output** page |
| 9 | All records in **Logs** (appended, never overwritten) |

---

## Traits displayed on screen

| Trait | Source |
|-------|--------|
| Height (cm) | body_height — side view |
| Body Length (cm) | shoulder_bone → pin_bone — side view |
| Body Depth (cm) | body_top → body_bottom — side view |
| Heart Girth (cm) | regression from chest height — side view |
| Rear Leg Set Score (1–9) | `38.15 − 0.4384 × angle` — side view |
| Rear Leg Rear View Score (1–7) | Random Forest model — rear view |
| Rump Angle (°) | hip_bone → pin_bone slope — side view |
| Rump Width (cm) | rump_left → rump_right — rear view |

---

## Equations used

| Measurement | Formula |
|-------------|---------|
| Heart girth | `1.588 × chest_height_cm + 73.43` |
| Body weight | `(heart_girth² × body_length) / 10840` |
| Body depth score | `25.2 + 0.34 × body_depth_cm` |
| Rear leg set score | `38.15 − 0.4384 × hock_pastern_angle_deg` → clipped 1–9 |
| Rear leg rear score | RF model with features: hock_dist, left_angle, right_angle, mean_angle, angle_diff → clipped 1–7 |

---

## Angle definitions

**Side view — rear leg set angle:**  
Vector: hock → pastern.  
Angle with the horizontal line through the pastern point.  
Acute angle taken (min of angle and 180°−angle).

**Rear view — hoof ground angle:**  
Vector: hock → hoof.  
Angle with the horizontal line through the hoof point.  
Acute angle taken for both left and right legs.

---

## Notes

- If `streamlit-image-coordinates` is not installed, a manual cm/px entry fallback is shown.
- If SAM2 is unavailable, a simple fallback scale estimate is used.
- Logs are **appended** to `cattle_logs.json` and `cattle_logs.xlsx` — never overwritten.
- Output annotated images are saved to `cattle_output/` and **overwrite** on each new run.
- All log measurements are stored in **cm**, not pixels.
