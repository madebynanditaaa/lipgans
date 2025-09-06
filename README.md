# 🎥 LipGANs: Text-to-Viseme GAN Framework

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-red?logo=keras)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Dataset: TCD--TIMIT](https://img.shields.io/badge/Dataset-TCD--TIMIT-lightgrey)

LipGANs is a **text-to-lip animation framework** that generates short video clips of **mouth movements directly from text**, without requiring any audio input.

This project bridges **natural language processing (text → phonemes)** and **computer vision (GAN-based video synthesis)** to create realistic lip articulations from scratch.

---

## 🚀 Features

- **Audio-free lip generation** → Converts raw text directly into viseme-based animations.  
- **Phoneme-to-Viseme Mapping** → Maps linguistic units to 10 distinct mouth shapes.  
- **Per-Viseme GAN Training** → A separate 3D Convolutional GAN is trained for each viseme class.  
- **Dataset Preprocessing** → Automatic segmentation, lip region extraction, and normalization.  
- **Smooth Video Synthesis** → Concatenates generated viseme clips with temporal blending.  
- **Built on TCD-TIMIT dataset** → Aligned audiovisual dataset for speech-driven lip synthesis.  

---

## 📂 Repository Structure

```bash
LipGANs/
│── data/                   # Preprocessed dataset (organized by viseme)
│   ├── raw/                # Original TCD-TIMIT dataset (not included)
│   ├── viseme_01_Closed_Lips/
│   ├── viseme_02_Teeth_Touching/
│   └── ...
│
│── models/                 # Saved GAN models per viseme
│   ├── viseme_01/
│   └── ...
│
│── results/                # Generated outputs & evaluation samples
│
│── src/                    # Core source code
│   ├── preprocessing/      # Dataset preprocessing scripts
│   │   ├── phoneme_segmentation.py
│   │   ├── roi_extraction.py
│   │   └── viseme_mapping.json
│   │
│   ├── training/           # GAN training code
│   │   ├── gan_model.py
│   │   ├── train.py
│   │   └── utils.py
│   │
│   ├── inference/          # Text-to-animation pipeline
│   │   ├── text_to_viseme.py
│   │   ├── generate.py
│   │   └── smoothing.py
│
│── requirements.txt        # Python dependencies
│── README.md               # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/lipgans.git
cd lipgans
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**Dependencies include:**
- TensorFlow / Keras  
- NumPy, OpenCV, Imageio  
- MediaPipe (for lip landmark detection)  
- ffmpeg (for slicing & assembling clips)  

---

## 📊 Dataset Setup (TCD-TIMIT)

1. **Download TCD-TIMIT** dataset manually:  
   [TCD-TIMIT Dataset](https://sigmedia.tcd.ie/tcd_timit_db)  

2. Place it under:  
   ```bash
   data/raw/
   ```

3. Run preprocessing scripts:  
   ```bash
   python src/preprocessing/phoneme_segmentation.py
   python src/preprocessing/roi_extraction.py
   ```

This will:
- Segment videos into **phoneme-aligned clips**.  
- Extract **mouth regions** using MediaPipe FaceMesh.  
- Map **phonemes → visemes (10 classes)**.  
- Save **normalized 3-frame 64×64 sequences** into `data/viseme_xx/`.  

---

## 🗣 What are Visemes?

A **viseme** is any of several speech sounds that **look the same on the lips**, for example when lip reading.  
Unlike **phonemes** (the smallest units of sound in language), **visemes represent groups of phonemes that appear visually identical** on the face when spoken.

👉 Example:
- The phonemes `/p/`, `/b/`, and `/m/` all map to the same viseme (closed lips).

This is why phoneme-to-viseme mapping is essential for lip animation:
- It reduces complexity.  
- It ensures natural-looking articulation.  

📌 Example mapping (simplified):

| Viseme Class        | Example Phonemes | Lip Shape Description       |
|---------------------|------------------|-----------------------------|
| Closed Lips         | /p/, /b/, /m/    | Lips fully closed           |
| Teeth Touching      | /t/, /d/         | Tongue touches teeth        |
| Open Mouth (wide)   | /a/, /aa/        | Jaw dropped, lips open wide |
| Rounded Lips        | /oo/, /uw/, /w/  | Lips rounded forward        |

---

## 🏋️ Training

Train a GAN for a specific viseme class:  

```bash
python src/training/train.py --viseme_id 03 --epochs 200
```

- `--viseme_id`: Viseme class (01–10).  
- `--epochs`: Number of training epochs (default = 200).  

Trained models will be stored in:  
```
models/viseme_xx/
```

---

## 🎬 Inference (Text → Animation)

> ⚠️ **Note:** The raw output of LipGANs is a **sequence of generated frames** (per viseme) for maximum clarity. These frames can then be concatenated into an animation video (MP4/GIF) if needed.

Generate a lip animation for any input text:  

```bash
python src/inference/generate.py --text "Hello world"
```

**Steps performed:**  
1. **Text → Phonemes** (using CMU Pronouncing Dictionary).  
2. **Phonemes → Visemes** (via `viseme_mapping.json`).  
3. **GAN Generation**: Loads each viseme GAN and generates 3-frame clips.  
4. **Chaining & Smoothing**: Concatenates clips with temporal blending.  

Output saved in:  
```
results/hello_world.mp4
```

📌 Example:  
```bash
python src/inference/generate.py --text "Good morning"
```

Output → `results/good_morning.mp4`  

---

## 📈 Results

| Approach | Output Quality |
|----------|----------------|
| Single Multi-Class GAN | Blurry, frequent mode collapse |
| Per-Viseme GANs (ours) | Sharper details, stable articulation |

✅ Generated clips show **accurate viseme realization** and **plausible articulation** across unseen speakers.  

---

## 🌍 Applications

- 🎭 **Virtual Avatars & Chatbots** → Realistic mouth articulation in animated characters.  
- 🗣 **Speech Therapy Tools** → Helping learners visualize correct articulation.  
- 🦻 **Assistive Technology for the Deaf/Hard of Hearing** →  
  Deaf children (or learners with hearing difficulties) can simply **type a word/sentence into the UI** and see a **sequence of lip movements (frames or animation)** showing how it would be spoken. This bridges the gap between written text and spoken articulation.  
- 🎮 **Gaming & AR/VR** → Lifelike lip-syncing for immersive experiences. Can be used by animated characters
- 🎬 **Audio Dubbing & Localization** → Generate realistic lip movements that match translated text for films, shows, and animations.

---

## 🔮 Roadmap

- 🔹 **Speaker-conditioned GANs** (identity preservation).  
- 🔹 **Variable-length viseme clips** for realistic timing.  
- 🔹 **Quantitative evaluation** using FVD, lip-reading accuracy.  
- 🔹 **Multilingual support** (phoneme mappings for other languages).  
- 🔹 **Real-time integration** for virtual avatars and chatbots.
- 🔹 **Integration with dubbing & localization pipelines** for film and media industries.  

---

## 🤝 Contributing

Contributions are welcome!  
- Fork the repo  
- Create a new branch (`feature-xyz`)  
- Commit your changes  
- Open a Pull Request 🚀  

---

## 📜 License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.  

---

## 🔗 Citation

If you use this project in your research, please cite:  

```bibtex
@misc{lipgans2025,
  author = {Nandita Singh},
  title = {LipGANs: Text-to-Viseme GAN Framework for Audio-Free Lip Animation Generation},
  year = {2025},
  url = {https://github.com/madebynanditaaa/lipgans}
}
```

---

✨ With **LipGANs**, we take the first step towards **speech-free, text-driven lip animation** for next-gen human–computer interaction and accessibility!  
