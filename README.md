#  LipGANS: Text-to-Viseme GAN Framework

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-red?logo=keras)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Dataset: TCD--TIMIT](https://img.shields.io/badge/Dataset-TCD--TIMIT-lightgrey)

**LipGANS** is a **one-of-its-kind text-to-lip animation framework** that generates short video clips of **mouth movements directly from text**, without requiring any audio input.  

Unlike audio-driven lip-sync systems — where phoneme durations are known from speech timing — **LipGANs must predict phoneme durations from text alone**, making the task significantly more challenging and unique.  

It combines **natural language processing** (text → phonemes) and **computer vision** (GAN-based video synthesis) to create realistic lip articulations from scratch.  

---

### 🔄 Pipeline

**Text → Phonemes → Predicted Durations → Visemes → GANs → Frames → Video**

---

## 🚀 Features

- **Audio-free lip generation** → Converts raw text directly into viseme-based animations.  
- **Phoneme-to-Viseme Mapping** → Maps linguistic units to 10 distinct mouth shapes.  
- **Per-Viseme GAN Training** → A separate 3D Convolutional GAN is trained for each viseme class.
- Automatic Dataset Preprocessing → Segmentation, lip ROI extraction, normalization. 
- **Built on TCD-TIMIT dataset** → Aligned audiovisual dataset for speech-driven lip synthesis.  

---

## 📂 Repository Structure

```bash
lipgans/
├─ README.md                # Project documentation
├─ requirements.txt         # Python dependencies
├─ .gitignore               # Git ignore rules
├─ config/
│   └─ paths.example.yaml   # Example YAML for setting dataset and model paths
├─ src/
│   └─ lipgans/
│       ├─ __init__.py
│       ├─ config.py            # Config options: paths, latent dims, FPS, frame size
│       ├─ phonemes.py          # Functions to convert word → phonemes → visemes
│       ├─ data/                # Dataset preprocessing utilities
│       │   ├─ mlf_parser.py         # Parses TCD-TIMIT phoneme MLF files
│       │   ├─ extract_viseme_clips.py # Segments video/audio into per-viseme clips
│       │   ├─ crop_mouth.py         # Crops mouth ROI from frames
│       │   └─ dataset.py            # Dataset helper: load & organize clips for GAN training
│       ├─ models/
│       │   └─ gan3d.py             # 3D convolutional GAN architecture per viseme
│       ├─ train/
│       │   └─ train_viseme.py      # Script to train a single viseme GAN
│       ├─ generate/
│       │   ├─ merge_gans.py        # Load per-viseme GANs, generate frames, save PNG/GIF/MP4
│       │   └─ frontend.py          # Optional GUI / interface to generate words interactively
│       └─ utils/
│           ├─ io.py                # File I/O helpers
│           ├─ video.py             # Video assembling & frame handling helpers
│           └─ seed.py              # Random seed initialization for reproducibility
├─ scripts/                     # High-level scripts for batch processing or experiments
│   ├─ extract_all.py           # Slice all videos into per-viseme clips
│   ├─ crop_all.py              # Crop mouth regions for all dataset videos
│   ├─ train_all.py             # Train GANs for all viseme classes
│   ├─ generate_word.py         # Generate lip animation for a single word
│   └─ preview_crops.py         # Quick preview of cropped mouth ROIs
└─ examples/                     # Example outputs
    └─ demo_words.txt            # List of example words for demo generation

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
- NLTK (for CMU Pronouncing Dictionary)
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
   python src/lipgans/data/extract_viseme_clips.py
   python src/lipgans/data/crop_mouth.py
   
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

The output is a sequence of generated frames (PNG), which can also be saved as GIF or MP4.

```bash
python src/lipgans/generate/generate_word.py
```

**Steps performed:**  
1. **Text → Phonemes** (using CMU Pronouncing Dictionary).  
2. **Phonemes → Visemes** (via `viseme_mapping.json`).  
3. **GAN Generation**: Loads each viseme GAN and generates 3-frame clips.  
4. **Chaining & Smoothing**: Concatenates clips with temporal blending.  

Output saved in:  
```
example/cat/
 ├─ cat_01.png
 ├─ cat_02.png
 ├─ cat_03.png
 ├─ ...
 ├─ cat.gif
 └─ cat.mp4

```
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