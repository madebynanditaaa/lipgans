import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import nltk
from nltk.corpus import cmudict

# Download CMUdict if needed
nltk.download('cmudict')
cmu_dict = cmudict.dict()

# Phoneme to viseme mapping
phoneme_to_mouth_shape = {
    "b": "01_Closed_Lips", "p": "01_Closed_Lips", "m": "01_Closed_Lips",
    "f": "02_Teeth_Touching", "v": "02_Teeth_Touching", "th": "02_Teeth_Touching", "dh": "02_Teeth_Touching",
    "aa": "03_Open_Mouth", "ae": "03_Open_Mouth", "ah": "03_Open_Mouth", "eh": "03_Open_Mouth",
    "ih": "03_Open_Mouth", "iy": "03_Open_Mouth", "er": "03_Open_Mouth", "ey": "03_Open_Mouth",
    "g": "03_Open_Mouth", "k": "03_Open_Mouth", "uh": "03_Open_Mouth", "uw": "03_Open_Mouth", "hh": "03_Open_Mouth",
    "aw": "04_Rounded_Lips", "ow": "04_Rounded_Lips", "oy": "04_Rounded_Lips", "w": "04_Rounded_Lips",
    "t": "05_Tongue_Behind_Teeth", "d": "05_Tongue_Behind_Teeth", "n": "05_Tongue_Behind_Teeth",
    "s": "05_Tongue_Behind_Teeth", "z": "05_Tongue_Behind_Teeth",
    "r": "06_Retroflex", "jh": "06_Retroflex",
    "sh": "07_Fricative_Sibilant", "zh": "07_Fricative_Sibilant", "ch": "07_Fricative_Sibilant",
    "ng": "08_Nasal",
    "l": "09_Lateral",
    "y": "10_Semi_Vowel",
    "ay": "03_Open_Mouth"
}

# --- Core helper functions ---

def get_phonemes(word):
    """Return phonemes for a word using CMUdict."""
    word = word.lower()
    if word not in cmu_dict:
        raise ValueError(f"No phonemes found for '{word}'")
    phonemes = cmu_dict[word][0]
    return [p.lower().strip("0123456789") for p in phonemes]

def load_gan_model(viseme_class_path, epoch=100):
    """Load generator model if exists, else return None."""
    model_path = os.path.join(viseme_class_path, f"generator_epoch_{epoch}.model.keras")
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

def generate_lip_frame(generator, latent_dim=100):
    """Generate one lip frame from a GAN."""
    z = np.random.normal(0, 1, (1, latent_dim))
    gen = generator.predict(z)
    if gen.ndim == 5:
        return gen[0, 0]
    elif gen.ndim == 4:
        return gen[0]
    else:
        return gen

def save_frame(image_array, save_path):
    """Save a numpy image array as PNG."""
    image_array = np.clip(image_array, 0, 1)
    image_array = (image_array * 255).astype(np.uint8)
    Image.fromarray(image_array).save(save_path)

def predict_durations(phonemes, base_duration=0.1):
    """Assign simple durations: longer for vowels."""
    vowels = ['aa','ae','ah','eh','ih','iy','er','ey','uh','uw','aw','ow','oy','ay']
    return [base_duration*1.5 if p in vowels else base_duration for p in phonemes]

def create_gif(folder, output_path, duration=100):
    """Create a GIF from saved PNG frames."""
    from PIL import Image
    files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    frames = [Image.open(os.path.join(folder, f)) for f in files]
    if frames:
        frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)

def create_mp4(folder, output_path, fps=25):
    """Create an MP4 video from saved PNG frames."""
    files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    if not files: return
    first_frame = cv2.imread(os.path.join(folder, files[0]))
    h, w, _ = first_frame.shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in files:
        frame = cv2.imread(os.path.join(folder, f))
        video.write(frame)
    video.release()
