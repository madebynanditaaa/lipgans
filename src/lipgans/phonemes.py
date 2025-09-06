from nltk.corpus import cmudict

# Load CMU dict once
CMU = cmudict.dict()

# Mapping: phonemes → viseme class
PHONEME_TO_VISEME = {
    # Closed lips
    "b": "01_Closed_Lips", "p": "01_Closed_Lips", "m": "01_Closed_Lips",
    
    # Teeth touching
    "f": "02_Teeth_Touching", "v": "02_Teeth_Touching", "th": "02_Teeth_Touching", "dh": "02_Teeth_Touching",
    
    # Open mouth vowels / related
    "aa": "03_Open_Mouth", "ah": "03_Open_Mouth", "ae": "03_Open_Mouth", "eh": "03_Open_Mouth",
    "ih": "03_Open_Mouth", "iy": "03_Open_Mouth", "er": "03_Open_Mouth", "ey": "03_Open_Mouth",
    "g": "03_Open_Mouth", "k": "03_Open_Mouth", "uh": "03_Open_Mouth", "uw": "03_Open_Mouth", "hh": "03_Open_Mouth",
    
    # Rounded lips
    "aw": "04_Rounded_Lips", "ow": "04_Rounded_Lips", "oy": "04_Rounded_Lips", "w": "04_Rounded_Lips",
    
    # Tongue behind teeth
    "t": "05_Tongue_Behind_Teeth", "d": "05_Tongue_Behind_Teeth", "n": "05_Tongue_Behind_Teeth",
    "s": "05_Tongue_Behind_Teeth", "z": "05_Tongue_Behind_Teeth",
    
    # Retroflex / affricates
    "r": "06_Retroflex", "jh": "06_Retroflex",
    
    # Fricative sibilants
    "sh": "07_Fricative_Sibilant", "zh": "07_Fricative_Sibilant", "ch": "07_Fricative_Sibilant",
    
    # Nasal (back)
    "ng": "08_Nasal",
    
    # Lateral
    "l": "09_Lateral",
    
    # Semi vowels
    "y": "10_Semi_Vowel",
    
    # Complex / diphthongs
    "ay": "12_Complex_Sounds",
}

# List of viseme classes
VISEME_CLASSES = [
    "01_Closed_Lips", "02_Teeth_Touching", "03_Open_Mouth", "04_Rounded_Lips",
    "05_Tongue_Behind_Teeth", "06_Retroflex", "07_Fricative_Sibilant",
    "08_Nasal", "09_Lateral", "10_Semi_Vowel", "12_Complex_Sounds"
]


def word_to_phonemes(word: str) -> list[str]:
    """Convert a word to its CMU phoneme sequence (stress markers stripped)."""
    w = word.lower()
    if w not in CMU:
        raise ValueError(f"No phonemes found for word '{word}' in CMUdict.")
    # Strip stress markers like AH0 → ah
    return [p.lower().strip("0123456789") for p in CMU[w][0]]


def phonemes_to_visemes(phonemes: list[str]) -> list[str]:
    """Map a list of phonemes to corresponding viseme classes."""
    out = []
    for p in phonemes:
        v = PHONEME_TO_VISEME.get(p)
        if v and v != "00_Silence":
            out.append(v)
    return out

#Example usage:
# print(word_to_phonemes("hello"))
# # ['hh', 'ah', 'l', 'ow']

# print(phonemes_to_visemes(word_to_phonemes("hello")))
# # ['03_Open_Mouth', '03_Open_Mouth', '09_Lateral', '04_Rounded_Lips']
