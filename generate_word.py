from src.lipgans.generate.merge_gans import (
    get_phonemes, phoneme_to_mouth_shape, load_gan_model,
    generate_lip_frame, save_frame, predict_durations,
    create_gif, create_mp4
)


import os

word = input("Enter a word: ").strip()
base_model_dir = r"C:\Users\nandita\lipgans\saved_gans"
save_root = "example"

phonemes = get_phonemes(word)
durations = predict_durations(phonemes)

save_dir = os.path.join(save_root, word)
os.makedirs(save_dir, exist_ok=True)

frame_idx = 1
for phoneme, duration in zip(phonemes, durations):
    viseme_class = phoneme_to_mouth_shape.get(phoneme)
    if not viseme_class:
        print(f"⚠️ Unknown phoneme '{phoneme}', skipping...")
        continue

    model_path = os.path.join(base_model_dir, viseme_class)
    generator = load_gan_model(model_path)
    if generator is None:
        print(f"⚠️ GAN model for '{viseme_class}' not found, skipping...")
        continue

    num_frames = max(1, int(duration * 25))
    for _ in range(num_frames):
        frame = generate_lip_frame(generator)
        save_name = f"{word}_{frame_idx:02d}.png"
        save_frame(frame, os.path.join(save_dir, save_name))
        frame_idx += 1

create_gif(save_dir, os.path.join(save_root, f"{word}.gif"), duration=40)
create_mp4(save_dir, os.path.join(save_root, f"{word}.mp4"), fps=25)

print(f"✅ Lip animation generated in {save_dir}")
