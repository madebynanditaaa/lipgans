import gradio as gr
from pathlib import Path
from ..config import Config
from .merge_gans import generate_word

def build_app(default_cfg_path: str):

    def _go(word: str, cfg_path: str):
        try:
            cfg = Config.load(cfg_path)  # load config dynamically
            gif, mp4 = generate_word(word, cfg)
            return str(gif), str(mp4)
        except Exception as e:
            return f"Error: {e}", None

    with gr.Blocks() as demo:
        gr.Markdown("# LipGANs — type a word, get a lip GIF/MP4")
        with gr.Row():
            word_input = gr.Textbox(label="Word", value="cat")
            cfg_input = gr.Textbox(label="Config Path", value=default_cfg_path)
        btn = gr.Button("Generate")
        gif_out = gr.Image(label="GIF", type="filepath")
        mp4_out = gr.Video(label="MP4")
        
        btn.click(_go, inputs=[word_input, cfg_input], outputs=[gif_out, mp4_out])

    return demo

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True, help="Path to default config file")
    args = ap.parse_args()
    app = build_app(args.paths)
    app.launch()
