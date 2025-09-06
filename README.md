<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LipGANs — Text-to-Viseme GAN Framework</title>
  <style>
    :root{--bg:#0f1724;--card:#0b1220;--muted:#94a3b8;--accent:#60a5fa;--glass:rgba(255,255,255,0.03)}
    body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,"Helvetica Neue",Arial;line-height:1.6;color:#e6eef8;background:linear-gradient(180deg,#071127 0,#071827 100%);padding:40px}
    .container{max-width:980px;margin:0 auto;background:linear-gradient(180deg,rgba(255,255,255,0.02),transparent);border-radius:12px;padding:28px;box-shadow:0 8px 30px rgba(2,6,23,0.7)}
    header{display:flex;align-items:center;gap:18px}
    h1{margin:0;font-size:28px}
    .badge{background:var(--glass);padding:6px 10px;border-radius:999px;font-size:13px;color:var(--accent);border:1px solid rgba(96,165,250,0.12)}
    .meta{color:var(--muted);font-size:14px;margin-top:6px}
    section{margin-top:20px;padding-top:14px;border-top:1px dashed rgba(255,255,255,0.02)}
    pre,code{background:#071827;padding:10px;border-radius:8px;color:#dbeafe;overflow:auto}
    .grid{display:grid;grid-template-columns:1fr 260px;gap:20px}
    .card{background:var(--card);padding:16px;border-radius:10px;border:1px solid rgba(255,255,255,0.02)}
    ul{margin:8px 0 0 18px}
    table{width:100%;border-collapse:collapse;margin-top:8px}
    th,td{padding:10px;border-bottom:1px solid rgba(255,255,255,0.02);text-align:left}
    .cta{display:inline-block;padding:10px 14px;border-radius:8px;background:linear-gradient(90deg,var(--accent),#38bdf8);color:#07203a;text-decoration:none;font-weight:600}
    footer{color:var(--muted);font-size:13px;margin-top:18px;text-align:center}
    @media (max-width:900px){.grid{grid-template-columns:1fr}}
    .note{background:rgba(96,165,250,0.06);padding:10px;border-radius:8px;border:1px solid rgba(96,165,250,0.08);color:var(--muted)}
  </style>
</head>
<body>
  <div class="container" role="main">
    <header>
      <div>
        <h1>LipGANs: Text-to-Viseme GAN Framework</h1>
        <div class="meta">Audio-free lip animation from text • Per-viseme 3D‑Conv GANs • Built on TCD‑TIMIT</div>
      </div>
      <div style="margin-left:auto;text-align:right">
        <div class="badge">MIT License</div>
        <div style="font-size:12px;color:var(--muted);margin-top:6px">2025 • Your Name</div>
      </div>
    </header>

    <section>
      <h2>Overview</h2>
      <p>LipGANs converts input <strong>text</strong> into short visual sequences of mouth movements (visemes) without using any audio. The pipeline uses a phoneme-to-viseme mapping and trains an independent 3D convolutional GAN for each viseme class to produce sharper, more stable results than a single, multi-class model.</p>
    </section>

    <section class="grid">
      <div>
        <h2>Features</h2>
        <ul>
          <li>Audio-free text → lip animation.</li>
          <li>Phoneme → Viseme mapping (10 classes).</li>
          <li>Separate 3D-Conv GAN per viseme for higher quality.</li>
          <li>Preprocessing with MediaPipe FaceMesh for robust ROI extraction.</li>
          <li>Temporal smoothing & blending for coherent output.</li>
        </ul>

        <h3>Quick Example</h3>
        <pre><code>python src/inference/generate.py --text "Good morning"
# Output: results/good_morning.mp4
</code></pre>

        <h3>Why per-viseme GANs?</h3>
        <p class="note">Training a dedicated GAN for each viseme reduces inter-class competition and mode collapse seen in multi-class models — resulting in sharper mouth shapes and better articulation fidelity.</p>

        <h2 style="margin-top:18px">Applications</h2>
        <ul>
          <li>Virtual avatars & chatbots</li>
          <li>Speech therapy & pronunciation visualization</li>
          <li>Language learning</li>
          <li><strong>Assistive tech for deaf / hard-of-hearing users:</strong> Users can type words/sentences into a UI and see a frame sequence or animation demonstrating articulation — a direct visual bridge between written and spoken language.</li>
          <li>Gaming, AR/VR lip-syncing</li>
        </ul>
      </div>

      <aside class="card">
        <h3>Repository Structure</h3>
        <pre><code>LipGANs/
├─ data/
├─ models/
├─ results/
└─ src/
   ├─ preprocessing/
   ├─ training/
   └─ inference/
</code></pre>

        <h3>Results Summary</h3>
        <table>
          <thead><tr><th>Approach</th><th>Quality</th></tr></thead>
          <tbody>
            <tr><td>Single Multi-Class GAN</td><td>Blurry / mode collapse</td></tr>
            <tr><td>Per-Viseme GANs (ours)</td><td>Sharper, stable articulation</td></tr>
          </tbody>
        </table>

        <h3 style="margin-top:10px">Tech Stack</h3>
        <ul>
          <li>TensorFlow / Keras</li>
          <li>NumPy, OpenCV, Imageio</li>
          <li>MediaPipe</li>
          <li>ffmpeg</li>
        </ul>
      </aside>
    </section>

    <section>
      <h2>Installation</h2>
      <pre><code>git clone https://github.com/your-username/lipgans.git
cd lipgans
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
</code></pre>
    </section>

    <section>
      <h2>Dataset Setup (TCD‑TIMIT)</h2>
      <ol>
        <li>Download TCD‑TIMIT manually and place under <code>data/raw/</code>.</li>
        <li>Run preprocessing:
          <pre><code>python src/preprocessing/phoneme_segmentation.py
python src/preprocessing/roi_extraction.py
</code></pre>
        </li>
        <li>Preprocessing produces phoneme-aligned 3-frame 64×64 sequences saved to <code>data/viseme_xx/</code>.</li>
      </ol>
      <p class="note">The TCD‑TIMIT dataset is not redistributable; users must obtain it directly from its provider and comply with its license.</p>
    </section>

    <section>
      <h2>Training</h2>
      <p>Train a GAN for a single viseme class:</p>
      <pre><code>python src/training/train.py --viseme_id 03 --epochs 200
</code></pre>
      <p>Trained models are saved under <code>models/viseme_xx/</code>.</p>
    </section>

    <section>
      <h2>Inference</h2>
      <p>Generate animation from text:</p>
      <pre><code>python src/inference/generate.py --text "Hello world"
# Output: results/hello_world.mp4
</code></pre>
      <p><strong>Pipeline:</strong> Text → CMU phonemes → Viseme mapping → per‑viseme GAN generation → chaining & smoothing → saved video.</p>
    </section>

    <section>
      <h2>Evaluation & Metrics</h2>
      <p>Planned / recommended metrics:</p>
      <ul>
        <li>Frechet Video Distance (FVD)</li>
        <li>Lip-reading accuracy (WER/CER on generated videos)</li>
        <li>User studies for perceptual quality</li>
      </ul>
    </section>

    <section>
      <h2>Roadmap</h2>
      <ul>
        <li>Speaker-conditioned GANs for identity preservation</li>
        <li>Variable-length viseme clips for timing realism</li>
        <li>Quantitative evaluation & benchmarks</li>
        <li>Multilingual phoneme-to-viseme mappings</li>
        <li>Real-time UI integration</li>
      </ul>
    </section>

    <section>
      <h2>Contributing</h2>
      <p>Contributions welcome — fork, branch, commit, and open a pull request. Please open an issue first for larger features.</p>
    </section>

    <section>
      <h2>License</h2>
      <p>MIT License — see the <code>LICENSE</code> file for details.</p>
    </section>

    <section>
      <h2>Citation</h2>
      <pre><code>@misc{lipgans2025,
  author = {Your Name},
  title = {LipGANs: Text-to-Viseme GAN Framework for Audio-Free Lip Animation},
  year = {2025},
  url = {https://github.com/your-username/lipgans}
}
</code></pre>
    </section>

  </div>
</body>
</html>
