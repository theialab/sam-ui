# SAM-UI: Image labelling with SAM-2 made easy!


<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <p>
      If you use this tool, kindly cite the paper 
      <a href="https://github.com/theialab/3dgs-flats" target="_blank">
        3D Gaussian Flats: Hybrid 2D/3D Photometric Scene Reconstruction
      </a>
    </p>
    <pre>
<code>
@misc{3dgsflats,
      title={3D Gaussian Flats: Hybrid 2D/3D Photometric Scene Reconstruction}, 
      author={Maria Taktasheva and Lily Goli and Alessandro Fiorini and Zhen and Li and Daniel Rebain and Andrea Tagliasacchi},
      year={2025},
      eprint={2509.16423},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.16423},
}</code>
    </pre>
  </div>
</section>

## Usage

### Environment creation
```
conda create -n sam2 python=3.12
conda activate sam2
pip install opencv-python matplotlib scipy
pip install 'git+https://github.com/facebookresearch/sam2.git'
pip install huggingface-hub

cd sam-ui
pip install -e .
```

### Tracking GUI
```
python scripts/tracking_gui.py --frames-path %path/to/frames% 
```

Availiable arguments
- `--output-path` -- Path to the output directory
- `--ui-scale` -- Scale factor for the UI (in case the window is too big or small)
- `--clear-output` -- If provided, the script will clear the output directory before starting"

### GUI usage

- Arrow Down ⬇️ → Select previous object
- Arrow Up ⬆️ → Select next object
- Arrow Left ⬅️ → Go to previous frame
- Arrow Right ➡️ → Go to next frame
- P → Propagate changes to next frames
- R → Reset state completely
- S → Save all progress
- C → Clear output
- `0`-`9` → Select object No. 0-9

### GUI usage video

![Usage demo](assets/demo.gif)

- Note that some of the propagation results are inconsistent with actual objects that are prompted, and the error accumulates because of wrong predictions, which is a limitation of SAMv2
- We currently set up the propagation for 16 frames forward, you might experiment with changing that number for the better results
