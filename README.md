<div align="center">

# Photo Recognizer · Image Caption + Wikipedia

**BLIP-based image captioning with Wikipedia grounding - point it at a URL or a JPG, get a caption plus relevant encyclopedic context.**

<br/>

<img src="https://img.shields.io/badge/Model-BLIP-0D1117?style=for-the-badge&labelColor=161B22&color=FFA657" />
<img src="https://img.shields.io/badge/UI-Gradio-0D1117?style=for-the-badge&labelColor=161B22&color=58A6FF" />
<img src="https://img.shields.io/badge/Language-Python-0D1117?style=for-the-badge&logo=python&logoColor=FFA657&labelColor=161B22" />

</div>

---

## TL;DR

A small computer-vision sandbox exploring **self-supervised image captioning** (BLIP) with a Wikipedia enrichment step - the caption becomes a query, and the top matching article snippet is appended for context.

Originally built to explore how well captioning models transfer to real, noisy photos without fine-tuning.

---

## Pipeline

```mermaid
flowchart LR
    I[Image<br/>URL or file] --> B[BLIP caption<br/>generator]
    B --> C[Raw caption]
    C --> W[Wikipedia<br/>query]
    W --> CX[Article snippet]
    C --> OUT[Caption + context]
    CX --> OUT
    OUT --> G[Gradio UI]
```

---

## Files

| File | Purpose |
|------|---------|
| `image_cap.py` | Core caption generator (BLIP) |
| `image_captioning_app.py` | Gradio app wrapping the captioner |
| `automate_url_captioner.py` | Batch-caption a list of URLs |
| `glob.py` / `glob_` | Folder-walk helper for local images |
| `captions.txt` | Sample output |
| `BLIP-image recognizer based on self supervised.pdf` | Background / write-up |
| `gradio-deployML.pdf` | Deployment notes |
| `hello.py` | Sanity-check stub |

---

## Quick start

```bash
pip install transformers torch gradio wikipedia-api pillow requests
python image_captioning_app.py
```

Then open the local Gradio URL and upload an image or paste a URL.

---

<div align="center">
<sub>Part of <a href="https://github.com/pbathuri">@pbathuri</a>'s <a href="https://github.com/pbathuri/Map_Projects_MAC">project portfolio</a> - computer vision sandbox.</sub>
</div>
