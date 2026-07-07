# UniDrive

UniDrive is a unified vision-language and grounding framework for
interpretable risk understanding in autonomous driving.

The model is designed to answer safety-critical driving questions with both a
natural-language explanation and explicit visual evidence. Given a short driving
clip, UniDrive identifies the risk-relevant object, explains why it matters for
the ego vehicle, recommends a cautious driving response, and appends the
predicted risk-object box as text:

```text
The pedestrian ahead is entering the ego lane, so the ego car should slow down and yield.
<box>x1, y1, x2, y2</box>
```

This repository is built on top of LLaVA and adds the UniDrive dual-branch
architecture, high-resolution perception path, spatio-temporal fusion module,
and training data loader support for paired temporal frames and high-resolution
final-frame inputs.

## Paper

**UniDrive: A Unified Vision-Language and Grounding Framework for
Interpretable Risk Understanding in Autonomous Driving**

Xiaowei Gao, Pengxiang Li, Yitai Cheng, Ruihan Xu, James Haworth, Stephen Law,
Yun Ye

The paper studies a central limitation of driving-oriented MLLMs: models with
temporal context often lose fine spatial detail, while high-resolution
single-frame models often lack dynamic risk reasoning. UniDrive addresses this
by combining temporal reasoning, high-resolution perception, and gated
cross-attention fusion in one generative framework.

## Highlights

- **Temporal reasoning branch:** encodes multiple low-resolution video frames to
  capture motion, interactions, and risk evolution.
- **High-resolution perception branch:** encodes the latest frame at higher
  spatial fidelity to preserve small, distant, or partially occluded hazards.
- **Spatio-temporal fusion:** uses temporal features as queries and
  high-resolution spatial features as keys and values in a gated cross-attention
  module.
- **Unified generation and grounding:** emits natural-language risk explanations
  and bounding boxes in the same autoregressive output sequence.
- **Safety-oriented evaluation:** reports captioning, risk-object grounding,
  object-scale IoU, zero-shot transfer, robustness, efficiency, qualitative
  cases, and human preference results.

## Repository Layout

```text
llava/
  model/llava_arch.py                 # UniDrive dual-tower and fusion modules
  model/builder.py                    # Loads optional HR vision tower
  train/train.py                      # SFT loader with image_hr support
scripts/
  v1_5/finetune_unidrive_7b.sh        # Default UniDrive SFT recipe
  zero2.json, zero3.json              # DeepSpeed configs
docs/                                 # Inherited LLaVA documentation
```

Key UniDrive implementation points:

- `MeanTemporalAggregator` pools frame-level visual tokens along the temporal
  axis.
- `GatedCrossAttentionFusion` aligns dynamic temporal tokens with
  high-resolution spatial tokens.
- `vision_tower_hr`, `hr_mm_projector`, and `st_fusion` are initialized when
  `--vision_tower_hr` is provided.
- The supervised dataset accepts `image` as either one frame or a list of
  temporal frames, and accepts `image_hr` as the high-resolution final-frame
  input.

## Installation

UniDrive follows the LLaVA training stack and is intended for Linux/CUDA
environments.

```bash
git clone https://github.com/pixeli99/unidrive-dev.git
cd unidrive-dev

conda create -n unidrive python=3.10 -y
conda activate unidrive

pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

If `flash-attn` is not available for your CUDA/PyTorch combination, install a
compatible wheel or remove flash attention from the local training setup.

## Data Format

The default training script expects the following layout:

```text
playground/data/
  unidrive_sft.json
  unidrive/
    frames_lr/
      ...
    frames_hr/
      ...
```

Each SFT sample follows the LLaVA conversation format, with two visual entries:

```json
{
  "id": "sample_000001",
  "image": [
    "clip_000001/frame_000.jpg",
    "clip_000001/frame_015.jpg",
    "clip_000001/frame_030.jpg",
    "clip_000001/frame_045.jpg",
    "clip_000001/frame_059.jpg"
  ],
  "image_hr": "clip_000001/frame_059.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhich object is at the highest risk?"
    },
    {
      "from": "gpt",
      "value": "A pedestrian ahead is entering the ego lane. The ego car should slow down and yield. <box>x1, y1, x2, y2</box>"
    }
  ]
}
```

Use a consistent coordinate convention for all boxes in training and evaluation.
The paper reports risk-object grounding on the final frame.

## Training

Run the UniDrive supervised fine-tuning recipe:

```bash
bash scripts/v1_5/finetune_unidrive_7b.sh
```

Core training flags:

```bash
--model_name_or_path meta-llama/Llama-2-7b-hf
--data_path ./playground/data/unidrive_sft.json
--image_folder ./playground/data/unidrive/frames_lr
--image_hr_folder ./playground/data/unidrive/frames_hr
--vision_tower openai/clip-vit-large-patch14-336
--vision_tower_hr openai/clip-vit-large-patch14-336
--mm_projector_type mlp2x_gelu
--mm_temporal_aggregator mean
--mm_st_fusion_num_heads 8
--output_dir ./checkpoints/unidrive-llama2-7b
```

The paper uses a single supervised fine-tuning stage on the curated
DRAMA-Reasoning training set, without RLHF. It uses Llama2-7B as the backbone,
five uniformly sampled frames per clip, always includes the final frame for
bounding-box prediction, and trains with AdamW plus cosine learning-rate
scheduling on 4 NVIDIA A100 80GB GPUs.

## Inference

After training or downloading a UniDrive checkpoint, load it through the standard
LLaVA model builder. A UniDrive checkpoint should contain the high-resolution
vision tower config and the fusion module weights.

```python
import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def load_rgb(path):
    return Image.open(path).convert("RGB")


disable_torch_init()

tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path="./checkpoints/unidrive-llama2-7b",
    model_base=None,
    model_name="unidrive-llama2-7b",
)

image_processor_hr = getattr(model, "image_processor_hr", image_processor)

frame_paths = [
    "frame_000.jpg",
    "frame_015.jpg",
    "frame_030.jpg",
    "frame_045.jpg",
    "frame_059.jpg",
]
frames = [load_rgb(path) for path in frame_paths]

images = torch.stack([
    image_processor.preprocess(frame, return_tensors="pt")["pixel_values"][0]
    for frame in frames
]).unsqueeze(0).half().cuda()

images_hr = image_processor_hr.preprocess(
    frames[-1], return_tensors="pt"
)["pixel_values"].half().cuda()

question = "Which object is at the highest risk? Explain why and provide the box."
conv = conv_templates["llava_llama_2"].copy()
conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(
    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
).unsqueeze(0).cuda()

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=images,
        images_hr=images_hr,
        do_sample=False,
        temperature=0,
        max_new_tokens=256,
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

Expected output is a risk explanation followed by a `<box>...</box>` span.

## Paper Results

### DRAMA-Reasoning Validation Split

The paper evaluates captioning with BLEU-4 (B4), METEOR, CIDEr, and SPICE, and
evaluates grounding with mean IoU. `AVG` is the arithmetic mean of B4 and mIoU.

| Input | Method | B4 | CIDEr | mIoU | mIoU_S | AVG |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Image | BLIP-2 | 46.1 | 194.7 | 46.3 | 8.1 | 46.2 |
| Image | LLaVA | 47.5 | 198.6 | 47.2 | 8.0 | 47.4 |
| Image | InstructBLIP | 49.9 | 205.0 | 47.8 | 9.1 | 48.9 |
| Image | Shikra* | 49.8 | 204.7 | 50.3 | 10.4 | 50.1 |
| Image | Ours w/o ST | 55.2 | 246.7 | 59.8 | 29.8 | 57.5 |
| Video | eP-ALM | 51.4 | 225.1 | 43.2 | 7.2 | 47.3 |
| Video | Video-LLAMA | 53.9 | 229.5 | 42.8 | 6.9 | 48.4 |
| Video | UniDrive | 60.3 | 277.5 | 61.2 | 31.0 | 60.8 |

### Zero-Shot Generalization

| Dataset | Task | Baseline | UniDrive |
| --- | --- | ---: | ---: |
| NuScenes | VQA average accuracy | 68.1 | 75.3 |
| BDD100K | Risk-object mAP@0.50 | 41.3 | 52.7 |

On NuScenes, UniDrive improves most strongly on situation reasoning
(63.6 vs. 54.5 for Video-LLAMA). On BDD100K, it improves pedestrian, cyclist,
and car AP in zero-shot risk localization.

### Ablation Summary

| Variant | B4 | CIDEr | mIoU | mIoU_S | AVG |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full UniDrive | 60.3 | 277.5 | 61.2 | 31.0 | 60.8 |
| w/o Temporal Reasoning Branch | 55.2 | 246.7 | 59.8 | 29.8 | 57.5 |
| w/o High-Res Perception Branch | 52.8 | 238.2 | 47.9 | 12.4 | 50.4 |
| w/o Spatio-Temporal Fusion | 52.4 | 222.8 | 44.6 | 9.7 | 48.5 |
| w/o Box Token Grounding | 60.1 | 276.1 | 0.0 | 0.0 | 30.1 |

These ablations support the paper's main claim: temporal reasoning improves
risk explanation, high-resolution perception improves spatial localization, and
explicit gated fusion is needed to align the two.

### Robustness and Efficiency

| Condition | UniDrive mIoU | UniDrive CIDEr | Video-LLAMA mIoU | Video-LLAMA CIDEr |
| --- | ---: | ---: | ---: | ---: |
| Day / Clear | 61.2 | 277.5 | 42.8 | 229.5 |
| Night | 56.1 | 259.3 | 34.2 | 201.7 |
| Rainy | 57.5 | 265.8 | 36.1 | 210.4 |

| Method | Parameters (B) | GFLOPs | Speed (FPS) |
| --- | ---: | ---: | ---: |
| LLaVA-1.5 (7B) | 7.1 | 795 | 12.1 |
| Video-LLAMA (7B) | 7.8 | 910 | 10.5 |
| UniDrive | 8.2 | 980 | 9.8 |

The paper reports the speed on a single NVIDIA A100 GPU.

## Limitations

UniDrive can still select a plausible but non-primary hazard when multiple
candidate risk objects appear together. The paper identifies two common failure
modes: over-prioritizing dynamic agents over static but path-relevant obstacles,
and over-prioritizing large nearby objects over smaller trajectory-critical
objects. Future work should add risk ranking, trajectory-aware relevance,
multi-risk annotations, BEV or map priors, and more deployment-oriented
latency/reliability validation.

## Citation

```bibtex
@article{gao2026unidrive,
  title = {UniDrive: A Unified Vision-Language and Grounding Framework for Interpretable Risk Understanding in Autonomous Driving},
  author = {Gao, Xiaowei and Li, Pengxiang and Cheng, Yitai and Xu, Ruihan and Haworth, James and Law, Stephen and Ye, Yun},
  journal = {Preprint submitted to Elsevier},
  year = {2026}
}
```

## Acknowledgements

This codebase builds on
[LLaVA](https://github.com/haotian-liu/LLaVA) and uses components from the
LLaVA/LLaMA/CLIP ecosystem. UniDrive is evaluated on an extended
DRAMA-Reasoning setting derived from the DRAMA risk localization and captioning
benchmark. Please follow the licenses and terms for all base models, datasets,
and checkpoints used in your experiments.

## License

The inherited LLaVA code is released under the Apache License 2.0. Dataset,
model, and checkpoint usage may be subject to their own licenses and terms.
