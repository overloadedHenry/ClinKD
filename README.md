<div id="top" align="center">
<p align="center">
    <img src="https://capsule-render.vercel.app/api?type=waving&height=300&color=gradient&text=ClinKD:%20Cross-Modal%20Clinical%20Knowledge%20Distiller%20For%20Multi-Task%20Medical%20Images&reversal=false&fontSize=20&textBg=false&fontAlignY=42" alt="Header Image">
  </a>
</p>
</div>

## 💡 Highlights 

- 🔥 **Med-CLIP Guided RoPE:** We propose the Med-CLIP Guided RoPE to improve image-text alignment by fixing distinct intervals between different modal features.
- 🔥 **Clinical Knowledge Distiller:** The Clinical Knowlegde Distiller comprise Pseudo-Labels Medical Distillation and Reflective Correction Training. We use pseudo-labels to overcome the limitation caused by medical knowledge gap.
- 🔥 **Semantic-Aware Selective Generation:** The SASG part is used for the best answer with semantic similarity.

## Dataset
- For the images downloading, please refer to the [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D).
- For QA pairs, please search the git repo of [BiRD](https://github.com/ShawnHuang497/BiRD?tab=readme-ov-file).

## 🛠️ Usage
We recommend [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for training Qwen2-VL. You may need to convert the original dataset format to the ShareGPT format. We provide the python code for converting format in `data/transfer2sharegpt.py`
###  Train
Please refer to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

### Evaluation
### Step 1: Inference

```shell
sh test_metric/infer_all.sh
```

### Step 2: Evaluate
```shell
sh test_metric/eval_all.sh
```