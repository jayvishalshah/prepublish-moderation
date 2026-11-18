# Multimodal AI Content Moderation System

An AI-powered video content moderation system that analyzes user-uploaded videos and accompanying text before publication. Built using the pretrained CLIP (Contrastive Language–Image Pre-training) model, the system analyzes video frames through image-text similarity and example-based classification to identify potentially policy-violating content.

The system classifies content into three moderation categories: **Allowed**, **Not Allowed**, and **Flagged for Human Review**.

**Live Demo:** [Try the Hugging Face Space](https://huggingface.co/spaces/jayvishalshah/prepublish-moderation-demo)

**GitHub Repository:** [prepublish-moderation](https://github.com/jayvishalshah/prepublish-moderation)

---

## Project Overview

Online platforms receive large volumes of user-generated video content. Manually reviewing every upload is time-consuming and difficult to scale.

This project explores an AI-based approach to video content moderation using CLIP, a pretrained vision-language model.

The system extracts representative frames from uploaded videos and uses CLIP to compare visual content with text-based moderation descriptions and examples. Based on the classification results and custom moderation policies, it generates a moderation decision.

The objective is to improve moderation efficiency while maintaining a human-in-the-loop approach for ambiguous or context-dependent cases.

## Key Features

- **Vision-Language Analysis:** Uses CLIP to understand visual content through image-text similarity.
- **Example-Based Classification:** Compares extracted video frames against predefined text descriptions and examples.
- **Video Frame Analysis:** Extracts representative frames from uploaded videos for moderation.
- **Three-Tier Moderation:** Classifies content into:
  - 🟢 **Allowed:** Content that meets platform policies.
  - 🔴 **Not Allowed:** Content identified as violating platform policies.
  - 🟡 **Flagged for Human Review:** Ambiguous or context-dependent content requiring manual evaluation.
- **Custom Policy Evaluation:** Evaluates visual content against predefined moderation policies.
- **Pre-Publication Moderation:** Integrates moderation into a video upload workflow before publication.
- **Human-in-the-Loop Review:** Allows uncertain cases to receive additional human evaluation.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Model Framework | Hugging Face Transformers |
| Vision-Language Model | CLIP (Pretrained) |
| Video Processing | OpenCV |
| Application Framework | Streamlit |
| Model Platform | Hugging Face Spaces |

## System Architecture

The moderation workflow consists of the following stages:

1. **Video Input:** The user submits a video through the application.
2. **Frame Extraction:** OpenCV processes the video and extracts representative frames.
3. **Visual Analysis:** CLIP evaluates extracted frames using image-text similarity.
4. **Example-Based Classification:** Visual content is compared against predefined moderation descriptions and examples.
5. **Policy Evaluation:** Classification results are evaluated against custom moderation policies.
6. **Moderation Decision:** The system assigns one of three outcomes: Allowed, Not Allowed, or Flagged for Human Review.
7. **Human Review:** Flagged cases can be escalated for manual assessment.

## Moderation Categories

| Category | Description | Action |
|---|---|---|
| Allowed | No policy violation identified | Eligible for publication |
| Not Allowed | Policy violation identified | Block publication |
| Flagged for Human Review | Uncertain or context-dependent result | Manual evaluation |

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/jayvishalshah/prepublish-moderation.git
cd prepublish-moderation
```

### 2. Navigate to the project directory

```bash
cd prepublish-moderation-demo
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run <path-to-app-entrypoint.py>
```

Replace the placeholder with the actual Streamlit application file in the `src` directory.

## Deployment

The application is deployed using Hugging Face Spaces.

**Live Application:**

https://huggingface.co/spaces/jayvishalshah/prepublish-moderation-demo

## Current Development Status

The system is currently deployed and functional.

The project uses the pretrained CLIP model without fine-tuning. Visual classification is explored through image-text similarity and example-based comparisons.

Ongoing development focuses on:

- Expanding the evaluation dataset.
- Improving classification performance on edge cases.
- Evaluating CLIP's image-text similarity results.
- Improving contextual policy interpretation.
- Refining the human-review workflow.
- Reducing false positives and false negatives.

## Limitations and Considerations

AI-based moderation is not perfect. Model predictions can be affected by contextual ambiguity, visual quality, frame selection, and limitations in pretrained model capabilities.

CLIP's image-text similarity scores are not calibrated probabilities of policy violations. Results can also depend on the quality and selection of text descriptions and reference examples.

The three-tier classification approach is intended to support human moderation rather than replace human judgment entirely.

Flagged cases should receive additional review before a final moderation decision.

## Future Improvements

- Advanced CLIP-based classification strategies.
- Improved contextual understanding of video content.
- Expanded policy-specific evaluation datasets.
- Confidence-based moderation and escalation.
- Model evaluation using precision, recall, F1-score, and confusion matrices.
- Exploration of multimodal video analysis improvements.
- Scalable video moderation pipelines.

## Author

**Jayvishal Shah**

AI/ML Engineering | AI Safety | Content Moderation | Multimodal Machine Learning

- GitHub: https://github.com/jayvishalshah
- Hugging Face: https://huggingface.co/jayvishalshah

---

*This project explores the application of pretrained vision-language models to scalable, policy-aware video content moderation with human oversight.*
