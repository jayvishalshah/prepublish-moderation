# Multimodal AI Content Moderation System

An AI-powered video content moderation system that analyzes user-uploaded videos and accompanying text before publication. Built using fine-tuned CLIP and DistilBERT models, the system identifies potentially policy-violating content and classifies it into three moderation categories: **Allowed**, **Not Allowed**, and **Flagged for Human Review**.

**Live Demo:** [Try the Hugging Face Space](https://huggingface.co/spaces/jayvishalshah/prepublish-moderation-demo)

**GitHub Repository:** [prepublish-moderation](https://github.com/jayvishalshah/prepublish-moderation)

---

## Project Overview

Online platforms receive large volumes of user-generated video content. Manually reviewing every upload is time-consuming and difficult to scale.

This project explores a multimodal AI approach to content moderation by combining visual and textual analysis to identify potential policy violations before content goes live.

The system processes video frames and associated text captions, evaluates them against custom platform policies, and generates a moderation decision.

The objective is to improve moderation efficiency while maintaining a human-in-the-loop approach for ambiguous or context-dependent cases.

## Key Features

- **Multimodal Content Analysis:** Combines visual analysis of video frames with textual analysis of captions.
- **AI-Based Classification:** Uses fine-tuned CLIP and DistilBERT models for visual and textual content understanding.
- **Three-Tier Moderation:** Classifies content into:
  - 🟢 **Allowed:** Content that meets platform policies.
  - 🔴 **Not Allowed:** Content identified as violating platform policies.
  - 🟡 **Flagged for Human Review:** Ambiguous or context-dependent content requiring manual evaluation.
- **Context-Aware Policy Enforcement:** Evaluates content against custom moderation policies.
- **Pre-Upload Moderation:** Integrates moderation into a video upload workflow before publication.
- **Human-in-the-Loop Review:** Allows uncertain cases to receive additional human evaluation.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Model Framework | Hugging Face Transformers |
| Vision Model | CLIP |
| Text Model | DistilBERT |
| Video Processing | OpenCV |
| Application Framework | Streamlit |
| Model Platform | Hugging Face |

## System Architecture

The moderation workflow consists of the following stages:

1. **Video Input:** The user submits a video through the application.
2. **Frame Extraction:** OpenCV processes the video and extracts representative frames for analysis.
3. **Visual Analysis:** CLIP evaluates visual content in the extracted frames.
4. **Textual Analysis:** DistilBERT analyzes the accompanying text or captions.
5. **Policy Evaluation:** Model outputs are evaluated against custom moderation policies.
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

If running locally with Streamlit, use:

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

Ongoing development focuses on:

- Expanding the training dataset.
- Improving classification performance on edge cases.
- Evaluating multimodal model outputs.
- Improving contextual policy interpretation.
- Refining the human-review workflow.
- Reducing false positives and false negatives.

## Limitations and Considerations

AI-based moderation is not perfect. Model predictions can be affected by contextual ambiguity, visual quality, incomplete captions, and limitations in training data.

The three-tier classification approach is intended to support human moderation rather than replace human judgment entirely.

Flagged cases should receive additional review before a final moderation decision.

## Future Improvements

- Advanced multimodal fusion strategies.
- Improved contextual understanding of video content.
- Expanded policy-specific training datasets.
- Confidence-based moderation and escalation.
- Model evaluation using precision, recall, F1-score, and confusion matrices.
- Scalable video moderation pipelines.

## Author

**Jay Vishal Shah**

AI/ML Engineering | AI Safety | Content Moderation | Multimodal Machine Learning

- GitHub: https://github.com/jayvishalshah
- Hugging Face: https://huggingface.co/jayvishalshah

---

*This project explores the application of multimodal machine learning to scalable, policy-aware content moderation with human oversight.*
