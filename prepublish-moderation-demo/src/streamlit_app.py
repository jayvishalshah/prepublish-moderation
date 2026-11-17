import torch
from transformers import CLIPModel, CLIPProcessor
import os
import tempfile
import json
import joblib
import numpy as np
import cv2
from PIL import Image

import streamlit as st

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Pre-Publish Moderation Demo", layout="wide")

MODEL_REPO = "jay12redteamer/prepublish-moderation-model"  # your HF model repo
CLF_FILENAME = "moderation_clf.joblib"
LABEL_FILENAME = "label_info.joblib"

# -----------------------------
# CLIP image embedding (512-dim)
# -----------------------------
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"  # outputs 512-dim image embeddings

@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    return model, processor

def frame_bgr_to_clip_embedding(frame_bgr: np.ndarray) -> np.ndarray:
    model, processor = load_clip()
    
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    
    inputs = processor(images=img, return_tensors="pt")
    
    with torch.no_grad():
        feats = model.get_image_features(pixel_values=inputs["pixel_values"])  # (1, 512)
        
        # Extract tensor from model output
        if isinstance(feats, torch.Tensor):
            feats = feats
        elif hasattr(feats, 'pooler_output'):
            feats = feats.pooler_output
        elif hasattr(feats, 'last_hidden_state'):
            feats = feats.last_hidden_state[:, 0]
        else:
            feats = feats[0] if isinstance(feats, tuple) else feats
        
        # Normalize the embeddings
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    
    return feats.cpu().numpy().astype(np.float32)

# Thresholds for demo decisions (adjusted for 50% model accuracy)
REMOVE_THRESHOLD = 0.35      # if not_allowed prob >= this -> removed
FLAG_THRESHOLD = 0.28        # else if flag_review prob >= this -> flag
# otherwise allowed


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_model_assets():
    """
    Loads model + label mapping.
    Tries local first (if you later add files to the Space), else downloads from Hugging Face Hub.
    """
    # 1) local paths (Space repo)
    local_clf = CLF_FILENAME
    local_lbl = LABEL_FILENAME

    try:
        clf = joblib.load(local_clf)
        label_info = joblib.load(local_lbl)
        return clf, label_info, "Loaded from Space files"
    except Exception:
        pass

    # 2) download from HF Hub
    from huggingface_hub import hf_hub_download

    clf_path = hf_hub_download(repo_id=MODEL_REPO, filename=CLF_FILENAME)
    lbl_path = hf_hub_download(repo_id=MODEL_REPO, filename=LABEL_FILENAME)

    clf = joblib.load(clf_path)
    label_info = joblib.load(lbl_path)
    return clf, label_info, f"Downloaded from Hugging Face: {MODEL_REPO}"


def sample_frames(video_path, num_frames=3):
    """
    Samples N frames evenly from a video and returns list of PIL Images.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        # fallback: just read first few frames
        frames = []
        for _ in range(num_frames):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()
        return frames

    idxs = np.linspace(0, max(total - 1, 0), num_frames).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()
    return frames


def simple_frame_features(pil_img):
    """
    Very simple feature extractor:
    - mean RGB
    - std RGB
    - image brightness mean/std
    Returns 8-dim vector.
    """
    arr = np.array(pil_img).astype(np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        arr = np.stack([arr, arr, arr], axis=-1)

    mean_rgb = arr.mean(axis=(0, 1))
    std_rgb = arr.std(axis=(0, 1))

    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2])
    mean_gray = np.array([gray.mean()], dtype=np.float32)
    std_gray = np.array([gray.std()], dtype=np.float32)

    feat = np.concatenate([mean_rgb, std_rgb, mean_gray, std_gray], axis=0)
    return feat


def video_to_embedding(video_path: str, num_frames: int = 3) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return None

    idxs = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)

    embs = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue

        e = frame_bgr_to_clip_embedding(frame_bgr)  # (1, 512)
        if e is not None:
            embs.append(e)

    cap.release()

    if len(embs) == 0:
        return None

    emb = np.mean(np.vstack(embs), axis=0, keepdims=True)  # (1, 512)
    return emb


def decide(probs_dict):
    """
    Apply thresholds to decide allowed / flag_review / not_allowed
    """
    p_not = probs_dict.get("not_allowed", 0.0)
    p_flag = probs_dict.get("flag_review", 0.0)
    p_allow = probs_dict.get("allowed", 0.0)

    if p_not >= REMOVE_THRESHOLD:
        return "not_allowed", p_not
    if p_flag >= FLAG_THRESHOLD:
        return "flag_review", p_flag
    return "allowed", p_allow


# -----------------------------
# UI
# -----------------------------
st.title("Pre-Publish Moderation Demo (Prototype)")
st.caption(
    "Upload a short video → model predicts **allowed / flag_review / not_allowed** → "
    "we simulate what would happen before publishing."
)

# Add warning about prototype status
st.warning(
    "⚠️ **PROTOTYPE NOTICE**: This model was trained on 46 samples with 50% accuracy. "
    "Results are for demonstration purposes only and should not be used in production. "
    "For reliable moderation, the model needs significantly more diverse training data (200+ videos per class recommended)."
)

with st.sidebar:
    st.subheader("Demo settings")
    num_frames = st.slider("Frames to sample", 1, 8, 3)
    st.write("Decision thresholds")
    st.write(f"- Remove if **not_allowed ≥ {REMOVE_THRESHOLD:.2f}**")
    st.write(f"- Flag if **flag_review ≥ {FLAG_THRESHOLD:.2f}** (and not removed)")
    st.divider()

    try:
        clf, label_info, load_msg = load_model_assets()
        st.success(load_msg)
    except Exception as e:
        st.error("Failed to load model files.")
        st.code(str(e))
        st.stop()

    st.write("Label mapping (label_id → name):")
    st.json(label_info if isinstance(label_info, dict) else {"label_info": str(label_info)})
    
    # Show model training stats
    st.divider()
    st.subheader("Model Performance")
    st.metric("Training Accuracy", "50%", delta="-50% from ideal", delta_color="inverse")
    st.caption("Trained on 46 samples total")
    with st.expander("Class distribution"):
        st.write("- allowed: 1 sample")
        st.write("- not_allowed: 5 samples")  
        st.write("- flag_review: 4 samples")
        st.write("- Test set: 10 samples")

uploaded = st.file_uploader("Upload an MP4 video", type=["mp4", "mov", "m4v", "avi"])

if uploaded:
    # Save upload to temp file
    suffix = os.path.splitext(uploaded.name)[1].lower() or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    st.video(uploaded)

    st.write("### Running moderation…")
    emb = video_to_embedding(tmp_path, num_frames=num_frames)
    
    if emb is None:
        st.error("Could not read frames from this video. Try another file.")
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        st.stop()
    
    # Predict probabilities
    proba = clf.predict_proba(emb)[0]
    class_ids = clf.classes_

    # label_info is a dict like {"id2label": {...}, "label2id": {...}}
    id2label = {}
    if isinstance(label_info, dict):
        id2label = label_info.get("id2label", {}) or {}

    # make sure keys work whether they are "0" or 0
    def _id_to_label(cid):
        try:
            k = int(cid)
        except Exception:
            return str(cid)

        return id2label.get(str(k), id2label.get(k, str(k)))

    probs = {}
    for cid, p in zip(class_ids, proba):
        probs[_id_to_label(cid)] = float(p)

    decision, confidence = decide(probs)
    result = {
        "filename": os.path.basename(tmp_path),
        "decision": decision,
        "confidence": float(confidence),
        "probs": probs,
    }

    st.markdown("### Moderation Result")
    st.json(result)
    
    # Add interpretation help
    st.info(
        f"**Interpretation**: The model is {confidence:.1%} confident this video should be **{decision}**. "
        f"Due to limited training data (46 samples) and 50% accuracy, results should be validated by human reviewers."
    )

    # Show frames preview
    frames = sample_frames(tmp_path, num_frames=num_frames)
    if frames:
        st.markdown("### Sampled frames")
        cols = st.columns(min(len(frames), 4))
        for i, im in enumerate(frames[:4]):
            cols[i].image(im, caption=f"Frame {i+1}", use_container_width=True)

    # Simulate publish step
    st.markdown("### Pre-publish decision")
    if decision == "not_allowed":
        st.error("❌ NOT ALLOWED → Removed (not published)")
    elif decision == "flag_review":
        st.warning("⚠️ FLAG FOR REVIEW → Needs human review before publish")
    else:
        st.success("✅ ALLOWED → Published")

    try:
        os.remove(tmp_path)
    except Exception:
        pass
else:
    st.info("Upload a short video to test the prototype.")