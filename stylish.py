# main.py
import streamlit as st
import numpy as np
import pickle, os, io
from PIL import Image, ImageEnhance
import faiss
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D

# ---------------------------
# Config
# ---------------------------
TOP_K = 6  # we will show TOP_K-1 recommendations per upload (skip self)
RECOMMEND_PER_UPLOAD = 5

# ---------------------------
# Utilities & caching
# ---------------------------
@st.cache_data(show_spinner=True)
def load_embeddings(emb_path='embeddings_all.pkl', filenames_path='filenames_all.pkl'):
    if not os.path.exists(emb_path) or not os.path.exists(filenames_path):
        raise FileNotFoundError(f"Place '{emb_path}' and '{filenames_path}' in the project folder.")
    embeddings = np.array(pickle.load(open(emb_path, 'rb')))
    filenames = pickle.load(open(filenames_path, 'rb'))
    return embeddings.astype('float32'), filenames

@st.cache_data(show_spinner=True)
def load_metadata():
    """
    Try to load metadata (optional). Support metadata.pkl, metadata.json, metadata.csv.
    metadata should be a dict mapping filename -> { 'category':..., 'gender':..., 'color':... }
    If none found, create simple fallback metadata by inferring category from parent folder or 'Unknown'.
    """
    # Try pkl/json/csv
    for fname in ('metadata.pkl', 'metadata.json', 'metadata.csv'):
        if os.path.exists(fname):
            try:
                if fname.endswith('.pkl'):
                    return pickle.load(open(fname,'rb'))
                elif fname.endswith('.json'):
                    import json
                    return json.load(open(fname,'r', encoding='utf-8'))
                else:  # csv
                    import csv
                    md = {}
                    with open(fname, newline='', encoding='utf-8') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            key = row.get('filename') or row.get('file') or row.get('path')
                            if not key: continue
                            md[key] = {
                                'category': row.get('category','Unknown'),
                                'gender': row.get('gender','Unisex'),
                                'color': row.get('color','Unknown')
                            }
                    return md
            except Exception:
                pass
    return None  # no metadata found

@st.cache_resource
def load_model():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base.trainable = False
    model = tf.keras.Sequential([base, GlobalMaxPooling2D()])
    return model

def enhance_display(img: Image.Image, sharpness=1.8, color=1.2, brightness=1.03):
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = ImageEnhance.Color(img).enhance(color)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    return img

def extract_feature_from_pil(img: Image.Image, model):
    x = img.resize((224,224)).convert('RGB')
    arr = np.asarray(x).astype('float32')
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    feat = model.predict(arr, verbose=0)
    feat = feat / np.linalg.norm(feat)
    return feat.astype('float32').ravel()

# ---------------------------
# Load data
# ---------------------------
try:
    embeddings, filenames = load_embeddings()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

metadata = load_metadata()  # may be None

# If no metadata, build a lightweight inferred metadata dict using folder names
if metadata is None:
    md = {}
    for p in filenames:
        base = os.path.basename(p)
        # infer category from parent directory name if available
        parent = os.path.basename(os.path.dirname(p))
        if parent == '':
            parent = 'Unknown'
        md[base] = {'category': parent, 'gender': 'Unisex', 'color': 'Unknown'}
    metadata = md

# Build sets for filters
all_categories = sorted({v.get('category','Unknown') for v in metadata.values() if v})
all_genders = sorted({v.get('gender','Unisex') for v in metadata.values() if v})
all_colors = sorted({v.get('color','Unknown') for v in metadata.values() if v})

# ---------------------------
# FAISS index for the whole dataset
# ---------------------------
d = embeddings.shape[1]
index_full = faiss.IndexFlatL2(d)
index_full.add(embeddings)  # store all embeddings

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(layout='wide', page_title="Outfit Builder & Filtered Recommender")
st.markdown("""
    <style>
    .title { text-align:center; font-size:28px; font-weight:700; margin-bottom:4px; }
    .subtitle { text-align:center; color:#555; margin-top:0; margin-bottom:18px; }
    .card { border-radius:12px; box-shadow: 0 6px 18px rgba(0,0,0,0.12); overflow:hidden; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="title">🧾 Outfit Builder & Filtered Recommender</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload multiple items (shirt, shoes...) and get matching recommendations within selected filters.</p>', unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("Filters")
gender_sel = st.sidebar.selectbox("Gender", options=["All"] + all_genders, index=0)
category_sel = st.sidebar.multiselect("Categories to search (optional)", options=["All"] + all_categories, default=["All"])
color_sel = st.sidebar.selectbox("Dominant Color", options=["All"] + all_colors, index=0)
num_recs = st.sidebar.slider("Recommendations per uploaded item", 1, 8, 5)

# Multi-upload for outfit builder
st.markdown("### Upload items for outfit building")
uploaded_files = st.file_uploader("Upload multiple images (shirt, pants, shoes...)",
                                  accept_multiple_files=True,
                                  type=['jpg','jpeg','png'])

# Helper: create filtered index and mapping
@st.cache_data
def build_filtered_index(category_sel, gender_sel, color_sel):
    # find indices that match filters
    keep_mask = np.ones(len(filenames), dtype=bool)
    # handle category
    if category_sel and "All" not in category_sel:
        cats = set(category_sel)
        keep_mask = np.array([ (metadata.get(os.path.basename(p),{}).get('category') in cats) for p in filenames ])
    # gender
    if gender_sel and gender_sel != "All":
        keep_mask = keep_mask & np.array([ metadata.get(os.path.basename(p),{}).get('gender') == gender_sel for p in filenames ])
    # color
    if color_sel and color_sel != "All":
        keep_mask = keep_mask & np.array([ metadata.get(os.path.basename(p),{}).get('color') == color_sel for p in filenames ])
    # build index on the filtered embeddings
    idxs = np.where(keep_mask)[0].astype('int32')
    if len(idxs)==0:
        return None, None, None
    filtered_feats = embeddings[idxs]
    idx_map = idxs  # map filtered index -> global index
    # build a small flat index for quick search
    idx = faiss.IndexFlatL2(d)
    idx.add(filtered_feats)
    return idx, idx_map, filtered_feats.shape[0]

# Build filtered index
filtered_index, filtered_map, filtered_count = build_filtered_index(category_sel, gender_sel, color_sel)

if filtered_index is None:
    st.warning("No images match the chosen filters. Try 'All' or different filters.")
else:
    st.success(f"{filtered_count} images match the filters.")

# If user uploaded images, process
if uploaded_files:
    # Show uploaded preview
    cols_preview = st.columns(len(uploaded_files))
    uploaded_pil = []
    for i, uf in enumerate(uploaded_files):
        try:
            img = Image.open(io.BytesIO(uf.read())).convert('RGB')
            uploaded_pil.append((uf.name, img))
            with cols_preview[i]:
                st.image(enhance_display(img:=img.copy() if 'img' in locals() else img), caption=uf.name, use_column_width=True)
        except Exception as e:
            st.error(f"Could not read {uf.name}: {e}")

    model = load_model()

    # For each uploaded image, extract feature and search within filtered index (or full index if None)
    all_results = []  # list of dicts per uploaded image
    for name, img in uploaded_pil:
        feat = extract_feature_from_pil(img, model)  # normalized feature
        # choose index to use
        if filtered_index is not None:
            D, I = filtered_index.search(np.array([feat]), k=min(TOP_K, filtered_index.ntotal))
            # map local indices to global
            global_idxs = [int(filtered_map[i]) for i in I[0]]
        else:
            D, I = index_full.search(np.array([feat]), k=TOP_K)
            global_idxs = [int(i) for i in I[0]]
        # Remove identical (if uploaded image is from dataset) by skipping exact match if present
        # Gather top N
        result_idxs = []
        for gi in global_idxs:
            if gi < 0 or gi >= len(filenames): continue
            result_idxs.append(gi)
            if len(result_idxs) >= RECOMMEND_PER_UPLOAD:
                break
        all_results.append({'uploaded_name': name, 'uploaded_feat': feat, 'recommendations': result_idxs})

    # Display recommendations per upload
    st.markdown("---")
    st.markdown("## Recommendations per uploaded item")
    for res in all_results:
        st.markdown(f"**Uploaded:** {res['uploaded_name']}")
        recs = res['recommendations']
        cols = st.columns(len(recs))
        for i, idx in enumerate(recs):
            with cols[i]:
                p = filenames[idx]
                # display high-res original with enhancement
                try:
                    rec_img = Image.open(p).convert('RGB')
                    rec_img = enhance_display(rec_img)
                    st.image(rec_img, use_column_width=True)
                    st.caption(os.path.basename(p))
                except Exception as e:
                    st.text("image load error")

    # Outfit assembly: pick one recommendation per uploaded item and show as a set
    st.markdown("---")
    st.markdown("## Assemble an outfit")
    st.write("Click the button to auto-assemble one top recommendation per uploaded image.")
    if st.button("Assemble outfit"):
        assembled = []
        for res in all_results:
            if res['recommendations']:
                assembled.append(res['recommendations'][0])
        if assembled:
            cols = st.columns(len(assembled))
            for i, idx in enumerate(assembled):
                with cols[i]:
                    try:
                        img = Image.open(filenames[idx]).convert('RGB')
                        img = enhance_display(img)
                        st.image(img, use_column_width=True)
                        st.caption(os.path.basename(filenames[idx]))
                    except:
                        st.text("load error")
        else:
            st.info("No assembled items (maybe filters are too strict).")

# Footer small notes
st.markdown("---")
st.markdown("Built with ❤️ — upload multiple items (shirt + shoes) to see paired recommendations. Filters restrict the search subset.")
