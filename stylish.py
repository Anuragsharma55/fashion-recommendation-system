# stylish.py
import streamlit as st
import numpy as np
import pickle, os, io
from PIL import Image, ImageEnhance
import faiss
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D

st.set_page_config(layout='wide', page_title="Fashion Recommender")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------
# Utilities & caching
# ---------------------------
@st.cache_data(show_spinner=True)
def load_embeddings(emb_path='embeddings_demo.pkl', filenames_path='filenames_demo.pkl'):
    emb_path = os.path.join(BASE_DIR, emb_path)
    filenames_path = os.path.join(BASE_DIR, filenames_path)
    if not os.path.exists(emb_path) or not os.path.exists(filenames_path):
        raise FileNotFoundError(
            f"Place '{os.path.basename(emb_path)}' and '{os.path.basename(filenames_path)}' in the project folder."
        )
    embeddings = np.array(pickle.load(open(emb_path, 'rb')))
    filenames = pickle.load(open(filenames_path, 'rb'))
    # make image paths work on any machine (local or cloud)
    filenames = [os.path.join(BASE_DIR, f) for f in filenames]
    return embeddings.astype('float32'), filenames


@st.cache_data(show_spinner=True)
def load_metadata():
    """
    Optional metadata file (metadata.pkl, metadata.json or metadata.csv) mapping
    image filename -> {'category': ..., 'gender': ..., 'color': ...}.
    Returns None if no metadata file exists.
    """
    for fname in ('metadata.pkl', 'metadata.json', 'metadata.csv'):
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            try:
                if fname.endswith('.pkl'):
                    return pickle.load(open(path, 'rb'))
                elif fname.endswith('.json'):
                    import json
                    return json.load(open(path, 'r', encoding='utf-8'))
                else:
                    import csv
                    md = {}
                    with open(path, newline='', encoding='utf-8') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            key = row.get('filename') or row.get('file') or row.get('path')
                            if not key:
                                continue
                            md[key] = {
                                'category': row.get('category', 'Unknown'),
                                'gender': row.get('gender', 'Unisex'),
                                'color': row.get('color', 'Unknown'),
                            }
                    return md
            except Exception:
                pass
    return None


@st.cache_resource
def load_model():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    model = tf.keras.Sequential([base, GlobalMaxPooling2D()])
    return model


def enhance_display(img: Image.Image, sharpness=1.8, color=1.2, brightness=1.03):
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = ImageEnhance.Color(img).enhance(color)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    return img


def extract_feature_from_pil(img: Image.Image, model):
    x = img.resize((224, 224)).convert('RGB')
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

metadata = load_metadata()  # None if no metadata file is present
has_metadata = metadata is not None

if has_metadata:
    all_categories = sorted({v.get('category', 'Unknown') for v in metadata.values() if v})
    all_genders = sorted({v.get('gender', 'Unisex') for v in metadata.values() if v})
    all_colors = sorted({v.get('color', 'Unknown') for v in metadata.values() if v})

# ---------------------------
# FAISS index for the whole dataset
# ---------------------------
d = embeddings.shape[1]
index_full = faiss.IndexFlatL2(d)
index_full.add(embeddings)

# ---------------------------
# UI
# ---------------------------
st.markdown("""
    <style>
    .title { text-align:center; font-size:28px; font-weight:700; margin-bottom:4px; }
    .subtitle { text-align:center; color:#555; margin-top:0; margin-bottom:18px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="title">Fashion Recommender</div>', unsafe_allow_html=True)
st.markdown(
    f'<p class="subtitle">Upload clothing images and get visually similar items '
    f'from a collection of {len(filenames):,} images.</p>',
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.header("Options")
num_recs = st.sidebar.slider("Recommendations per uploaded item", 1, 8, 5)

gender_sel, category_sel, color_sel = "All", ["All"], "All"
if has_metadata:
    st.sidebar.header("Filters")
    gender_sel = st.sidebar.selectbox("Gender", options=["All"] + all_genders, index=0)
    category_sel = st.sidebar.multiselect("Categories to search (optional)",
                                          options=["All"] + all_categories, default=["All"])
    color_sel = st.sidebar.selectbox("Dominant Color", options=["All"] + all_colors, index=0)

st.markdown("### Upload items")
uploaded_files = st.file_uploader("Upload one or more images (shirt, pants, shoes...)",
                                  accept_multiple_files=True,
                                  type=['jpg', 'jpeg', 'png'])


@st.cache_data
def build_filtered_index(category_sel, gender_sel, color_sel):
    keep_mask = np.ones(len(filenames), dtype=bool)
    if category_sel and "All" not in category_sel:
        cats = set(category_sel)
        keep_mask = keep_mask & np.array(
            [metadata.get(os.path.basename(p), {}).get('category') in cats for p in filenames])
    if gender_sel and gender_sel != "All":
        keep_mask = keep_mask & np.array(
            [metadata.get(os.path.basename(p), {}).get('gender') == gender_sel for p in filenames])
    if color_sel and color_sel != "All":
        keep_mask = keep_mask & np.array(
            [metadata.get(os.path.basename(p), {}).get('color') == color_sel for p in filenames])
    idxs = np.where(keep_mask)[0].astype('int32')
    if len(idxs) == 0:
        return None, None, None
    idx = faiss.IndexFlatL2(d)
    idx.add(embeddings[idxs])
    return idx, idxs, len(idxs)


# Choose which index to search
if has_metadata:
    search_index, search_map, search_count = build_filtered_index(category_sel, gender_sel, color_sel)
    if search_index is None:
        st.warning("No images match the chosen filters. Try 'All' or different filters.")
    else:
        st.success(f"{search_count} images match the filters.")
else:
    search_index, search_map = index_full, None

# ---------------------------
# Process uploads
# ---------------------------
if uploaded_files and search_index is not None:
    cols_preview = st.columns(len(uploaded_files))
    uploaded_pil = []
    for i, uf in enumerate(uploaded_files):
        try:
            img = Image.open(io.BytesIO(uf.read())).convert('RGB')
            uploaded_pil.append((uf.name, img))
            with cols_preview[i]:
                st.image(enhance_display(img), caption=uf.name, use_container_width=True)
        except Exception as e:
            st.error(f"Could not read {uf.name}: {e}")

    model = load_model()

    all_results = []
    for name, img in uploaded_pil:
        feat = extract_feature_from_pil(img, model)
        k = min(num_recs, search_index.ntotal)
        D, I = search_index.search(np.array([feat]), k=k)
        if search_map is not None:
            global_idxs = [int(search_map[i]) for i in I[0] if i >= 0]
        else:
            global_idxs = [int(i) for i in I[0] if i >= 0]
        result_idxs = [gi for gi in global_idxs if 0 <= gi < len(filenames)]
        all_results.append({'uploaded_name': name, 'recommendations': result_idxs})

    st.markdown("---")
    st.markdown("## Recommendations per uploaded item")
    for res in all_results:
        st.markdown(f"**Uploaded:** {res['uploaded_name']}")
        recs = res['recommendations']
        if not recs:
            st.info("No recommendations found for this item.")
            continue
        cols = st.columns(len(recs))
        for i, idx in enumerate(recs):
            with cols[i]:
                p = filenames[idx]
                try:
                    rec_img = Image.open(p).convert('RGB')
                    st.image(enhance_display(rec_img), use_container_width=True)
                    st.caption(os.path.basename(p))
                except Exception as e:
                    st.text(f"image load error: {e}")

    # Outfit view: top match for each uploaded item
    st.markdown("---")
    st.markdown("## Outfit view")
    st.write("Shows the top match for each uploaded item side by side.")
    if st.button("Show top matches"):
        top_matches = [r['recommendations'][0] for r in all_results if r['recommendations']]
        if top_matches:
            cols = st.columns(len(top_matches))
            for i, idx in enumerate(top_matches):
                with cols[i]:
                    try:
                        img = Image.open(filenames[idx]).convert('RGB')
                        st.image(enhance_display(img), use_container_width=True)
                        st.caption(os.path.basename(filenames[idx]))
                    except Exception as e:
                        st.text(f"image load error: {e}")
        else:
            st.info("No matches to show.")

st.markdown("---")
st.markdown("Recommendations are the most visually similar items, found with ResNet50 image embeddings and FAISS.")
