# Fashion Recommendation System

Image-based fashion recommender. Product images are converted to embeddings
using a pretrained ResNet50 (ImageNet), and visually similar items are
retrieved with FAISS similarity search. Built with Streamlit.

## Features
- Upload one or more images and get visually similar items
- Filters for category, gender and color (require a metadata file)
- Outfit view showing the top match per uploaded item

## Setup
1. `pip install -r requirements.txt`
2. Download the dataset from: [DATASET LINK]
3. Generate embeddings with [SCRIPT NAME]; `merge.py` combines the batch files into `embeddings_all.pkl`
4. `streamlit run stylish.py`

The `.pkl` embedding files are not in this repo because of GitHub's file size limit.

## Tech
Python, TensorFlow/Keras, ResNet50 (pretrained), FAISS, Streamlit
