import pickle
import numpy as np
import os

# Check all batch files exist
embedding_batches = [f'embeddings_batch{i}.pkl' for i in range(1, 5)]
filename_batches = [f'filenames_batch{i}.pkl' for i in range(1, 5)]

for f in embedding_batches + filename_batches:
    if not os.path.exists(f):
        raise FileNotFoundError(f"File not found: {f}")

# Merge embeddings
all_embeddings = []
for f in embedding_batches:
    with open(f, 'rb') as file:
        batch = pickle.load(file)
        all_embeddings.append(batch)
all_embeddings = np.concatenate(all_embeddings, axis=0)

# Merge filenames
all_filenames = []
for f in filename_batches:
    with open(f, 'rb') as file:
        batch = pickle.load(file)
        all_filenames.extend(batch)

# Save merged files
with open('embeddings_all.pkl', 'wb') as f:
    pickle.dump(all_embeddings, f)

with open('filenames_all.pkl', 'wb') as f:
    pickle.dump(all_filenames, f)

print(f"Merge completed! Total images: {len(all_filenames)}")
