import pickle, os, random
import numpy as np
from PIL import Image

N = 3000
random.seed(42)

emb = np.array(pickle.load(open('embeddings_all.pkl', 'rb'))).astype('float32')
names = pickle.load(open('filenames_all.pkl', 'rb'))

idx = sorted(random.sample(range(len(names)), N))
os.makedirs('demo_images', exist_ok=True)

demo_emb, demo_names = [], []
for i in idx:
    base = os.path.basename(names[i])
    try:
        img = Image.open(names[i]).convert('RGB')
        img.thumbnail((300, 300))
        img.save(os.path.join('demo_images', base), quality=75)
    except Exception as e:
        print('skip', base, e)
        continue
    demo_emb.append(emb[i])
    demo_names.append('demo_images/' + base)

pickle.dump(np.array(demo_emb, dtype='float32'), open('embeddings_demo.pkl', 'wb'))
pickle.dump(demo_names, open('filenames_demo.pkl', 'wb'))
print(len(demo_names), 'items saved')