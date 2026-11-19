# Another Angle

Inspiration for photographers and writers that learns your taste.

- **Pick what moves you:** choose 8+ public-domain photos and passages. A taste model (Bayesian logistic regression on CLIP embeddings) learns from your picks.
- **Your feed:** each card pairs a photo with a passage that matches its meaning. Hearts and "less like this" keep training the model. Thompson sampling keeps it exploring where it is unsure.
- **More like this / search / your photo:** find work by vibe, by words ("quiet desert heat"), or by uploading your own photo. CLIP runs in your browser, so nothing is uploaded.
- **Uncharted (its own tab, `uncharted.html`):** after 8+ choices, scroll to walk over reflective water through a WebGL (three.js) gallery. Rooms are made by machine learning: liked works are clustered by aesthetic with spherical k-means on their CLIP embeddings (k chosen by silhouette score), and each cluster is named after its nearest curated theme. Each room is then generated, not picked from a menu: CLIP scores the room's works on twelve design dials (built ↔ overgrown, daylit ↔ nocturnal, sparse ↔ crowded, still ↔ turbulent, grounded ↔ weightless, cold ↔ warm, pristine ↔ ruined, real ↔ dreamlike, dry ↔ wet, intimate ↔ monumental, irregular ↔ ordered, ancient ↔ futuristic), each a direction between two sets of text prompts in CLIP space (`data/dials.json`, made by `tools/build_dials.py`). One procedural generator reads all twelve numbers at once to shape the terrain, plants and trees, buildings and their arrangement, colonnades and lamps, sky (sun, moon, stars, extra worlds, aurora, clouds), weather (rain, snow, dust, embers, fireflies) and light. Every room also gets a signature dial no other room has (an optimal one-to-one matching), so no two rooms are built around the same idea; the title card shows each room's strongest dials. Colors blend the theme palette with the room's own photos, and fog follows how tight the cluster is. Each room ends in a painterly backdrop made from its strongest photo. The Compass also adds rooms of likely matches and works near its uncertain boundary, and the walk ends in a ring room you turn through by scrolling. Move the pointer to look around; click a work for its archive credit, a related work from the other medium, and optional like/dislike feedback, which regroups the rooms. The 3-D map is readable and tied to the gallery: areas of the land are named by the same CLIP design dials (each word used once, e.g. "nocturnal · ancient", "wet · weightless"), hovering anywhere shows what that spot is like and its nearest works, and each gallery room is drawn as a constellation of its works that you can click to walk into that room. The route-finding checkpoints and logbook remain available; each checkpoint presents nearby archive photos and passages once, with titles and credits, per-work On track/Off track feedback, and a collection-level response. Checkpoint works follow a scroll-responsive spiral, with a small look/read cursor on desktop. Shown works are remembered across reloads so they are not proposed again. Rather than generating a composite image or passage, the checkpoint leaves connections to the viewer. Your gallery feedback trains the same taste model.
- **My photos:** add your own portfolio. CLIP reads it in your browser and it's saved on your device only (IndexedDB). Rediscover ranks your photos by your taste so forgotten strong ones resurface, and your photos join the feed, search, and the Uncharted map.
- **Writing coach (its own tab):** CreateAI, our teammate's writer's block coach, rebuilt for the web with the same stages and content, plus meaning-based diagnosis (MiniLM), exercise ranking that learns what helps you, and a piece of inspiration from your feed.

## Files
- `index.html`, `style.css`, `js/` — the site (no build step).
- `js/lens.js` — the cursor gently underlines the nearby word without blurring the passage; pick photos respond with a local hover zoom.
- `js/coach_data.js` — the teammate's categories, exercises, suggestions and diagnosis rules.
- `tools/build_data.py` — builds `data/items.json` and `data/emb.bin` from the Library of Congress, Art Institute of Chicago, The Met and Project Gutenberg.

## Build the archive data (once)
In Colab:
```
!pip -q install transformers torch pillow requests
!python build_data.py --out data
```
Then put the `data` folder next to `index.html`. (It also runs on a laptop: `pip install transformers torch pillow requests scikit-learn`, then `python tools/build_data.py --out data`.) After rebuilding the data, run `python tools/build_dials.py` to remake the gallery's design dials.

## Run locally
```
python -m http.server 8000
```
and open http://localhost:8000.
