# Another Angle

Inspiration for photographers and writers that learns your taste.

- **Pick what moves you:** choose 8+ public-domain photos and passages. A taste model (Bayesian logistic regression on CLIP embeddings) learns from your picks.
- **Your feed:** each card pairs a photo with a passage that matches its meaning. Hearts and "less like this" keep training the model. Thompson sampling keeps it exploring where it is unsure.
- **More like this / search / your photo:** find work by vibe, by words ("quiet desert heat"), or by uploading your own photo. CLIP runs in your browser, so nothing is uploaded.
- **Uncharted (its own tab, `uncharted.html`):** a 3-D map of the whole archive that you can drag to turn, in four sections (seed, map, checkpoint, logbook). Hills are your taste; fog shows where the model is unsure of your taste; routes (A*) search for empty spots your taste points to, and checkpoints ask you where it's foggiest. Starts from your Another Angle picks, and your answers train the same model. "I'm stuck here" uses the coach.
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
Then put the `data` folder next to `index.html`. (It also runs on a laptop: `pip install transformers torch pillow requests scikit-learn`, then `python tools/build_data.py --out data`.)

## Run locally
```
python -m http.server 8000
```
and open http://localhost:8000.
