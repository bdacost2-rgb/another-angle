"""Build Another Angle's archive: public-domain photos and passages, each with a
CLIP embedding, written to data/items.json and data/emb.bin.

Runs on GitHub (Actions > "Build archive data" > Run workflow) or in Colab:
    !pip -q install transformers torch pillow requests scikit-learn
    !python build_data.py --out data

Photos stay hosted by the museums (we store links, not copies). The embeddings
use the same CLIP model the website runs in the browser (openai/clip-vit-base-patch32),
so your own photo or words land in the same space as the archive.
"""
import argparse
import io
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

UA = {"User-Agent": "AnotherAngle/1.0 (ASU student class project; public-domain inspiration site)"}
SEED = 7

# Everyday subjects, so concert, pet, street, and nature photographers all find themselves here.
QUERIES = ["autumn", "desert", "snow", "field", "forest", "mountains", "ocean", "night", "portrait", "kitchen",
           "road", "river", "fog", "farm", "garden", "rain", "children", "dog", "horse", "street", "dance", "music",
           "window", "storm", "train", "lake", "flowers", "city", "church", "sunset"]

# Uncharted's themes: placed on the map where their closest photos and passages sit.
UNCHARTED_THEMES = [
    dict(n="Southern Gothic", desc="decaying southern houses, spanish moss, family secrets, heat and religion",
         place="the Mississippi Delta", year=1932, words=["spanish moss", "a revival tent", "rot in the porch beams", "heat that hums"],
         pal=["#3b4a2a", "#7a4b2a", "#c9b38a", "#1d1a14"]),
    dict(n="Film noir", desc="night city, hard shadows, rain, crime, venetian blinds, black and white",
         place="Los Angeles", year=1947, words=["venetian-blind light", "wet asphalt", "a phone that won't stop", "cigarette smoke"],
         pal=["#0e0e10", "#5a5a5f", "#d9d4c7", "#2b2e3a"]),
    dict(n="Afrofuturism", desc="Black futures, ancestral technology, cosmic, vivid gold and color",
         place="Lagos", year=2140, words=["brass and kente", "a choir in orbit", "ancestors in the code", "solar sails"],
         pal=["#2a1b4a", "#d59a2b", "#18a49a", "#f2e4c4"]),
    dict(n="Magical realism", desc="ordinary village life where impossible things happen calmly, warm color",
         place="a river town", year=1967, words=["yellow butterflies", "a grandmother who floats", "rain for four years", "letters from the dead"],
         pal=["#6c8a5b", "#e2b45a", "#b5523b", "#f4e9cf"]),
    dict(n="Pastoral", desc="countryside, fields, sheep, soft natural light, quiet rural life",
         place="the Yorkshire moors", year=1890, words=["a stone wall", "sheep in fog", "a kettle on", "the long field"],
         pal=["#7a9a5c", "#d8cfa2", "#9db6c4", "#4d5a35"]),
    dict(n="Cyberpunk", desc="neon city at night, rain, technology, crowded streets, pink and cyan light",
         place="Neo-Seoul", year=2077, words=["neon in puddles", "a vending-machine shrine", "rain on chrome", "borrowed memories"],
         pal=["#0b0f2a", "#ff3ea5", "#28e0ff", "#2a2f4a"]),
    dict(n="Surrealism", desc="dreamlike impossible scenes, strange juxtapositions, melting forms",
         place="Paris", year=1928, words=["a clock melting", "a door in the sea", "eyes in the bread", "a staircase to nothing"],
         pal=["#d9c9a3", "#2e5c8a", "#b8392a", "#efe6d2"]),
    dict(n="Documentary", desc="real people, streets, work, protest, everyday life, candid",
         place="Harlem", year=1968, words=["a stoop at dusk", "hands at work", "a protest sign", "a kitchen radio"],
         pal=["#3a3a3a", "#8c8577", "#d1cbbd", "#5b4a3a"]),
    dict(n="Minimalism", desc="empty space, single object, clean lines, calm, muted",
         place="Kyoto", year=1975, words=["one chair", "a white wall", "a single branch", "silence"],
         pal=["#e9e6df", "#b9b2a6", "#2c2c2c", "#c8cfd2"]),
    dict(n="Folk horror", desc="rural rituals, villages, harvest, unsettling old traditions",
         place="a Cornish village", year=1973, words=["straw figures", "a maypole", "the wrong hymn", "harvest smoke"],
         pal=["#2d2a1e", "#8a6d3b", "#c1b59a", "#5e1f17"]),
    dict(n="Cosmic", desc="stars, night sky, space, vast emptiness, deserts at night",
         place="the Atacama desert", year=1974, words=["a radio dish", "too many stars", "a signal answering", "the cold"],
         pal=["#05060f", "#3d3a7a", "#c7b8ff", "#1a1f3d"]),
    dict(n="Romanticism", desc="dramatic nature, storms, lakes, mountains, longing, the sublime",
         place="the Lake District", year=1818, words=["a storm over the lake", "a lone wanderer", "ruins", "a letter never sent"],
         pal=["#3e4c5e", "#c69c6d", "#e8dcc2", "#6b3e3a"]),
    dict(n="Harlem Renaissance", desc="1920s Black art, jazz clubs, poetry, elegant portraits, city nights",
         place="Harlem", year=1926, words=["a trumpet through the window", "a poet's rent party", "pressed suits", "a streetlamp choir"],
         pal=["#1f1a2e", "#c8973f", "#e9d7b0", "#6b2f3a"]),
    dict(n="Dark academia", desc="old libraries, candlelight, tweed, gothic universities, secret societies",
         place="Oxford", year=1912, words=["a library after hours", "ink on cuffs", "a locked reading room", "chapel bells"],
         pal=["#2b2118", "#6e4b2d", "#c9b48f", "#40352b"]),
    dict(n="Cottagecore", desc="cottages, gardens, baking, wildflowers, soft homemade rural comfort",
         place="a Cotswolds cottage", year=1952, words=["bread cooling on the sill", "wildflowers in a jar", "a quilt on the line", "honey"],
         pal=["#a7b88a", "#f0e3c4", "#d99a8a", "#6f7f52"]),
    dict(n="Brutalism", desc="raw concrete buildings, massive geometry, stark shadows, modernist housing",
         place="East Berlin", year=1971, words=["a concrete stairwell", "a thousand identical windows", "wind between towers", "a lone tram"],
         pal=["#7d7a74", "#b6b1a7", "#2f2e2c", "#9a8f7e"]),
    dict(n="Vaporwave", desc="pastel pink and teal, 80s malls, statues, glitch, nostalgic digital dreams",
         place="a dead shopping mall", year=1989, words=["a fountain in an empty mall", "a marble bust in pink light", "a VHS hum", "palm trees on a screen"],
         pal=["#ff9ad5", "#7ee8fa", "#2d1b4e", "#f6d1ff"]),
    dict(n="Solarpunk", desc="hopeful green futures, gardens on buildings, solar panels, community",
         place="Nairobi", year=2085, words=["rooftop orchards", "a solar market", "vines on glass towers", "bicycles in the sun"],
         pal=["#4e8f4a", "#f2d15a", "#8fd0c7", "#f5f1dc"]),
    dict(n="Americana", desc="road trips, diners, motels, gas stations, open highway, small towns",
         place="Route 66", year=1958, words=["a neon motel sign", "a diner at 2am", "heat on the highway", "a jukebox"],
         pal=["#b2392a", "#e8c07d", "#3c6e8f", "#f3ead7"]),
    dict(n="Dystopia", desc="surveillance, grey crowds, control, propaganda, cold authoritarian cities",
         place="Airstrip One", year=1984, words=["a screen that watches", "ration lines", "a slogan on concrete", "whispered names"],
         pal=["#3b3f42", "#8a8f8c", "#c1272d", "#1e2124"]),
    dict(n="Fairy tale", desc="enchanted forests, castles, wolves, witches, old storybook illustration",
         place="the Black Forest", year=1812, words=["a path of breadcrumbs", "a wolf at the door", "a glass slipper", "a thorn hedge"],
         pal=["#23402f", "#b9893a", "#e6d7b4", "#5e2a4a"]),
    dict(n="Greek myth", desc="ancient temples, marble gods, the sea, heroes, oracles and tragedy",
         place="Delphi", year=-480, words=["an oracle's smoke", "a wine-dark sea", "a broken column", "laurel leaves"],
         pal=["#e8e1d3", "#2f5d8a", "#c27a3a", "#8c7b65"]),
    dict(n="Desert Southwest", desc="saguaro cactus, monsoon storms, adobe, red rock, heat shimmer, big sky",
         place="the Sonoran Desert", year=1949, words=["a monsoon on the horizon", "a saguaro at dusk", "adobe walls", "creosote after rain"],
         pal=["#c4622d", "#e9b872", "#5d7d8c", "#3d2a1e"]),
    dict(n="Maritime", desc="the sea, lighthouses, fishing boats, harbors, storms, sailors",
         place="a Maine harbor", year=1903, words=["a lighthouse beam", "nets drying", "a widow's walk", "salt on everything"],
         pal=["#1f3a4d", "#7aa0b0", "#e3ddd0", "#8c3b2e"]),
    dict(n="Urban decay", desc="abandoned buildings, rust, graffiti, overgrown ruins, empty factories",
         place="Detroit", year=2009, words=["a piano in a flooded ballroom", "trees through a roof", "rust blooms", "graffiti saints"],
         pal=["#4a4036", "#8d6e4f", "#6f8f5e", "#c9c0b0"]),
    dict(n="Coming of age", desc="teenage summers, friendship, first love, bikes, suburbs, nostalgia",
         place="a suburban cul-de-sac", year=1996, words=["bikes on a lawn", "a sprinkler at dusk", "a mixtape", "the last day of summer"],
         pal=["#e7b75f", "#7fb3c9", "#e98a6b", "#f6efe0"]),
]

BOOKS = [1342, 84, 2701, 768, 1260, 1322, 12242, 205, 2814, 64317, 345, 174, 43, 5200, 76, 219, 160, 98, 11, 1952,
         1661, 1400, 4300, 16328, 2600]


def get(url, **kw):
    last = RuntimeError("rate limited: " + url)
    for t in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=40, **kw)
            if r.status_code == 429:
                time.sleep(3 + 3 * t)
                continue
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1 + t)
    raise last


# ---------------- photo sources ----------------
def loc(n):
    out, seen = [], set()
    for q in QUERIES:
        if len(out) >= n * 1.5:
            break
        try:
            res = get("https://www.loc.gov/photos/", params={"q": q, "fo": "json", "c": 50, "fa": "online-format:image"}).json()
        except Exception as e:  # noqa: BLE001
            print("  LoC", q, e)
            continue
        for r in res.get("results", []):
            urls = [u for u in (r.get("image_url") or []) if isinstance(u, str)]
            if not urls or r.get("id") in seen:
                continue
            def width(u):
                m = re.search(r"w=(\d+)", u)
                return int(m.group(1)) if m else 0
            ok = [u for u in urls if 0 < width(u) <= 1024] or urls
            url = max(ok, key=width).split("#")[0]
            url = "https:" + url if url.startswith("//") else url
            seen.add(r.get("id"))
            out.append({"img": url, "title": (r.get("title") or "Untitled")[:120], "creator": "", "date": r.get("date") or "",
                        "source": "Library of Congress", "link": r.get("url") or r.get("id") or ""})
    random.Random(SEED).shuffle(out)
    return out[:n]


def aic(n):
    out, seen = [], set()
    fields = "id,title,image_id,artist_title,date_display,is_public_domain"
    for q in ["photograph"] + QUERIES:
        if len(out) >= n * 1.5:
            break
        try:
            data = get("https://api.artic.edu/api/v1/artworks/search", params={"q": q, "limit": 60, "fields": fields,
                       "query[term][is_public_domain]": "true"}).json().get("data", [])
        except Exception as e:  # noqa: BLE001
            print("  AIC", q, e)
            continue
        for a in data:
            if not a.get("image_id") or not a.get("is_public_domain") or a["id"] in seen:
                continue
            seen.add(a["id"])
            out.append({"img": f"https://www.artic.edu/iiif/2/{a['image_id']}/full/600,/0/default.jpg",
                        "title": (a.get("title") or "")[:120], "creator": a.get("artist_title") or "",
                        "date": a.get("date_display") or "", "source": "Art Institute of Chicago",
                        "link": f"https://www.artic.edu/artworks/{a['id']}"})
    random.Random(SEED).shuffle(out)
    return out[:n]


def met(n):
    ids = []
    for q in ["photograph"] + QUERIES[:12]:
        try:
            ids += (get("https://collectionapi.metmuseum.org/public/collection/v1/search",
                        params={"q": q, "hasImages": "true", "departmentId": 19}).json().get("objectIDs") or [])[:200]
        except Exception as e:  # noqa: BLE001
            print("  Met", q, e)
    ids = list(dict.fromkeys(ids))
    random.Random(SEED).shuffle(ids)

    def meta(oid):
        try:
            o = get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}").json()
            if o.get("isPublicDomain") and o.get("primaryImageSmall"):
                return {"img": o["primaryImageSmall"], "title": (o.get("title") or "")[:120],
                        "creator": o.get("artistDisplayName") or "", "date": o.get("objectDate") or "",
                        "source": "The Met", "link": o.get("objectURL") or ""}
        except Exception:  # noqa: BLE001
            return None
        return None

    with ThreadPoolExecutor(6) as ex:
        found = [m for m in ex.map(meta, ids[: n * 2]) if m]
    return found[:n]


# ---------------- writing ----------------
def gutenberg(book_id, per_book):
    text = get(f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt").text
    title = re.search(r"^Title:\s*(.+)$", text, re.M)
    author = re.search(r"^Author:\s*(.+)$", text, re.M)
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF", text)
    body = text[start.end() if start else 0: end.start() if end else len(text)]
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", body)]
    paras = [p for p in paras if len(p.split()) >= 6 and sum(c.isupper() for c in p) < 0.3 * len(p)
             and not re.match(r"^(chapter|book|part|contents)\b", p, re.I)]
    chunks, cur = [], []
    for p in paras:
        cur.append(p)
        n = sum(len(x.split()) for x in cur)
        if n >= 35:
            if n <= 90:  # short enough to read on a card, and for CLIP to read in full
                chunks.append(" ".join(cur))
            cur = []
    random.Random(SEED + book_id).shuffle(chunks)
    meta = {"title": title.group(1).strip() if title else str(book_id), "creator": author.group(1).strip() if author else "",
            "source": "Project Gutenberg", "link": f"https://www.gutenberg.org/ebooks/{book_id}", "date": ""}
    return [{**meta, "text": c} for c in chunks[:per_book]]


# ---------------- embeddings ----------------
def load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).eval()
    proc = CLIPProcessor.from_pretrained(name)
    return torch, model, proc


def fetch_image(url):
    try:
        img = Image.open(io.BytesIO(get(url).content)).convert("RGB")
        img.thumbnail((448, 448))
        return img
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--loc", type=int, default=600)
    ap.add_argument("--aic", type=int, default=600)
    ap.add_argument("--met", type=int, default=400)
    ap.add_argument("--per-book", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    photos = []
    for name, fn, n in [("Library of Congress", loc, args.loc), ("Art Institute of Chicago", aic, args.aic), ("The Met", met, args.met)]:
        if n:
            got = fn(n)
            print(f"{name}: {len(got)} candidates")
            photos += got
    texts = []
    for b in BOOKS:
        try:
            got = gutenberg(b, args.per_book)
            print(f"Gutenberg {b}: {len(got)} passages from {got[0]['title'] if got else '?'}")
            texts += got
        except Exception as e:  # noqa: BLE001
            print(f"Gutenberg {b}: skipped ({e})")

    torch, model, proc = load_clip()
    print(f"Downloading and embedding {len(photos)} images...")
    kept, vecs = [], []
    B = 32
    for i in range(0, len(photos), B):
        batch = photos[i:i + B]
        with ThreadPoolExecutor(8) as ex:
            imgs = list(ex.map(lambda p: fetch_image(p["img"]), batch))
        pairs = [(p, im) for p, im in zip(batch, imgs) if im is not None]
        if not pairs:
            continue
        with torch.no_grad():
            e = model.get_image_features(**proc(images=[im for _, im in pairs], return_tensors="pt"))
        vecs.append(torch.nn.functional.normalize(e, dim=-1).numpy())
        kept += [p for p, _ in pairs]
        print(f"  {len(kept)} images", end="\r")
    print()
    print(f"Embedding {len(texts)} passages...")
    tvecs = []
    for i in range(0, len(texts), 64):
        batch = [t["text"] for t in texts[i:i + 64]]
        with torch.no_grad():
            e = model.get_text_features(**proc(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77))
        tvecs.append(torch.nn.functional.normalize(e, dim=-1).numpy())

    items = [{"id": f"p{i}", "kind": "photo", **p} for i, p in enumerate(kept)] + \
            [{"id": f"t{i}", "kind": "text", **t} for i, t in enumerate(texts)]
    E = np.concatenate(vecs + tvecs)
    emb = E.astype("<f2")

    # Uncharted's map: one 2-D layout of every photo and passage (t-SNE on CLIP, each kind centered
    # separately so photos and passages mix by meaning instead of sitting in two islands).
    from sklearn.manifold import TSNE
    nP = len(kept)
    def l2n(a):
        return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    Pm, Tm = E[:nP].mean(0), E[nP:].mean(0)
    X = np.vstack([l2n(E[:nP] - Pm), l2n(E[nP:] - Tm)])
    print("Laying out the Uncharted map (t-SNE)...")
    xy = TSNE(n_components=2, init="pca", perplexity=35, random_state=SEED).fit_transform(X)
    xy = 0.04 + 0.92 * (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-9)
    for it, (x, y) in zip(items, xy):
        it["x"], it["y"] = round(float(x), 4), round(float(y), 4)
    with torch.no_grad():
        th = model.get_text_features(**proc(text=[f"{t['n']}: {t['desc']}" for t in UNCHARTED_THEMES],
                                            return_tensors="pt", padding=True))
    TH = l2n(l2n(th.numpy()) - Tm)
    themes = []
    for k, t in enumerate(UNCHARTED_THEMES):
        near = np.argsort(-(X @ TH[k]))[:20]
        x, y = np.median(xy[near], 0)
        themes.append({**t, "x": round(float(x), 4), "y": round(float(y), 4), "vec": [round(float(v), 4) for v in TH[k]]})
    assert len(items) == emb.shape[0]
    with open(os.path.join(args.out, "items.json"), "w", encoding="utf-8") as f:
        json.dump({"model": "openai/clip-vit-base-patch32", "dim": int(emb.shape[1]), "items": items, "themes": themes}, f, ensure_ascii=False)
    emb.tofile(os.path.join(args.out, "emb.bin"))
    print(f"Done: {len(kept)} photos, {len(texts)} passages -> {args.out}/items.json, {args.out}/emb.bin")


if __name__ == "__main__":
    sys.exit(main())
