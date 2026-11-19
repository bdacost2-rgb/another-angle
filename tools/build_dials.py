"""Design dials for the Uncharted gallery: data/dials.json.

Each dial is a pair of opposite descriptions ("built, geometric" vs "wild, overgrown"). CLIP embeds
both poles with the same model the archive uses, and the dial is the direction from one pole to the
other in the archive's centered space. The gallery scores each room's works along every dial, and a
procedural generator builds the room from those scores.

Run after build_data.py (it reads data/items.json and data/emb.bin, and does not change them):
    python tools/build_dials.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from build_data import feats, load_clip  # noqa: E402

# id, low word, high word, low-pole prompts, high-pole prompts
DIALS = [
    ("growth", "built", "overgrown",
     ["a built place of concrete, steel and glass", "geometric man-made architecture", "a city street of buildings", "machines, factories and straight lines"],
     ["a wild overgrown landscape of trees and grass", "untamed nature, leaves and flowers", "a forest full of plants", "moss, vines and wild growth"]),
    ("night", "daylit", "nocturnal",
     ["bright midday sunlight", "a sunny day under a clear sky", "a brightly lit daytime scene", "harsh daylight and shadows"],
     ["a dark night with stars", "darkness, night and moonlight", "a scene lit only by lamps at night", "the black night sky"]),
    ("crowd", "sparse", "crowded",
     ["an empty, sparse, minimal scene", "a lone figure in a vast open space", "emptiness and silence", "a bare landscape with almost nothing in it"],
     ["a crowded, busy, cluttered scene", "a dense crowd of people", "many things packed together", "a bustling market full of people"]),
    ("storm", "still", "turbulent",
     ["a still, calm, peaceful scene", "quiet serene water at rest", "tranquil and motionless", "a gentle restful afternoon"],
     ["a violent storm with wind and waves", "chaos, turbulence and motion", "a raging fire and smoke", "war, struggle and upheaval"]),
    ("float", "grounded", "weightless",
     ["heavy, solid, earthbound stone", "a massive weight resting on the ground", "rooted, grounded and still", "a heavy iron machine"],
     ["floating, weightless and airborne", "flying through the air", "clouds, feathers and drifting", "levitating in the sky"]),
    ("warmth", "cold", "warm",
     ["a cold icy winter scene with snow", "cold blue light and frost", "a frozen landscape", "chilly grey weather"],
     ["a warm golden summer scene", "warm sunlight, heat and fire", "a hot glowing sunset", "warm amber candlelight"]),
    ("ruin", "pristine", "ruined",
     ["a pristine, new, polished and clean place", "brand new and perfectly kept", "spotless and orderly", "fresh, gleaming and untouched"],
     ["ruins, decay and abandonment", "a crumbling, weathered, broken building", "rust, dust and neglect", "an abandoned place falling apart"]),
    ("dream", "real", "dreamlike",
     ["an ordinary everyday documentary photograph", "a plain realistic scene of daily life", "a factual record of a real place", "ordinary people at work"],
     ["a surreal dreamlike fantasy", "a strange, magical, impossible vision", "an uncanny dream", "myth, magic and the fantastic"]),
    ("wet", "dry", "wet",
     ["a dry dusty desert", "arid sand and sun-baked rock", "parched cracked earth", "a dry barren plain"],
     ["the sea, rain and water", "a misty wet river", "waves, rain and ocean spray", "a lake in the rain"]),
    ("scale", "intimate", "monumental",
     ["a small intimate domestic interior", "a close-up of a small object", "a cozy little room", "a quiet private moment up close"],
     ["a vast monumental landscape", "towering mountains and huge structures", "an enormous epic panorama", "a giant cathedral"]),
    ("order", "irregular", "ordered",
     ["irregular, scattered and random", "a messy jumble of shapes", "wild, asymmetric and chaotic forms", "things thrown about at random"],
     ["symmetrical, regular and ordered", "a grid of repeated identical shapes", "rows of columns in perfect order", "precise geometric symmetry"]),
    ("era", "ancient", "futuristic",
     ["ancient ruins and old antiques", "an old historical scene from long ago", "a medieval or classical past", "old handmade things"],
     ["a futuristic world of neon and technology", "science fiction, spaceships and the future", "glowing screens and electronics", "a modern high-tech city"]),
]


def load_archive(path):
    meta = json.load(open(os.path.join(path, "items.json"), encoding="utf-8"))
    emb = np.fromfile(os.path.join(path, "emb.bin"), dtype="<f2").astype(np.float32).reshape(len(meta["items"]), meta["dim"])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    return meta, emb


def l2n(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data"
    meta, E = load_archive(path)
    kinds = np.array([it["kind"] for it in meta["items"]])
    means = {k: E[kinds == k].mean(0) for k in ("photo", "text")}
    # the same centered space as the site (js/data.js): each kind minus its own mean
    X = l2n(np.stack([E[i] - means[k] for i, k in enumerate(kinds)]))

    torch, model, proc = load_clip()

    def embed(texts):
        with torch.no_grad():
            e = feats(model.get_text_features(**proc(text=texts, return_tensors="pt", padding=True)))
        return l2n(l2n(e.numpy()) - means["text"])

    out = []
    for did, low, high, lows, highs in DIALS:
        axis = l2n(embed(highs).mean(0) - embed(lows).mean(0))
        s = X @ axis
        stats = {k: [round(float(s[kinds == k].mean()), 5), round(float(s[kinds == k].std()), 5)] for k in ("photo", "text")}
        out.append({"id": did, "low": low, "high": high, "vec": [round(float(v), 5) for v in axis], "stats": stats})
        order = np.argsort(s)
        title = lambda i: (meta["items"][i].get("title") or meta["items"][i].get("text", ""))[:40]  # noqa: E731
        print(f"{did:7s} low: {' | '.join(title(i) for i in order[:3])}")
        print(f"{'':7s} high: {' | '.join(title(i) for i in order[-3:])}")

    S = np.stack([X @ np.array(d["vec"]) for d in out])
    C = np.corrcoef(S)
    print("most correlated dial pairs:")
    pairs = sorted(((abs(C[i, j]), out[i]["id"], out[j]["id"], C[i, j]) for i in range(len(out)) for j in range(i + 1, len(out))), reverse=True)
    for _, a, b, c in pairs[:6]:
        print(f"  {a} / {b}: {c:+.2f}")
    with open(os.path.join(path, "dials.json"), "w", encoding="utf-8") as f:
        json.dump({"model": meta["model"], "dials": out}, f, separators=(",", ":"))
    print(f"Wrote {len(out)} dials -> {path}/dials.json")


if __name__ == "__main__":
    main()
