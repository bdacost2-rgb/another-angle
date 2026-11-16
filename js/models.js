// AI models that run in the visitor's browser (free, private: nothing is uploaded).
// Loaded only when first needed, from the transformers.js CDN.
//   CLIP (ViT-B/32): reads your photo or your words, in the same space as the archive.
//   MiniLM: reads what you tell the coach, so it can match meaning, not just keywords.
const TJS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2";
let lib = null;
const cache = {};

async function tf() {
  if (!lib) lib = await import(TJS);
  return lib;
}

async function clipParts() {
  if (!cache.clip) {
    cache.clip = (async () => {
      const t = await tf();
      const id = "Xenova/clip-vit-base-patch32";
      const [processor, vision, tokenizer, text] = await Promise.all([
        t.AutoProcessor.from_pretrained(id),
        t.CLIPVisionModelWithProjection.from_pretrained(id),
        t.AutoTokenizer.from_pretrained(id),
        t.CLIPTextModelWithProjection.from_pretrained(id),
      ]);
      return { t, processor, vision, tokenizer, text };
    })();
  }
  return cache.clip;
}

export async function clipImage(fileOrUrl) {
  const { t, processor, vision } = await clipParts();
  const url = typeof fileOrUrl === "string" ? fileOrUrl : URL.createObjectURL(fileOrUrl);
  const image = await t.RawImage.read(url);
  const inputs = await processor(image);
  const { image_embeds } = await vision(inputs);
  return Float32Array.from(image_embeds.data);
}

export async function clipText(str) {
  const { tokenizer, text } = await clipParts();
  const inputs = tokenizer([str], { padding: true, truncation: true });
  const { text_embeds } = await text(inputs);
  return Float32Array.from(text_embeds.data);
}

export async function sentences(list) {
  if (!cache.mini) {
    cache.mini = (async () => (await tf()).pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2"))();
  }
  const extract = await cache.mini;
  const out = await extract(list, { pooling: "mean", normalize: true });
  const dim = out.dims[out.dims.length - 1];
  return list.map((_, i) => out.data.slice(i * dim, (i + 1) * dim));
}
