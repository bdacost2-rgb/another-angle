"""Real history for every work: data/history.json.

Nothing here is generated. Each record is fetched from the archive that holds the work (the Art Institute of
Chicago, the Library of Congress, The Met), in that archive's own words, with a link back to the source.
For the Project Gutenberg passages, the books' first-publication years and the authors' life dates are
well-established facts, listed below.

Run after build_data.py (it reads data/items.json and does not change it):
    python tools/build_history.py
"""
import html
import json
import os
import re
import sys
import time
import urllib.request

UA = {"User-Agent": "AnotherAngle-class-project/1.0 (educational; history build)"}

# first publication (or composition, for Beowulf) and the author's life dates
BOOKS = {
    "98": ("1859", "Charles Dickens, 1812–1870"),
    "76": ("1884", "Mark Twain, 1835–1910"),
    "11": ("1865", "Lewis Carroll, 1832–1898"),
    "16328": ("c. 700–1000", "Anonymous Old English poet"),
    "345": ("1897", "Bram Stoker, 1847–1912"),
    "2814": ("1914", "James Joyce, 1882–1941"),
    "84": ("1818", "Mary Wollstonecraft Shelley, 1797–1851"),
    "1400": ("1861", "Charles Dickens, 1812–1870"),
    "219": ("1899", "Joseph Conrad, 1857–1924"),
    "1260": ("1847", "Charlotte Brontë, 1816–1855"),
    "1322": ("1855", "Walt Whitman, 1819–1892"),
    "5200": ("1915", "Franz Kafka, 1883–1924"),
    "2701": ("1851", "Herman Melville, 1819–1891"),
    "12242": ("1890", "Emily Dickinson, 1830–1886 (published after her death)"),
    "1342": ("1813", "Jane Austen, 1775–1817"),
    "1661": ("1892", "Arthur Conan Doyle, 1859–1930"),
    "160": ("1899", "Kate Chopin, 1850–1904"),
    "64317": ("1925", "F. Scott Fitzgerald, 1896–1940"),
    "174": ("1890", "Oscar Wilde, 1854–1900"),
    "1952": ("1892", "Charlotte Perkins Gilman, 1860–1935"),
    "43": ("1886", "Robert Louis Stevenson, 1850–1894"),
    "4300": ("1922", "James Joyce, 1882–1941"),
    "205": ("1854", "Henry David Thoreau, 1817–1862"),
    "2600": ("1869", "Leo Tolstoy, 1828–1910"),
    "768": ("1847", "Emily Brontë, 1818–1848"),
}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def plain(text, words=70):
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    parts = text.split(" ")
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


def year_of(*values):
    for v in values:
        m = re.search(r"(1[0-9]{3}|20[0-2][0-9])", str(v or ""))
        if m:
            return int(m.group(1))
    return None


BOILERPLATE = ("title from", "general information", "unverified", "rights", "copyright", "caption card", "digital id", "transferred", "gift of", "online format", "call number", "reproduction number", "no known restrictions")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data"
    items = json.load(open(os.path.join(path, "items.json"), encoding="utf-8"))["items"]
    out_path = os.path.join(path, "history.json")
    out = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}

    # Project Gutenberg: book dates and author lives
    for it in items:
        if it.get("source") == "Project Gutenberg":
            book = re.search(r"ebooks/(\d+)", it.get("link", ""))
            if book and book.group(1) in BOOKS:
                made, maker = BOOKS[book.group(1)]
                out[it["id"]] = {"made": f"first published {made}" if made[0].isdigit() else f"composed {made}",
                                 "year": year_of(made) or 900, "maker": maker, "src": it.get("link")}

    # Art Institute of Chicago: 100 records per request
    aic = {re.search(r"artworks/(\d+)", it["link"]).group(1): it for it in items
           if it.get("source") == "Art Institute of Chicago" and re.search(r"artworks/(\d+)", it.get("link", ""))}
    ids = [i for i, it in aic.items() if it["id"] not in out]
    fields = "id,title,description,short_description,artist_display,date_display,place_of_origin,style_title,medium_display,classification_title"
    for k in range(0, len(ids), 100):
        batch = ids[k:k + 100]
        try:
            data = get_json(f"https://api.artic.edu/api/v1/artworks?ids={','.join(batch)}&limit=100&fields={fields}")["data"]
        except Exception as e:  # noqa: BLE001
            print("AIC batch failed:", e); continue
        for d in data:
            it = aic.get(str(d["id"]))
            if not it:
                continue
            about = plain(d.get("short_description") or d.get("description"))
            out[it["id"]] = {
                "about": about or None,
                "maker": (d.get("artist_display") or "").replace("\n", ", ") or None,
                "made": d.get("date_display"), "year": year_of(d.get("date_display"), it.get("date")),
                "place": d.get("place_of_origin"), "style": d.get("style_title"), "medium": d.get("medium_display"),
                "src": it.get("link"),
            }
        print(f"AIC {min(k + 100, len(ids))}/{len(ids)}")
        time.sleep(1.2)

    # The Met
    for it in items:
        if it.get("source") != "The Met" or it["id"] in out:
            continue
        m = re.search(r"search/(\d+)", it.get("link", ""))
        if not m:
            continue
        try:
            d = get_json(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{m.group(1)}")
        except Exception as e:  # noqa: BLE001
            print("Met failed:", e); continue
        maker = ", ".join(x for x in [d.get("artistDisplayName"), d.get("artistDisplayBio")] if x)
        out[it["id"]] = {"maker": maker or None, "made": d.get("objectDate"), "year": year_of(d.get("objectDate"), it.get("date")),
                         "place": d.get("country") or d.get("culture") or None, "medium": d.get("medium"), "src": it.get("link")}
        time.sleep(0.6)

    # Library of Congress: one record at a time, politely
    loc = [it for it in items if it.get("source") == "Library of Congress" and it["id"] not in out]
    for n, it in enumerate(loc):
        try:
            d = get_json(it["link"].rstrip("/") + "/?fo=json").get("item", {})
        except Exception as e:  # noqa: BLE001
            print("LoC failed:", it["link"], e); time.sleep(3); continue
        notes = [x for x in (d.get("notes") or []) if isinstance(x, str) and not any(b in x.lower() for b in BOILERPLATE)]
        summary = d.get("summary")
        summary = summary[0] if isinstance(summary, list) and summary else summary
        about = plain(summary or " ".join(notes[:2]))
        made = d.get("date") or ", ".join(d.get("created_published") or [])
        out[it["id"]] = {
            "about": about or None, "made": made or it.get("date"), "year": year_of(d.get("date"), it.get("date")),
            "subjects": [s for s in (d.get("subject_headings") or [])][:4],
            "place": ", ".join((d.get("location") or [])[:2]) or None,
            "medium": ", ".join(d.get("medium") or []) or None, "src": it.get("link"),
        }
        if n % 25 == 0:
            print(f"LoC {n}/{len(loc)}")
            json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
        time.sleep(0.5)

    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    with_about = sum(1 for v in out.values() if v.get("about"))
    with_year = sum(1 for v in out.values() if v.get("year"))
    print(f"Done: {len(out)} records, {with_about} with an archive description, {with_year} with a year -> {out_path}")


if __name__ == "__main__":
    main()
