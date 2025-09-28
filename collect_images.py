import asyncio
import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import aiohttp
from PIL import Image
from io import BytesIO

try:
    from duckduckgo_search import DDGS
except Exception as e:
    DDGS = None


# Categories and multilingual query seeds
CATEGORY_QUERIES: Dict[str, List[str]] = {
    "clean": [
        "intact mural painting high quality",
        "完好 壁画 高清",
    ],
    "crack": [
        "mural wall painting crack damage",
        "壁画 裂缝 病害",
    ],
    "peel": [
        "mural paint flaking detachment",
        "壁画 起甲 剥落",
    ],
    "disc": [
        "mural discoloration fading",
        "壁画 褪色 变色",
    ],
    "stain": [
        "mural stain grime mold",
        "壁画 污渍 霉斑",
    ],
    "salt": [
        "stone wall efflorescence salt damage",
        "石墙 盐析 白花 风化",
    ],
    "bio": [
        "mural biological growth algae moss",
        "壁画 生物 附着 藻类 苔藓",
    ],
}


@dataclass
class DownloadResult:
    ok: bool
    reason: str
    filepath: Path = None
    meta: dict = None


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


async def fetch_bytes(session: aiohttp.ClientSession, url: str, timeout_s: int = 15) -> Tuple[bool, bytes]:
    try:
        async with session.get(url, timeout=timeout_s) as resp:
            if resp.status != 200:
                return False, b""
            data = await resp.read()
            return True, data
    except Exception:
        return False, b""


def valid_image(bytes_data: bytes, min_w: int = 256, min_h: int = 256, max_ar: float = 5.0) -> Tuple[bool, dict]:
    try:
        img = Image.open(BytesIO(bytes_data))
        img.load()
        w, h = img.size
        if w < min_w or h < min_h:
            return False, {}
        ar = max(w, h) / max(1, min(w, h))
        if ar > max_ar:
            return False, {}
        return True, {"width": w, "height": h, "format": img.format}
    except Exception:
        return False, {}


async def download_one(session: aiohttp.ClientSession, url: str, out_dir: Path, stem: str) -> DownloadResult:
    ok, data = await fetch_bytes(session, url)
    if not ok or not data:
        return DownloadResult(False, "network_failed")
    ok_img, meta = valid_image(data)
    if not ok_img:
        return DownloadResult(False, "invalid_image")
    digest = sha1_bytes(data)
    ext = (meta.get("format") or "jpg").lower()
    if ext == "jpeg":
        ext = "jpg"
    filename = f"{stem}_{digest[:12]}.{ext}"
    path = out_dir / filename
    # Dedup by content hash
    if path.exists():
        return DownloadResult(False, "dup_filename", path, meta)
    # Also keep a simple hash index
    (out_dir / ".hash_index").mkdir(exist_ok=True)
    hash_marker = out_dir / ".hash_index" / f"{digest}.txt"
    if hash_marker.exists():
        return DownloadResult(False, "dup_hash", path, meta)
    path.write_bytes(data)
    hash_marker.write_text(url)
    return DownloadResult(True, "ok", path, {**meta, "sha1": digest, "src": url})


async def search_image_urls(category: str, queries: List[str], max_per_query: int = 50) -> List[dict]:
    if DDGS is None:
        raise RuntimeError("duckduckgo_search 未安装，请先安装：pip install duckduckgo-search")
    urls = []
    with DDGS() as ddgs:
        for q in queries:
            try:
                for r in ddgs.images(q, safesearch="Off", max_results=max_per_query, size=None, color=None, type_image=None):
                    if not isinstance(r, dict):
                        continue
                    url = r.get("image") or r.get("thumbnail")
                    if not url:
                        continue
                    urls.append({
                        "url": url,
                        "title": r.get("title", ""),
                        "source": r.get("source", ""),
                        "query": q,
                        "category": category
                    })
            except Exception:
                continue
    # De-dup by URL
    seen = set()
    uniq = []
    for r in urls:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        uniq.append(r)
    return uniq


async def collect_category(root: Path, category: str, target_count: int = 100) -> List[dict]:
    out_dir = root / "dataset_raw" / category
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = await search_image_urls(category, CATEGORY_QUERIES.get(category, []), max_per_query=max(50, target_count))
    results = []
    timeout = aiohttp.ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(ssl=False, limit=10)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers={"User-Agent": "Mozilla/5.0"}) as session:
        sem = asyncio.Semaphore(10)

        async def worker(item):
            async with sem:
                res = await download_one(session, item["url"], out_dir, category)
                if res.ok:
                    results.append({
                        "filepath": str(res.filepath.relative_to(root)),
                        "category": category,
                        "url": item["url"],
                        "title": item.get("title", ""),
                        "source": item.get("source", ""),
                        "width": res.meta.get("width"),
                        "height": res.meta.get("height"),
                        "sha1": res.meta.get("sha1"),
                        "query": item.get("query", "")
                    })

        await asyncio.gather(*[worker(u) for u in urls])
    return results


async def main(root: str = ".", per_category: int = 100):
    root_path = Path(root).resolve()
    all_rows: List[dict] = []
    categories = list(CATEGORY_QUERIES.keys())
    for cat in categories:
        print(f"[collect] {cat}...")
        rows = await collect_category(root_path, cat, target_count=per_category)
        print(f"[done] {cat}: {len(rows)} saved")
        all_rows.extend(rows)

    # write manifest CSV
    manifest = root_path / "dataset_raw_manifest.csv"
    with manifest.open("w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filepath","category","url","title","source","width","height","sha1","query"
        ])
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    print(f"Manifest written: {manifest}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--per_category", type=int, default=100)
    args = parser.parse_args()
    asyncio.run(main(args.root, args.per_category))

















