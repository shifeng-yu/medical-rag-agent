# -*- coding: utf-8 -*-
"""一次性工具：把 data/samples/pubmed/*.md 转换为 ingest_pubmed.py 可解析的
NCBI 风格 JSON（每文件一篇文章 {title, abstract, pmid, pub_date}）。

默认生成目录 data/pubmed/（全部 1709 篇，全量灌库会很久）。
演示/本机验证建议：python scripts/convert_pubmed_samples.py --limit 3
→ 每科室取前 N 篇到 data/pubmed_demo/，并用 .env PUBMED_OFFLINE_DIR 指向它。
"""
import re
import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "data" / "samples" / "pubmed"

PMID_RE = re.compile(r"\*\*PMID:\*\*\s*(\d+)")
YEAR_RE = re.compile(r"\*\*Year:\*\*\s*(\d{4})")
HEAD_RE = re.compile(
    r"^#{1,3}\s+(Background|Methods|Results|Conclusions|Objective|Abstract)\s*$",
    re.IGNORECASE,
)


def convert_one_md(md: Path, out_dir: Path, limit: int = 0) -> int:
    text = md.read_text(encoding="utf-8").replace("\r\n", "\n")
    parts = re.split(r"\n## ", text)
    n = 0
    for p in parts:
        if limit and n >= limit:
            break
        p = p.strip()
        if not p:
            continue
        lines = p.splitlines()
        title = lines[0].strip().lstrip("#").strip()
        if not title or "PubMed Literature" in title:
            continue
        body = "\n".join(lines[1:])
        m_pmid = PMID_RE.search(body)
        if not m_pmid:
            continue
        pmid = m_pmid.group(1)
        m_year = YEAR_RE.search(body)
        year = m_year.group(1) if m_year else ""

        # 重组成 structured abstract
        sections = []
        cur = None
        for line in body.splitlines():
            line = line.strip()
            h = HEAD_RE.match(line)
            if h:
                cur = h.group(1).upper()
                continue
            if line and cur and not line.startswith("**") and not line.startswith("---"):
                sections.append(f"{cur}: {line}")
        abstract = " ".join(sections)
        if len(abstract) < 80:
            abstract = re.sub(r"\s+", " ", body).strip()

        rec = {"title": title, "abstract": abstract, "pmid": pmid, "pub_date": year}
        (out_dir / f"{md.stem}_{pmid}.json").write_text(
            json.dumps(rec, ensure_ascii=False), encoding="utf-8"
        )
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="转换 pubmed 样例 md → 可灌库 json")
    parser.add_argument("--limit", type=int, default=0,
                        help="每科室最多转换 N 篇（0=全部）")
    parser.add_argument("--out", type=str, default="data/pubmed",
                        help="输出目录（默认 data/pubmed）")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for md in sorted(SRC.glob("*.md")):
        n = convert_one_md(md, out_dir, limit=args.limit)
        total += n
        print(f"{md.name}: {n} 篇")
    print(f"转换完成: 共 {total} 篇 → {out_dir}")
    if total == 0:
        print("警告: 0 篇，请检查解析正则")


if __name__ == "__main__":
    main()
