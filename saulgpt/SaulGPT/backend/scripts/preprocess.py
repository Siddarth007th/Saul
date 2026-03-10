"""
Legal data preprocessing pipeline.

Reads IPC JSON, Constitution JSON, and BNS raw text files.
Normalizes into a uniform schema, applies hierarchical chunking,
and outputs data/processed/chunks.jsonl.
"""

import json
import re
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paths relative to this script: backend/scripts/preprocess.py
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # munna-work/
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Text splitter for chunking
splitter = RecursiveCharacterTextSplitter(
    separators=["\nSection ", "\nArticle ", "\n\n", "\n", " "],
    chunk_size=450,
    chunk_overlap=50,
    length_function=len,
)


# ── IPC Parser ───────────────────────────────────────────────────────────
def parse_ipc(path: str) -> list[dict]:
    """Parse IPC JSON: {chapter, chapter_title, Section, section_title, section_desc}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for section in data:
        content = (
            f"Section {section['Section']}. {section['section_title']}\n"
            f"{section['section_desc']}"
        )
        metadata = {
            "section": str(section["Section"]),
            "act": "IPC",
            "title": section["section_title"],
            "chapter": section["chapter_title"],
            "jurisdiction": "India",
            "type": "statute",
            "source": "civictech_github",
        }
        docs.append({"content": content, "metadata": metadata})
    return docs


# ── Constitution Parser ─────────────────────────────────────────────────
def parse_constitution(path: str) -> list[dict]:
    """Parse Constitution JSON: {article, title, description}."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    # Strip BOM marker if present
    raw = raw.lstrip("\ufeff")
    data = json.loads(raw)

    docs = []
    for article in data:
        art_num = article["article"]
        title = article["title"]
        desc = article["description"]
        content = f"Article {art_num}. {title}\n{desc}"
        metadata = {
            "section": str(art_num),
            "act": "Constitution",
            "title": title,
            "jurisdiction": "India",
            "type": "statute",
            "source": "civictech_github",
        }
        docs.append({"content": content, "metadata": metadata})
    return docs


# ── BNS Parser (raw text) ───────────────────────────────────────────────
def _read_raw_text(path: str) -> str:
    """Read raw text file with encoding fallback."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, encoding=enc, errors="strict") as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Last resort: lossy utf-8
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _clean_bns_text(text: str) -> str:
    """Clean BNS raw text: remove marginal line numbers, excess whitespace."""
    # Remove standalone marginal line numbers (e.g., "5\n", "10\n", "15\n")
    text = re.sub(r"(?m)^\s*\d{1,2}\s*$", "", text)
    # Remove inline marginal line numbers at start/end of lines (e.g., "5\t" prefix or "\t5" suffix)
    text = re.sub(r"\t\d{1,2}\s", " ", text)
    text = re.sub(r"\s\d{1,2}\t", " ", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_body_start(text: str) -> int:
    """Find where the actual legislative text begins (after 'BE it enacted')."""
    match = re.search(r"BE it enacted by Parliament", text)
    if match:
        return match.start()
    # Fallback: try to find CHAPTER I after preliminary matter
    match = re.search(r"\n\s*CHAPTER\s+I\s", text)
    if match:
        return match.start()
    return 0


def _detect_act_name(filename: str) -> str:
    """Map filename to act name."""
    fname = filename.lower()
    if "nyaya" in fname or "bns_" in fname or fname.startswith("bharatiya_nyaya"):
        return "BNS"
    elif "bnss" in fname or "nagarik" in fname:
        return "BNSS"
    elif "bsa" in fname or "sakshya" in fname:
        return "BSA"
    return "BNS"


def parse_bns_file(path: str) -> list[dict]:
    """Parse a single BNS/BNSS/BSA raw text file into sections."""
    filepath = Path(path)
    raw = _read_raw_text(path)
    act_name = _detect_act_name(filepath.name)

    # Find body start
    body_start = _find_body_start(raw)
    body = raw[body_start:]
    body = _clean_bns_text(body)

    # Split on section boundaries: lines starting with a section number followed by a period
    # Pattern: start of line, optional whitespace, digit(s), period, space
    # e.g., "1. (1) This Act may be called..."  or  "2. In this Sanhita..."
    section_pattern = re.compile(r"(?m)(?=^\s*(\d+)\.\s)", re.MULTILINE)
    splits = list(section_pattern.finditer(body))

    docs = []
    for i, match in enumerate(splits):
        start = match.start()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(body)
        section_text = body[start:end].strip()

        section_num = match.group(1)

        # Extract title from first line
        first_line_match = re.match(r"\d+\.\s*(.*)", section_text)
        first_line = first_line_match.group(1) if first_line_match else ""
        # Try to extract a title: text before first parenthetical or newline
        title_match = re.match(r"([^(\n]+?)(?:\s*\(|$|\n)", first_line)
        title = (
            title_match.group(1).strip().rstrip(".")
            if title_match
            else f"Section {section_num}"
        )

        # Cap title length
        if len(title) > 120:
            title = title[:117] + "..."

        content = f"Section {section_num}. {section_text}"
        metadata = {
            "section": section_num,
            "act": act_name,
            "title": title,
            "jurisdiction": "India",
            "type": "statute",
            "source": "satta_github",
        }
        docs.append({"content": content, "metadata": metadata})

    return docs


def parse_bns(directory: str) -> list[dict]:
    """Parse all BNS raw text files in directory."""
    bns_dir = Path(directory)
    docs = []
    for txt_file in sorted(bns_dir.glob("*.txt")):
        file_docs = parse_bns_file(str(txt_file))
        print(f"  {txt_file.name}: {len(file_docs)} sections parsed")
        docs.extend(file_docs)
    return docs


# ── Chunking ─────────────────────────────────────────────────────────────
def chunk_documents(docs: list[dict]) -> list[dict]:
    """Apply recursive character text splitting to documents."""
    chunks = []
    for doc in docs:
        content = doc["content"]
        metadata = doc["metadata"]

        if len(content) <= 450:
            chunks.append({"content": content, "metadata": metadata})
        else:
            split_texts = splitter.split_text(content)
            for idx, text in enumerate(split_texts):
                chunk_meta = {**metadata, "chunk_index": idx}
                chunks.append({"content": text, "metadata": chunk_meta})
    return chunks


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Legal Data Preprocessing Pipeline")
    print("=" * 60)

    # Parse IPC
    ipc_path = DATA_RAW / "ipc" / "ipc.json"
    print(f"\n[1/3] Parsing IPC from {ipc_path}")
    ipc_docs = parse_ipc(str(ipc_path))
    print(f"  IPC sections: {len(ipc_docs)}")

    # Parse Constitution
    constitution_path = (
        DATA_RAW / "constitution" / "civictech" / "constitution_of_india.json"
    )
    print(f"\n[2/3] Parsing Constitution from {constitution_path}")
    constitution_docs = parse_constitution(str(constitution_path))
    print(f"  Constitution articles: {len(constitution_docs)}")

    # Parse BNS
    bns_dir = DATA_RAW / "bns"
    print(f"\n[3/3] Parsing BNS acts from {bns_dir}")
    bns_docs = parse_bns(str(bns_dir))
    print(f"  Total BNS sections: {len(bns_docs)}")

    # Combine all documents
    all_docs = ipc_docs + constitution_docs + bns_docs
    print(f"\nTotal raw documents: {len(all_docs)}")

    # Chunk
    print("\nChunking documents...")
    chunks = chunk_documents(all_docs)
    print(f"Total chunks after splitting: {len(chunks)}")

    # Write JSONL
    output_path = DATA_PROCESSED / "chunks.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"\nWritten to: {output_path}")

    # Stats
    print("\n" + "=" * 60)
    print("STATS")
    print("=" * 60)

    act_counts: dict[str, int] = {}
    lengths: list[int] = []
    for chunk in chunks:
        act = chunk["metadata"]["act"]
        act_counts[act] = act_counts.get(act, 0) + 1
        lengths.append(len(chunk["content"]))

    print(f"Total chunks: {len(chunks)}")
    for act, count in sorted(act_counts.items()):
        print(f"  {act}: {count} chunks")
    print(f"Avg chunk length: {sum(lengths) / len(lengths):.0f} chars")
    print(f"Max chunk length: {max(lengths)} chars")
    print(f"Min chunk length: {min(lengths)} chars")
    over_500 = sum(1 for l in lengths if l > 500)
    print(f"Chunks > 500 chars: {over_500}")

    if over_500 > 0:
        print(f"\nWARNING: {over_500} chunks exceed 500 characters!")
        sys.exit(1)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
