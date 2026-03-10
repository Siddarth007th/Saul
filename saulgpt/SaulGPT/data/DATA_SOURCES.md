# SaulGPT Legal Data Sources

> Last updated: 2026-03-09
> All raw data stored in `data/raw/` (gitignored)

---

## 1. Indian Penal Code (IPC)

| Field | Value |
|-------|-------|
| **Source** | [civictech-India/Indian-Law-Penal-Code-Json](https://github.com/civictech-India/Indian-Law-Penal-Code-Json) |
| **Format** | JSON (array of objects) |
| **Location** | `data/raw/ipc/ipc.json` |
| **Records** | 575 sections |
| **Chapters** | 23 |
| **File Size** | 320 KB |

### Schema

```json
{
  "chapter": 1,
  "chapter_title": "introduction",
  "Section": 1,
  "section_title": "Title and extent of operation of the Code",
  "section_desc": "This Act shall be called the Indian Penal Code, and shall extend to the whole of India except the State of Jammu and Kashmir."
}
```

### Fields
- `chapter` (int): Chapter number (1-23)
- `chapter_title` (string): Chapter name (lowercase)
- `Section` (int): Section number
- `section_title` (string): Section heading
- `section_desc` (string): Full text of the section

### Notes
- Well-structured, ready for RAG chunking
- Covers all 575 IPC sections across 23 chapters
- Historical code (replaced by BNS in 2024, but still widely referenced)

---

## 2. Constitution of India

### 2a. civictech-India Version (PRIMARY)

| Field | Value |
|-------|-------|
| **Source** | [civictech-India/constitution-of-india](https://github.com/civictech-India/constitution-of-india) |
| **Format** | JSON + CSV + SQLite |
| **Location** | `data/raw/constitution/civictech/` |
| **Records** | 465 articles |
| **File Size** | ~1.9 MB (entire repo) |

#### Schema (JSON)

```json
{
  "article": 0,
  "title": "Preamble",
  "description": "WE, THE PEOPLE OF INDIA, having solemnly resolved to constitute India into a SOVEREIGN SOCIALIST SECULAR DEMOCRATIC REPUBLIC..."
}
```

#### Fields
- `article` (int): Article number (0 = Preamble)
- `title` (string): Article title
- `description` (string): Full article text

#### Available Files
- `constitution_of_india.json` — 465 articles as JSON array
- `Constitution of India.csv` — Same data as CSV (has BOM marker `\ufeff`)
- `COI.db` — SQLite database

### 2b. Yash-Handa Version (SUPPLEMENTARY)

| Field | Value |
|-------|-------|
| **Source** | [Yash-Handa/The_Constitution_Of_India](https://github.com/Yash-Handa/The_Constitution_Of_India) |
| **Format** | JSON (nested structure) |
| **Location** | `data/raw/constitution/yash-handa/COI.json` |
| **Structure** | 2-part nested array: Part 0 (43 articles), Part 1 (3 parts with nested articles) |

#### Schema

```json
// Part 0 entries:
{
  "ArtNo": "0",
  "Name": "PREAMBLE",
  "ArtDesc": "WE, THE PEOPLE OF INDIA..."
}

// Part 1 entries:
{
  "PartNo": "I",
  "Name": "THE UNION AND ITS TERRITORY",
  "Articles": [...]
}
```

#### Notes
- Incomplete coverage (only 46 top-level entries vs civictech's 465)
- More structured (grouped by Parts) but less comprehensive
- **Recommendation**: Use civictech version as primary, Yash-Handa for cross-reference

---

## 3. Bharatiya Nyaya Sanhita (BNS) 2023

| Field | Value |
|-------|-------|
| **Source** | [jaideepkathiresan/SattaThunaivan](https://github.com/jaideepkathiresan/SattaThunaivan) (BNS_Data directory) |
| **Format** | Plain text (bill format) |
| **Location** | `data/raw/bns/` |
| **File Size** | 1.5 MB total |

### Files

| File | Lines | Size | Content |
|------|-------|------|---------|
| `Bharatiya_Nyaya_Sanhita_2023.txt` | 7,734 | 478 KB | BNS (replaces IPC) |
| `BNSS_2023.txt` | 17,954 | 902 KB | Bharatiya Nagarik Suraksha Sanhita (replaces CrPC) |
| `BSA_2023.txt` | 2,745 | 159 KB | Bharatiya Sakshya Adhiniyam (replaces Evidence Act) |

### BNS Structure
- 42 chapter headers found
- 687 numbered clauses
- Plain text format (requires parsing in Task 8)

### Sample (BNS clause)

```
CHAPTER I PRELIMINARY
1. Short title, commencement and application.
2. Definitions.
3. General Explanations and expressions.
```

### Notes
- Raw bill text, not structured JSON — needs parsing during preprocessing (Task 8)
- Includes all three new criminal laws that replaced colonial-era codes in 2024
- No structured JSON dataset found on GitHub/Kaggle for BNS
- This is the most current Indian criminal law

---

## 4. Aalap Instruction Dataset

| Field | Value |
|-------|-------|
| **Source** | [opennyaiorg/aalap_instruction_dataset](https://huggingface.co/datasets/opennyaiorg/aalap_instruction_dataset) |
| **Format** | Parquet (HuggingFace Datasets) |
| **Status** | **GATED** — requires HuggingFace access request |
| **Records** | 21,178 train + 1,094 test = **22,272 total** |
| **Download Size** | ~97 MB |
| **Dataset Size** | ~197 MB (uncompressed) |

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `input_text` | string | Input context/document |
| `system_prompt` | string | System instruction |
| `user_prompt` | string | User query |
| `output_text` | string | Expected response |
| `task` | string | Task category |
| `combined_input_prompt` | string | Merged prompt |

### Access
- Gated dataset (auto-approval)
- Requires HuggingFace account + access request
- License: "other" (check terms before commercial use)
- **Purpose**: Fine-tuning instruction dataset for legal AI

### Fallback Plan
If Aalap access is not granted:
- Task 11 includes manual curation of legal Q&A pairs
- Can use IPC/Constitution as source for generating instruction data
- Can use publicly available legal Q&A from Indian legal forums

---

## Summary

| Source | Records | Format | Status | Use Case |
|--------|---------|--------|--------|----------|
| IPC | 575 sections | JSON | Downloaded | RAG knowledge base |
| Constitution | 465 articles | JSON/CSV | Downloaded | RAG knowledge base |
| BNS 2023 | 687 clauses | Text | Downloaded | RAG knowledge base (needs parsing) |
| BNSS 2023 | ~500+ sections | Text | Downloaded | RAG knowledge base (needs parsing) |
| BSA 2023 | ~170+ sections | Text | Downloaded | RAG knowledge base (needs parsing) |
| Aalap | 22,272 examples | Parquet | Gated (not downloaded) | Fine-tuning |

### Total Disk Usage
- `data/raw/ipc/`: 324 KB
- `data/raw/constitution/`: 1.9 MB
- `data/raw/bns/`: 1.5 MB
- `data/raw/aalap/`: 0 (not yet downloaded)
- **Total**: ~3.7 MB (well under 500 MB limit)

### Next Steps (Task 8)
1. Parse BNS/BNSS/BSA text into structured JSON
2. Normalize IPC and Constitution schemas
3. Create unified document format for RAG pipeline
4. Request Aalap access on HuggingFace if needed for fine-tuning
