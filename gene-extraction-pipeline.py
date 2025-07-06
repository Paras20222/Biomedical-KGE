
# 1. Environment & Package Setup

!pip install --quiet biopython pandas nltk tqdm biothings_client mygene transformers torch

import os
import re
import time
import logging
import warnings
from pathlib import Path
from typing import List, Set

import nltk
import pandas as pd
import torch
from Bio import Entrez, Medline
from biothings_client import get_client
from tqdm import tqdm
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForTokenClassification,
)

# Silence noisy logs
logging.getLogger("biothings.client").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# 2. Configuration

BATCH_SIZE = 20
NER_MODEL = "dmis-lab/biobert-base-cased-v1.1"
DEVICE = 0 if torch.cuda.is_available() else -1

TOKENIZER = AutoTokenizer.from_pretrained(NER_MODEL)
MODEL = AutoModelForTokenClassification.from_pretrained(NER_MODEL, num_labels=2)
NER = pipeline("ner", model=MODEL, tokenizer=TOKENIZER,
               aggregation_strategy="simple", device=DEVICE)

GENE_CLIENT = get_client("gene")
HGNC_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/"
    "hgnc_complete_set.txt"
)

# 3. PubMed Retrieval

def build_pubmed_query(disease: str) -> str:
    """Generate a PubMed search query for a given disease term."""
    term_block = f'("{disease}"[Title/Abstract])'
    gene_block = (
        '("Gene"[Title/Abstract] OR "Genes"[Title/Abstract] OR '
        '"Gene Expression"[Title/Abstract] OR "Transcriptome"[Title/Abstract] '
        'OR "Genomics"[Title/Abstract])'
    )
    return f"{term_block} AND {gene_block}"


def fetch_pubmed_ids(query: str, retmax: int = 10_000) -> List[str]:
    """Return a list of PubMed IDs for a given query."""
    with Entrez.esearch(db="pubmed", term=query, retmax=retmax) as handle:
        record = Entrez.read(handle)
    return record["IdList"]


def fetch_medline_records(pmids: List[str]) -> List[dict]:
    """Batch-fetch MEDLINE records given PubMed IDs."""
    with Entrez.efetch(
        db="pubmed", id=",".join(pmids), rettype="medline", retmode="text"
    ) as handle:
        return list(Medline.parse(handle))



# 4. Gene Extraction Helpers

def _split_sentences(text: str) -> List[str]:
    """Sentence-level segmentation."""
    return re.split(r"(?<=[.!?])\s+", text or "")


def _chunk_text(text: str, max_tokens: int = 450) -> List[str]:
    """Chunk long text into overlaps respecting tokenizer limits."""
    tokens = TOKENIZER.encode(text, add_special_tokens=True)
    if len(tokens) <= max_tokens:
        return [text]

    sentences, chunks, current = _split_sentences(text), [], ""
    for sent in sentences:
        candidate = f"{current} {sent}".strip()
        if len(TOKENIZER.encode(candidate, add_special_tokens=True)) > max_tokens:
            if current:
                chunks.append(current)
            current = sent
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _is_valid_gene_symbol(symbol: str) -> bool:
    """Check symbol against MyGene and alias list."""
    try:
        res = GENE_CLIENT.query(
            symbol, species="human", fields="symbol,alias", size=10
        )
        for hit in res.get("hits", []):
            if symbol.upper() == hit.get("symbol", "").upper():
                return True
            if any(symbol.upper() == a.upper() for a in hit.get("alias", [])):
                return True
    except Exception:
        pass
    return False


def _candidate_genes(text: str) -> Set[str]:
    """Extract candidate gene tokens via BioBERT + regex."""
    candidates: Set[str] = set()
    for chunk in _chunk_text(text):
        try:
            entities = NER(chunk)
            for ent in entities:
                if ent["score"] > 0.85:
                    token = ent["word"].strip()
                    if 2 <= len(token) <= 15 and re.match(r"^[A-Za-z0-9-]+$", token):
                        candidates.add(token.upper())
        except Exception:
            continue

    # Regex fallback
    pattern = r"\b[A-Z][A-Za-z0-9-]{1,14}\b"
    candidates |= {m.upper() for m in re.findall(pattern, text)}
    return candidates


def extract_valid_genes(text: str) -> List[str]:
    """Return validated HGNC-approved genes found in text."""
    if not text or pd.isna(text):
        return []
    genes = [g for g in _candidate_genes(text) if _is_valid_gene_symbol(g)]
    return sorted(set(genes))

# 5. Pipeline Execution

def run_pipeline(disease: str, out_dir: str = "outputs") -> Path:
    """End-to-end pipeline: PubMed fetch → gene extraction → CSV."""
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)

    # PubMed retrieval
    query = build_pubmed_query(disease)
    pmids = fetch_pubmed_ids(query)

    records, data = [], []
    for start in tqdm(range(0, len(pmids), BATCH_SIZE), desc="Fetching MEDLINE"):
        batch = pmids[start : start + BATCH_SIZE]
        try:
            records.extend(fetch_medline_records(batch))
        except Exception:
            continue
        time.sleep(0.4)  # courteous delay

    for rec in tqdm(records, desc="Extracting genes"):
        data.append(
            {
                "PMID": rec.get("PMID", ""),
                "Title": rec.get("TI", ""),
                "Abstract": rec.get("AB", ""),
                "Valid_Genes": extract_valid_genes(rec.get("AB", "")),
            }
        )

    df = pd.DataFrame(data)
    csv_path = out_path / f"{disease.replace(' ', '_')}_gene_results.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# 6. Post-processing (HGNC Cross-Reference)

def hgnc_approve(df_path: Path) -> Path:
    """Filter extracted genes to HGNC-approved symbols."""
    df = pd.read_csv(df_path)
    df["Valid_Genes"] = df["Valid_Genes"].apply(eval)

    hgnc_df = pd.read_csv(HGNC_URL, sep="\t")
    approved = set(hgnc_df["symbol"].str.upper())

    df["Valid_Genes"] = df["Valid_Genes"].apply(
        lambda lst: [g for g in lst if g.upper() in approved]
    )

    clean_path = df_path.with_name(df_path.stem + "_HGNC.csv")
    df.to_csv(clean_path, index=False)
    return clean_path



if __name__ == "__main__":
    DISEASE = "High Altitude Hypoxia"  # ← Change freely
    raw_csv = run_pipeline(DISEASE)
    final_csv = hgnc_approve(raw_csv)
    print(f"Pipeline complete. Results → {final_csv}")
