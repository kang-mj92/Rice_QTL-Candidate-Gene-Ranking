# QTL Candidate Gene Ranking Pipeline

This pipeline prioritizes candidate genes within QTL intervals by combining variant effects, physical proximity to the QTL peak, annotation-based trait relevance, semantic similarity, and PubMed literature evidence.

## Pipeline overview

### 1. Configuration and data containers
Defines project paths, API settings, embedding options, ranking limits, score weights, and structured result objects. All paths are relative to the project root, and NCBI credentials are read from environment variables.

### 2. Shared utilities
Provides chromosome normalization, text cleaning, PMID extraction, score scaling, hashing, and safe Excel sheet-name generation.

### 3. RAP-DB annotation processing
Loads RAP-DB annotations, detects the gene identifier column, combines annotation fields into searchable text, and builds gene-specific symbols, phrases, ortholog-like symbols, and curated PMID references.

### 4. Functional and article-level scoring
Calculates keyword-based gene relevance, MeSH species support, publication-type weights, and article-level trait relevance from keyword, semantic, MeSH, and publication-type components.

### 5. Trait profiles and embeddings
Builds trait-specific keyword profiles, loads or creates RAP-DB document embeddings, generates query embeddings, and derives curated annotation metadata for each gene.

### 6. Trait-level caches
Caches trait keyword profiles, trait embedding vectors, and curated evidence maps so repeated QTLs for the same trait do not repeat expensive calculations.

### 7. Input validation and resource loading
Checks required files and directories, loads the sentence-transformer model, reads RAP-DB annotations, imports variant data, and reads QTL run specifications.

### 8. Variant filtering and initial ranking
Filters variants to each QTL interval, retains fixed differences between the two parental genotype columns, parses ANN annotations, normalizes gene IDs, and calculates variant, distance, functional, and semantic scores.

### 9. PubMed retrieval and scoring
Searches direct rice-gene and ortholog layers, deduplicates retrieved articles, evaluates curated PMID recovery, embeds article text, and assigns article-level trait relevance scores.

### 10. Literature aggregation and final scoring
Aggregates top article scores at the gene level, normalizes literature support within each QTL, and combines all score components into the final candidate-gene priority score.

### 11. Output export
Combines results across QTLs and writes all candidates, top-priority genes, PubMed evidence, and QTL-specific worksheets to Excel and TSV files.

### 12. Pipeline orchestration
Runs the complete workflow for every QTL listed in the input workbook and saves the final outputs.

## Expected project structure

```text
project_root/
├── data/
│   ├── variants.tsv
│   ├── rapdb.tsv
│   └── qtl_runs.xlsx
├── src/
│   ├── config.py
│   ├── filtering.py
│   ├── io_variants.py
│   ├── parse_ann.py
│   ├── pubmed_client.py
│   ├── scoring_variant.py
│   └── trait_profile_mining.py
├── cache/
├── output/
└── run_qtl_candidate_ranking.py
```

## Input files
### Data availability

Input datasets are not included in this repository because of their size and/or licensing restrictions.

Users should prepare the required input files described below before running the pipeline.

### `data/variants.tsv`
Variant table containing chromosome, position, genotype columns for two parents, and ANN annotations.

### `data/rapdb.tsv`
RAP-DB annotation table containing gene identifiers, descriptions, symbols, GO terms, InterPro terms, Oryzabase information, and literature PMIDs.

### `data/qtl_runs.xlsx`
Excel workbook with a `Runs` sheet containing the following columns:

| Column | Description | Example |
|---|---|---|
| `QTL` | QTL identifier | qHD7 |
| `TRAIT` | Trait name | heading date |
| `CHR` | Chromosome | chr07 | 
| `START` | QTL interval start position | 28377055 |
| `END` | QTL interval end position | 29623803 |
| `PEAK` | QTL peak position | 29529536 |

## Environment variables

```bash
export NCBI_EMAIL="your_email@example.org"
export NCBI_API_KEY="your_ncbi_api_key"
```

On Windows PowerShell:

```powershell
$env:NCBI_EMAIL="your_email@example.org"
$env:NCBI_API_KEY="your_ncbi_api_key"
```

The API key is optional. The contact email is recommended by NCBI.

## Installation

```bash
pip install numpy pandas openpyxl sentence-transformers
```

Install any additional dependencies required by the modules under `src/`.

## Running the pipeline

```bash
python run_qtl_candidate_ranking.py
```

## Main outputs

- `output/results.xlsx`
  - `PUBMED_EVIDENCE`: article-level literature evidence
  - `ALL_CANDIDATES`: all ranked genes
  - `TOP_PRIORITY`: top-ranked genes per QTL
  - QTL-specific ranking and evidence sheets
- `output/all_candidates_raw.tsv`: complete gene-ranking table

## Score structure

The initial priority score integrates variant impact, QTL-peak distance, and trait relatedness. PubMed evidence is then aggregated and normalized within each QTL before calculating the final priority score.

All weights can be changed in the `WeightSettings` dataclass.

## Privacy and reproducibility

- Personal file paths are not hard-coded.
- NCBI credentials are loaded from environment variables.
- Cache and output directories are created automatically.
- Curated PMIDs are used only for post hoc retrieval evaluation and are not inserted into PubMed search queries.

## Related publication

Kang, Mj., Ahn, IP., Kim, MH. et al. (2026).
[QTL mapping for heading date and yield-related traits in Korean japonica rice varieties and development of a candidate gene prioritization framework](https://doi.org/10.1007/s11816-026-01097-w).
*Plant Biotechnology Reports*, 20, Article 70.