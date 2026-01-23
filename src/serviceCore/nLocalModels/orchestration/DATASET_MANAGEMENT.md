# Dataset Management for nLocalModels Agent Categories

## Overview

This document describes the dataset loading and maintenance system for benchmark datasets aligned with nLocalModels agent categories. The system provides automated dataset downloading, versioning, validation, and maintenance capabilities.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Dataset Management                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │   Registry     │  │   Loader       │  │  Validator   │  │
│  │  (Metadata)    │──│  (Download)    │──│  (Integrity) │  │
│  └────────────────┘  └────────────────┘  └──────────────┘  │
│           │                   │                   │          │
│           └───────────────────┼───────────────────┘          │
│                               ↓                              │
│                    ┌──────────────────┐                      │
│                    │  Local Cache     │                      │
│                    │  data/benchmarks │                      │
│                    └──────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

## Agent Categories & Datasets

Based on `orchestration/catalog/task_categories.json`, we support 9 agent categories with corresponding benchmark datasets:

### 1. Math (Mathematical Reasoning)
- **GSM8K**: Grade School Math 8K - 8,792 grade school math problems
- **MATH**: Competition-level mathematics with step-by-step solutions

### 2. Code (Code Generation & Understanding)
- **HumanEval**: 164 hand-written Python programming problems
- **MBPP**: 974 Mostly Basic Python Problems

### 3. Reasoning (Complex Reasoning)
- **ARC-Challenge**: 2,590 grade-school science questions
- **HellaSwag**: 60,000 commonsense inference problems
- **MMLU**: 15,858 questions across 57 subjects
- **Winogrande**: 43,972 Winograd schema challenges

### 4. Summarization (Text Summarization)
- **SummScreen**: 4,348 TV show transcript summaries
- **GovReport**: 19,402 government report summaries

### 5. Time Series (Time Series Analysis)
- **M4**: 100,000 time series from forecasting competition
- **Monash**: 30+ datasets from time series archive

### 6. Relational (Relational Data & SQL)
- **Spider**: 10,181 text-to-SQL examples
- **WikiSQL**: 80,654 natural language to SQL examples

### 7. Vector Search (Vector Search & Embeddings)
- **MSMARCO**: 8.8M passage ranking examples
- **MTEB**: 56 datasets across 8 embedding tasks

### 8. OCR Extraction (OCR & Document Extraction)
- **DocVQA**: 50,000 document visual question answering
- **ChartQA**: 21,765 chart question answering

### 9. Graph (Graph Database Queries)
- **GraphQA**: Graph question answering
- **KGQA**: Knowledge graph question answering

## Components

### 1. Dataset Registry (`data/benchmarks/dataset_registry.json`)

Central registry containing metadata for all benchmark datasets:

```json
{
  "version": "1.0.0",
  "datasets": {
    "math": [
      {
        "id": "gsm8k",
        "name": "GSM8K",
        "category": "math",
        "benchmark": "gsm8k",
        "source": "huggingface",
        "source_path": "gsm8k",
        "splits": ["train", "test"],
        "metrics": ["accuracy"]
      }
    ]
  }
}
```

### 2. Dataset Loader (`dataset_loader.zig`)

CLI tool for dataset management:

**Features:**
- Download datasets from HuggingFace, Kaggle, custom sources
- Local caching with versioning
- Integrity validation (checksums)
- Metadata tracking
- Category-based organization

**Commands:**
```bash
# List all datasets
zig build run-dataset-loader -- list

# List datasets for specific category
zig build run-dataset-loader -- list math

# Download HuggingFace dataset
zig build run-dataset-loader -- download-hf gsm8k math gsm8k train

# Validate dataset integrity
zig build run-dataset-loader -- validate hf_gsm8k_train
```

### 3. Dataset Catalog (`data/benchmarks/dataset_catalog.json`)

Auto-generated catalog of downloaded datasets with metadata:
- Download timestamp
- File sizes
- Checksums
- Sample counts
- Local paths

## Installation & Setup

### Prerequisites

```bash
# HuggingFace CLI (for downloading HF datasets)
pip install huggingface_hub

# Kaggle API (for Kaggle datasets)
pip install kaggle
# Configure: ~/.kaggle/kaggle.json with API credentials

# Zig compiler (for building tools)
# Already available in project
```

### Build Dataset Loader

```bash
cd src/serviceCore/nLocalModels/orchestration
zig build
```

This creates the `dataset_loader` executable in `zig-out/bin/`.

## Usage Guide

### Downloading Datasets

#### Example 1: Download GSM8K for Math Category

```bash
# Download training split
zig build run-dataset-loader -- download-hf gsm8k math gsm8k train

# Download test split
zig build run-dataset-loader -- download-hf gsm8k math gsm8k test
```

#### Example 2: Download HumanEval for Code Category

```bash
zig build run-dataset-loader -- download-hf openai_humaneval code humaneval test
```

#### Example 3: Download MMLU for Reasoning Category

```bash
zig build run-dataset-loader -- download-hf cais/mmlu reasoning mmlu test
```

### Listing Datasets

```bash
# List all downloaded datasets
zig build run-dataset-loader -- list

# List datasets for specific category
zig build run-dataset-loader -- list code
zig build run-dataset-loader -- list reasoning
```

### Validating Datasets

```bash
# Validate specific dataset
zig build run-dataset-loader -- validate hf_gsm8k_train

# Validation checks:
# - Path exists
# - Checksum matches
# - File integrity
```

## Dataset Storage Structure

```
data/benchmarks/
├── dataset_registry.json          # Master registry
├── dataset_catalog.json           # Downloaded datasets catalog
├── huggingface/                   # HuggingFace datasets
│   ├── gsm8k/
│   │   ├── train/
│   │   └── test/
│   ├── openai_humaneval/
│   │   └── test/
│   └── cais__mmlu/
│       ├── test/
│       └── validation/
├── kaggle/                        # Kaggle datasets
│   └── m4-forecasting-competition/
└── custom/                        # Custom datasets
    └── monash/
```

## Maintenance

### Regular Tasks

1. **Weekly Validation**
   ```bash
   # Validate all datasets
   for dataset in $(cat dataset_catalog.json | jq -r '.datasets | keys[]'); do
       zig build run-dataset-loader -- validate "$dataset"
   done
   ```

2. **Update Registry**
   - Monitor for new benchmark releases
   - Update `dataset_registry.json` with new entries
   - Document version changes

3. **Cleanup Old Versions**
   ```bash
   # Remove old dataset versions (manual process)
   # Keep latest version + 1 previous version
   ```

### Backup Strategy

**Recommended: DVC + S3**

```bash
# Initialize DVC tracking
cd data/benchmarks
dvc add huggingface/
dvc add kaggle/
dvc add custom/

# Push to S3
dvc remote add -d storage s3://benchmarks-backup/
dvc push
```

## Integration with Benchmark Validator

The dataset loader integrates with the existing `benchmark_validator.zig`:

```bash
# 1. Download dataset
zig build run-dataset-loader -- download-hf gsm8k math gsm8k test

# 2. Run benchmarks on model
# (Your benchmark execution process)

# 3. Validate results
zig build run-validator -- shared/MODEL_REGISTRY.json
```

## Configuration

### Environment Variables

```bash
# HuggingFace token (for private datasets)
export HF_TOKEN="hf_xxxxx"

# Kaggle credentials
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# Cache directory (optional, defaults to data/benchmarks)
export DATASET_CACHE_DIR="data/benchmarks"
```

### Dataset Registry Customization

Add custom datasets to `dataset_registry.json`:

```json
{
  "datasets": {
    "custom_category": [
      {
        "id": "my_dataset",
        "name": "My Custom Dataset",
        "category": "custom_category",
        "benchmark": "my_benchmark",
        "source": "custom",
        "source_path": "https://example.com/dataset.zip",
        "splits": ["train", "test"],
        "format": "json",
        "download_url": "https://example.com/dataset.zip"
      }
    ]
  }
}
```

## Troubleshooting

### Download Failures

**Issue**: HuggingFace download fails
```bash
# Check HuggingFace CLI
huggingface-cli whoami

# Authenticate if needed
huggingface-cli login
```

**Issue**: Kaggle download fails
```bash
# Check Kaggle API setup
cat ~/.kaggle/kaggle.json

# Test connection
kaggle datasets list
```

### Storage Issues

**Issue**: Insufficient disk space
```bash
# Check available space
df -h data/benchmarks

# Remove old versions
rm -rf data/benchmarks/huggingface/*/old_version/
```

### Validation Failures

**Issue**: Checksum mismatch
```bash
# Re-download dataset
rm -rf data/benchmarks/huggingface/gsm8k/train
zig build run-dataset-loader -- download-hf gsm8k math gsm8k train
```

## Performance Considerations

### Download Times

| Dataset | Size | Download Time (100 Mbps) |
|---------|------|--------------------------|
| GSM8K | 3.5 MB | < 1 second |
| HumanEval | 0.5 MB | < 1 second |
| MMLU | 166 MB | ~15 seconds |
| MSMARCO | 3.5 GB | ~5 minutes |
| DocVQA | 2.5 GB | ~4 minutes |

### Storage Requirements

**Minimal Setup** (Small datasets only):
- Math: 55 MB
- Code: 2.5 MB
- Total: ~60 MB

**Standard Setup** (Most categories):
- Math + Code + Reasoning: ~240 MB
- Total: ~300 MB

**Complete Setup** (All datasets):
- All categories: ~15 GB
- Recommended: 20 GB available

## Best Practices

1. **Start Small**: Download test splits first to verify setup
2. **Validate Regularly**: Run validation weekly
3. **Version Control**: Use DVC for dataset versioning
4. **Backup**: Push to S3 regularly
5. **Document Changes**: Update registry when adding new datasets
6. **Monitor Usage**: Track which datasets are actively used
7. **Cleanup**: Remove unused datasets to save space

## Future Enhancements

- [ ] Automatic dataset updates
- [ ] Delta downloads (only download changes)
- [ ] Parallel downloads
- [ ] Progress bars for large downloads
- [ ] Dataset preprocessing pipeline
- [ ] Automatic format conversion
- [ ] Dataset statistics generation
- [ ] Integration with model training pipeline

## References

- **Task Categories**: `src/serviceCore/nLocalModels/orchestration/catalog/task_categories.json`
- **Benchmark Validator**: `src/serviceCore/nLocalModels/orchestration/benchmark_validator.zig`
- **HuggingFace Datasets**: https://huggingface.co/datasets
- **Kaggle Datasets**: https://www.kaggle.com/datasets
- **DVC Documentation**: https://dvc.org/doc

## Support

For issues or questions:
1. Check this documentation
2. Review dataset_registry.json for dataset specifications
3. Check logs in dataset_loader output
4. Validate dataset integrity
5. Report issues with specific dataset ID and error message
