# Data Organization

Centralized data storage for training, benchmarking, and model testing.

## Structure

```
data/
├── training/              # Training datasets
│   ├── kaggle/           # Kaggle datasets
│   ├── huggingface/      # Hugging Face datasets
│   └── custom/           # Custom/proprietary datasets
├── benchmarks/           # Benchmark datasets
│   ├── kaggle/           # Kaggle benchmark datasets
│   ├── huggingface/      # Hugging Face benchmark datasets
│   └── custom/           # Custom benchmarks
└── model-testing/        # Model evaluation datasets
    ├── kaggle/           # Kaggle test datasets
    ├── huggingface/      # Hugging Face test datasets
    └── custom/           # Custom test datasets
```

## Purpose

### Training Data (`data/training/`)
Datasets used for training and fine-tuning models.

**Sources:**
- **Kaggle** - Competition datasets, community datasets
- **Hugging Face** - Datasets from Hugging Face Hub
- **Custom** - Proprietary or self-curated datasets

**Examples:**
```
data/training/
├── kaggle/
│   ├── arabic-nlp-dataset/
│   ├── code-generation-dataset/
│   └── math-problems-dataset/
├── huggingface/
│   ├── wikitext/
│   ├── squad/
│   └── arabic-corpus/
└── custom/
    ├── internal-code-corpus/
    └── proprietary-documents/
```

### Benchmark Data (`data/benchmarks/`)
Datasets for benchmarking and performance evaluation.

**Purpose:**
- Model comparison
- Performance tracking
- Regression testing
- Capability assessment

**Examples:**
```
data/benchmarks/
├── kaggle/
│   ├── glue-benchmark/
│   └── superglue-benchmark/
├── huggingface/
│   ├── mmlu/
│   ├── hellaswag/
│   └── arc/
└── custom/
    ├── arabic-nlp-benchmark/
    └── code-quality-benchmark/
```

### Model Testing Data (`data/model-testing/`)
Datasets specifically for testing trained models.

**Purpose:**
- Validation
- Integration testing
- Quality assurance
- Production readiness

**Examples:**
```
data/model-testing/
├── kaggle/
│   └── test-sets/
├── huggingface/
│   └── evaluation-datasets/
└── custom/
    ├── acceptance-tests/
    └── edge-cases/
```

## Data Management

### DVC Integration

All data directories are managed with DVC for:
- Version control of large datasets
- Efficient storage in SAP Object Store
- Team collaboration
- Reproducibility

### Adding New Datasets

#### From Kaggle

```bash
# 1. Download from Kaggle
kaggle datasets download -d dataset-name -p data/training/kaggle/

# 2. Extract
unzip data/training/kaggle/dataset-name.zip -d data/training/kaggle/dataset-name/

# 3. Add to DVC
dvc add data/training/kaggle/dataset-name/

# 4. Commit DVC file
git add data/training/kaggle/dataset-name.dvc .gitignore
git commit -m "Add dataset-name from Kaggle"

# 5. Push to remote storage
dvc push
```

#### From Hugging Face

```bash
# 1. Download using datasets library
python scripts/data/download_hf_dataset.py dataset-name --output data/training/huggingface/

# 2. Add to DVC
dvc add data/training/huggingface/dataset-name/

# 3. Commit and push
git add data/training/huggingface/dataset-name.dvc .gitignore
git commit -m "Add dataset-name from Hugging Face"
dvc push
```

#### Custom Datasets

```bash
# 1. Place data in appropriate directory
cp -r /path/to/custom-data data/training/custom/my-dataset/

# 2. Add to DVC
dvc add data/training/custom/my-dataset/

# 3. Commit and push
git add data/training/custom/my-dataset.dvc .gitignore
git commit -m "Add custom dataset: my-dataset"
dvc push
```

### Retrieving Datasets

```bash
# Pull all datasets
dvc pull

# Pull specific dataset
dvc pull data/training/kaggle/dataset-name.dvc

# Pull only training data
dvc pull data/training/
```

## Dataset Catalog

### Training Datasets

| Name | Source | Size | Purpose | Status |
|------|--------|------|---------|--------|
| arabic-nlp-corpus | Kaggle | 5GB | Arabic NLP training | Active |
| code-generation | Custom | 10GB | Code generation training | Active |
| wikitext | HuggingFace | 2GB | General language modeling | Active |

### Benchmark Datasets

| Name | Source | Size | Purpose | Status |
|------|--------|------|---------|--------|
| MMLU | HuggingFace | 500MB | Multi-task language understanding | Active |
| HumanEval | Custom | 100MB | Code generation benchmark | Active |
| Arabic-GLUE | Custom | 1GB | Arabic NLP benchmark | Active |

### Testing Datasets

| Name | Source | Size | Purpose | Status |
|------|--------|------|---------|--------|
| validation-set | Custom | 500MB | Model validation | Active |
| edge-cases | Custom | 100MB | Edge case testing | Active |

## Storage Backend

All datasets are stored in **SAP Object Store** (AWS S3):
- **Bucket**: `hcp-055af4b0-2344-40d2-88fe-ddc1c4aad6c5`
- **Prefix**: `dvc-storage/data/`
- **Region**: `us-east-1`

## Best Practices

### 1. Organization
- Use descriptive directory names
- Include metadata files (README, LICENSE)
- Document data provenance
- Track data versions

### 2. DVC Usage
- Always use DVC for datasets > 10MB
- Commit .dvc files to Git
- Push to remote after adding
- Document in this README

### 3. Licensing
- Check dataset licenses
- Include LICENSE files
- Document usage restrictions
- Respect data provider terms

### 4. Data Quality
- Validate data integrity
- Document preprocessing steps
- Include data statistics
- Test before training

### 5. Security
- Don't commit raw data to Git
- Use DVC for all datasets
- Respect privacy regulations
- Document data sensitivity

## Scripts

### Data Download

```bash
# Download Kaggle dataset
./scripts/models/download_kaggle_datasets.sh

# Download Hugging Face dataset
python scripts/data/download_hf_dataset.py <dataset-name>
```

### Data Validation

```bash
# Validate dataset structure
python scripts/data/validate_dataset.py data/training/kaggle/dataset-name/

# Check data statistics
python scripts/data/data_statistics.py data/training/kaggle/dataset-name/
```

### Data Preprocessing

```bash
# Preprocess dataset
python scripts/data/preprocess.py \
  --input data/training/kaggle/dataset-name/ \
  --output data/training/kaggle/dataset-name-processed/
```

## Monitoring

### Storage Usage

```bash
# Check local storage
du -sh data/

# Check remote storage
aws s3 ls s3://hcp-055af4b0-2344-40d2-88fe-dvc1c4aad6c5/dvc-storage/data/ \
  --recursive --human-readable --summarize
```

### DVC Status

```bash
# Check DVC status
dvc status

# Check remote status
dvc status --remote

# List tracked files
dvc list . --dvc-only data/
```

## Troubleshooting

### Dataset Not Found

**Problem**: DVC can't find dataset  
**Solution**:
```bash
dvc pull data/training/kaggle/dataset-name.dvc
```

### Storage Full

**Problem**: Local storage full  
**Solution**:
```bash
# Remove local cache but keep remote
dvc gc --workspace --cloud
```

### Corrupted Dataset

**Problem**: Dataset checksum mismatch  
**Solution**:
```bash
# Re-download from remote
dvc pull --force data/training/kaggle/dataset-name.dvc
```

## Related Documentation

- [DVC Setup Guide](../docs/02-setup/DVC_SAP_S3_SETUP.md)
- [Model Training Guide](../docs/05-development/)
- [Benchmarking Guide](../docs/09-reference/)

---

**Storage**: SAP Object Store (AWS S3)  
**Version Control**: DVC  
**Last Updated**: January 23, 2026
