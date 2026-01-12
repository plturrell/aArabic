#!/bin/bash
# Download Kaggle Datasets for Arabic Financial Translation Fine-tuning

set -e

echo "📥 Downloading Kaggle Datasets for Arabic Financial Translation"
echo "================================================================"

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "❌ Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Create data directory
DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/kaggle_datasets"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo ""
echo "📁 Saving to: $DATA_DIR"
echo ""

# Dataset 1: SANAD Dataset (Arabic Financial News)
echo "1️⃣  Downloading SANAD Dataset (Arabic Financial News)..."
if [ ! -d "sanad" ]; then
    kaggle datasets download -d haithemhermessi/sanad-dataset
    unzip -q sanad-dataset.zip -d sanad
    rm sanad-dataset.zip
    echo "   ✅ SANAD dataset extracted"
else
    echo "   ⏭️  SANAD dataset already exists"
fi

# Dataset 2: Arabic News Dataset
echo ""
echo "2️⃣  Downloading Arabic News Dataset..."
if [ ! -d "arabic-news" ]; then
    kaggle datasets download -d asmaaabdelwahab/arabic-news-dataset
    unzip -q arabic-news-dataset.zip -d arabic-news
    rm arabic-news-dataset.zip
    echo "   ✅ Arabic News dataset extracted"
else
    echo "   ⏭️  Arabic News dataset already exists"
fi

# Dataset 3: Arabic BERT Corpus
echo ""
echo "3️⃣  Downloading Arabic BERT Corpus..."
if [ ! -d "arabic-bert" ]; then
    kaggle datasets download -d abedkhooli/arabic-bert-corpus
    unzip -q arabic-bert-corpus.zip -d arabic-bert
    rm arabic-bert-corpus.zip
    echo "   ✅ Arabic BERT corpus extracted"
else
    echo "   ⏭️  Arabic BERT corpus already exists"
fi

echo ""
echo "================================================================"
echo "✅ All datasets downloaded!"
echo ""
echo "📊 Dataset Summary:"
echo "   1. SANAD: Financial news (AR→EN)"
echo "   2. Arabic News: General news corpus"
echo "   3. Arabic BERT: Large Arabic text corpus"
echo ""
echo "📁 Location: $DATA_DIR"
echo ""
echo "🚀 Next Steps:"
echo "   1. Run training pipeline:"
echo "      cd src/serviceIntelligence/serviceTranslation"
echo "      python training_pipeline.py --fast --max-pairs 10000"
echo ""
echo "   2. Or run full training:"
echo "      python training_pipeline.py --kaggle-data ../../data/kaggle_datasets"
echo ""
