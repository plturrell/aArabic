# Business Documentation - Trial Balance Process

This directory contains business documentation, requirements, sample data, and extraction utilities for the Trial Balance application.

## Directory Structure

```
BusDocs/
├── README.md                          # This file
├── requirements/                      # Business requirements and cases
│   ├── Business case - Trial Balance (1).doc
│   ├── Business case - BSS Risk Assessment (1).doc
│   └── DOI for Trial Balance Process.docx
├── procedures/                        # Standard operating procedures
│   └── Operation of Month End Close Controls - Trial Balance Review - GLO.pdf
├── sample-data/                       # Sample datasets for testing
│   ├── HKG_PL review Nov'25.xlsb
│   ├── HKG_TB review Nov'25.xlsb
│   └── extracted/                     # Extracted CSV files
│       ├── HKG_PL review Nov'25(Checklist).csv
│       ├── HKG_PL review Nov'25(MTD-TB).csv
│       ├── HKG_PL review Nov'25(Names).csv
│       ├── HKG_PL review Nov'25(PL Variance).csv
│       ├── HKG_PL review Nov'25(Rates).csv
│       ├── HKG_PL review Nov'25(Raw TB Nov'25).csv
│       └── HKG_PL review Nov'25(Raw TB oct'25).csv
└── scripts/                           # Data extraction utilities
    ├── extract_live.py                # Live Excel extraction using xlwings
    └── extract_xlsb.py                # XLSB file extraction using pyxlsb
```

## Contents Overview

### 1. Requirements Documentation (`requirements/`)

Contains the business case documentation and process definitions:

- **Business case - Trial Balance (1).doc**: Core business case for the Trial Balance application
- **Business case - BSS Risk Assessment (1).doc**: Risk assessment documentation for Business Support Services
- **DOI for Trial Balance Process.docx**: Definition of Interface (DOI) document outlining the trial balance process workflow

### 2. Standard Operating Procedures (`procedures/`)

Operational documentation for month-end processes:

- **Operation of Month End Close Controls - Trial Balance Review - GLO.pdf**: Global procedures for month-end close controls and trial balance review processes

### 3. Sample Data (`sample-data/`)

Real-world sample data from Hong Kong operations (November 2025):

#### Excel Binary Files (XLSB)
- `HKG_PL review Nov'25.xlsb`: Profit & Loss review workbook
- `HKG_TB review Nov'25.xlsb`: Trial Balance review workbook

#### Extracted CSV Files (`sample-data/extracted/`)
Pre-extracted CSV datasets from the XLSB files for testing:
- **Checklist**: Review checklist items
- **MTD-TB**: Month-to-Date Trial Balance data
- **Names**: Account names and descriptions
- **PL Variance**: Profit & Loss variance analysis
- **Rates**: Exchange rates and conversion factors
- **Raw TB Nov'25**: Raw trial balance data for November 2025
- **Raw TB oct'25**: Raw trial balance data for October 2025 (comparison period)

### 4. Extraction Scripts (`scripts/`)

Python utilities for extracting data from Excel files:

#### `extract_live.py`
Live Excel extraction using `xlwings` (requires Excel application):
- Opens Excel files invisibly
- Extracts formatted data with formulas calculated
- Outputs JSON format
- **Use case**: When you need live Excel calculation engine

**Usage:**
```bash
python scripts/extract_live.py /path/to/file.xlsb
```

#### `extract_xlsb.py`
Direct XLSB file parsing using `pyxlsb`:
- No Excel application required
- Fast extraction of raw values
- Outputs JSON format
- **Use case**: Automated batch processing, CI/CD pipelines

**Usage:**
```bash
python scripts/extract_xlsb.py /path/to/file.xlsb
```

**Dependencies:**
```bash
pip install xlwings pyxlsb pandas openpyxl
```

## Data Schema

The extracted data follows this structure:

```json
[
  {
    "description": "Account name or line item",
    "current": 1234.56,
    "previous": 1000.00,
    "variance_abs": 234.56,
    "variance_pct": 23.46,
    "comments": "Explanation of variance"
  }
]
```

## Usage Guidelines

### For Developers
1. Use the sample data in `sample-data/` for development and testing
2. Use extraction scripts to process new XLSB files
3. Reference the requirements docs to understand business logic
4. Follow procedures documentation for workflow implementation

### For Business Analysts
1. Review requirements documentation to understand the business case
2. Use procedures documentation for operational guidance
3. Sample data provides real-world examples of data structures

### For Testers
1. Use the extracted CSV files for unit testing
2. Use XLSB files for integration testing with extraction scripts
3. Verify calculations against the sample data

## Data Privacy & Security

⚠️ **Important**: The sample data in this directory contains real financial data from Hong Kong operations. 

- **Do not** commit additional real financial data without proper data masking
- **Do not** share these files outside the authorized development team
- **Do** use this data only for development and testing purposes
- **Do** ensure proper access controls are maintained

## Integration with Application

The Trial Balance application uses these documents and data for:

1. **Backend Development**: Understanding data structures and business rules
2. **Frontend Development**: Sample data for UI component testing
3. **Integration Testing**: End-to-end workflow validation
4. **User Training**: Real-world examples for training materials
5. **API Development**: Schema definition and validation rules

## Maintenance

### Adding New Sample Data
1. Place raw XLSB files in `sample-data/`
2. Run extraction scripts to generate CSV files
3. Move CSV files to `sample-data/extracted/`
4. Update this README with data description

### Adding New Documentation
1. Place business documents in `requirements/`
2. Place operational procedures in `procedures/`
3. Update the relevant section in this README
4. Ensure documents are reviewed for sensitive information

## Related Documentation

- [Main Application README](../README.md)
- [Backend API Documentation](../backend/docs/API.md)
- [Integration Guide](../integrations/docs/API_INTEGRATION.md)
- [Workflow Configuration](../config/workflow/README.md)

## Questions or Issues?

Contact the Trial Balance development team for:
- Data format questions
- Extraction script issues
- Business process clarifications
- Access to additional documentation