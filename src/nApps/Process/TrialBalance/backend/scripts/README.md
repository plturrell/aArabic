# Data Extraction Scripts

This directory contains Python utilities for extracting data from Excel Binary (XLSB) files used in the Trial Balance process.

## Scripts

### extract_live.py
Live Excel extraction using `xlwings` - requires Excel application to be installed.

**Features:**
- Opens Excel files invisibly using the Excel application
- Extracts formatted data with formulas calculated in real-time
- Handles complex Excel features (pivot tables, formulas, etc.)
- Outputs JSON format

**Usage:**
```bash
python extract_live.py /path/to/file.xlsb
```

**Pros:**
- Full Excel calculation engine support
- Accurate formula evaluation
- Handles all Excel features

**Cons:**
- Requires Microsoft Excel to be installed
- Slower performance
- Platform-dependent (macOS/Windows)

---

### extract_xlsb.py
Direct XLSB file parsing using `pyxlsb` - no Excel application required.

**Features:**
- Direct binary file parsing
- Fast extraction of raw values
- Cross-platform compatible
- Outputs JSON format

**Usage:**
```bash
python extract_xlsb.py /path/to/file.xlsb
```

**Pros:**
- No Excel installation required
- Fast performance
- Ideal for automation and CI/CD
- Cross-platform

**Cons:**
- Extracts only raw values (no formula calculation)
- May not handle all Excel features

## Installation

Install required dependencies:

```bash
pip install xlwings pyxlsb pandas openpyxl
```

For xlwings (macOS/Windows only):
```bash
# macOS
pip install xlwings

# Windows
pip install xlwings
```

## Output Format

Both scripts output JSON in the following format:

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

In case of errors:
```json
{
  "error": "Error message description"
}
```

## Which Script to Use?

**Use `extract_live.py` when:**
- You need accurate formula calculations
- Working with complex Excel features
- Developing/debugging locally with Excel installed
- Data integrity is critical

**Use `extract_xlsb.py` when:**
- Running in automated environments (CI/CD)
- Excel is not available
- You need fast batch processing
- Working with raw data values

## Integration with Trial Balance App

These scripts are used by the Trial Balance application backend to:

1. Import historical trial balance data
2. Process monthly review files
3. Extract variance analysis data
4. Populate the database with sample/test data

## Error Handling

Both scripts handle common errors:
- File not found
- Invalid file format
- Missing required columns
- Data parsing errors

All errors are returned in JSON format with an `error` key.

## Development

To extend or modify these scripts:

1. Both scripts follow a similar structure with a main extraction function
2. The data schema is defined by the column mapping in each script
3. Add additional fields by updating the record structure
4. Test with the sample data in `../sample-data/`

## Troubleshooting

**xlwings issues:**
- Ensure Excel is installed and licensed
- On macOS, you may need to grant permissions to Python
- Try running with `visible=True` for debugging

**pyxlsb issues:**
- Some Excel features may not be supported
- Verify the XLSB file is not corrupted
- Check that all required columns exist

## Examples

Extract data from a sample file:
```bash
# Using live extraction
python extract_live.py ../sample-data/HKG_PL\ review\ Nov\'25.xlsb

# Using direct extraction
python extract_xlsb.py ../sample-data/HKG_PL\ review\ Nov\'25.xlsb
```

Save output to file:
```bash
python extract_xlsb.py ../sample-data/HKG_TB\ review\ Nov\'25.xlsb > output.json
```

Process multiple files:
```bash
for file in ../sample-data/*.xlsb; do
    echo "Processing $file"
    python extract_xlsb.py "$file" > "${file%.xlsb}.json"
done
```

## Related Documentation

- [Parent BusDocs README](../README.md)
- [Sample Data Documentation](../sample-data/README.md)
- [Trial Balance Application README](../../README.md)