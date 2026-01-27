# Sample Data - Trial Balance

This directory contains real-world sample data from Hong Kong operations (November 2025) used for development, testing, and training purposes.

## Directory Structure

```
sample-data/
├── README.md                           # This file
├── HKG_PL review Nov'25.xlsb          # Profit & Loss review workbook
├── HKG_TB review Nov'25.xlsb          # Trial Balance review workbook
└── extracted/                          # Pre-extracted CSV files
    ├── HKG_PL review Nov'25(Checklist).csv
    ├── HKG_PL review Nov'25(MTD-TB).csv
    ├── HKG_PL review Nov'25(Names).csv
    ├── HKG_PL review Nov'25(PL Variance).csv
    ├── HKG_PL review Nov'25(Rates).csv
    ├── HKG_PL review Nov'25(Raw TB Nov'25).csv
    └── HKG_PL review Nov'25(Raw TB oct'25).csv
```

## Files Description

### Excel Binary Files (XLSB)

#### HKG_PL review Nov'25.xlsb
**Purpose:** Profit & Loss review workbook for Hong Kong operations  
**Period:** November 2025  
**Contains:**
- Monthly P&L statements
- Variance analysis (current vs. previous month)
- Comments and explanations
- Review checklist
- Exchange rates and conversion factors

**Key Sheets:**
- MTD-TB: Month-to-Date Trial Balance
- PL Variance: Profit & Loss variance analysis
- Checklist: Review items and sign-offs
- Rates: Exchange rate tables
- Raw Data: Unformatted transaction data

#### HKG_TB review Nov'25.xlsb
**Purpose:** Trial Balance review workbook for Hong Kong operations  
**Period:** November 2025  
**Contains:**
- Complete trial balance for November 2025
- Prior month comparison (October 2025)
- Account-level detail
- Balance sheet and P&L accounts
- Variance calculations

**Use Cases:**
- Development: Test data extraction and processing logic
- Testing: Validate balance calculations and variance analysis
- Training: Real-world examples for user documentation

### Extracted CSV Files

Pre-processed CSV files extracted from the XLSB workbooks, ready for direct import and testing.

#### HKG_PL review Nov'25(Checklist).csv
**Contents:** Review checklist items and completion status  
**Columns:**
- Item description
- Responsibility
- Status
- Due date
- Notes

**Use Case:** Testing workflow approval processes

#### HKG_PL review Nov'25(MTD-TB).csv
**Contents:** Month-to-Date Trial Balance data  
**Columns:**
- Account code
- Account name
- Opening balance
- Debits
- Credits
- Closing balance

**Use Case:** Database seeding, balance calculation testing

#### HKG_PL review Nov'25(Names).csv
**Contents:** Account names and descriptions  
**Columns:**
- Account code
- Account name
- Account type
- Category
- Sub-category

**Use Case:** Master data setup, dropdown lists

#### HKG_PL review Nov'25(PL Variance).csv
**Contents:** Profit & Loss variance analysis  
**Columns:**
- Description (account/line item)
- Current period
- Previous period
- Variance (absolute)
- Variance (percentage)
- Comments

**Use Case:** Variance analysis testing, AI narrative generation

#### HKG_PL review Nov'25(Rates).csv
**Contents:** Exchange rates and conversion factors  
**Columns:**
- Currency pair
- Rate
- Effective date
- Rate type (spot, average, closing)

**Use Case:** Multi-currency calculations, currency conversion testing

#### HKG_PL review Nov'25(Raw TB Nov'25).csv
**Contents:** Raw trial balance data for November 2025  
**Columns:**
- Account code
- Account name
- Debit balance
- Credit balance
- Balance type

**Use Case:** Current period data processing

#### HKG_PL review Nov'25(Raw TB oct'25).csv
**Contents:** Raw trial balance data for October 2025  
**Columns:**
- Account code
- Account name
- Debit balance
- Credit balance
- Balance type

**Use Case:** Prior period comparison, trend analysis

## Data Schema

### Common Data Structure

All variance analysis files follow this JSON schema:

```json
{
  "description": "string",     // Account or line item name
  "current": "number",          // Current period value
  "previous": "number",         // Previous period value
  "variance_abs": "number",     // Absolute variance
  "variance_pct": "number",     // Percentage variance
  "comments": "string"          // Explanation or notes
}
```

## Usage Guidelines

### For Developers

**Backend Development:**
```bash
# Import sample data into development database
cd ../scripts
python extract_xlsb.py ../sample-data/HKG_TB\ review\ Nov\'25.xlsb > tb_data.json

# Use in Zig backend for testing
zig test -- --test-data=../BusDocs/sample-data/extracted/
```

**Frontend Development:**
```javascript
// Load sample data for UI testing
import sampleData from './BusDocs/sample-data/extracted/HKG_PL review Nov\'25(PL Variance).csv';
```

### For Testers

**Unit Testing:**
- Use individual CSV files for specific feature testing
- Variance calculations: Use PL Variance file
- Balance validation: Use Raw TB files
- Workflow testing: Use Checklist file

**Integration Testing:**
- Use complete XLSB files with extraction scripts
- Test end-to-end data flow
- Verify calculations against Excel formulas

**Load Testing:**
- Duplicate and modify CSV files for volume testing
- Use scripts to generate additional test data

### For Business Analysts

**Understanding Data Structure:**
- Open XLSB files in Excel to see actual layouts
- Review CSV files to understand data relationships
- Use as reference for requirement validation

**Training Material:**
- Real-world examples for user training
- Demonstrate variance analysis workflows
- Show month-end close process

## Data Privacy & Security

⚠️ **IMPORTANT SECURITY NOTICE**

This directory contains real financial data from Hong Kong operations:

### Data Classification
- **Confidentiality:** Internal - Restricted
- **Contains:** Actual financial balances and variances
- **Period:** November 2025 (Historical data)

### Security Requirements

**DO:**
- ✅ Use only for authorized development and testing
- ✅ Keep data within the development environment
- ✅ Follow company data handling policies
- ✅ Ensure proper access controls are maintained
- ✅ Delete from local machines when no longer needed

**DO NOT:**
- ❌ Share outside the authorized development team
- ❌ Commit additional real data without proper masking
- ❌ Use in production environments
- ❌ Include in public repositories
- ❌ Share via unsecured channels (email, chat, etc.)
- ❌ Store on personal devices without encryption

### Data Masking (For Additional Data)

If you need to add more sample data:

1. **Anonymize sensitive fields:**
   - Replace actual account names with generic names
   - Modify balances by applying random factors
   - Remove or mask comments containing specific details

2. **Use data masking tools:**
   ```bash
   # Example: Mask financial values
   python mask_data.py input.csv output.csv --mask-amounts --factor=1.5
   ```

3. **Review before committing:**
   - Ensure no customer/vendor names
   - Verify no sensitive comments
   - Check for any identifying information

## Maintenance

### Adding New Sample Data

1. **Place new XLSB files in this directory**
   ```bash
   cp /path/to/new/file.xlsb ./
   ```

2. **Extract to CSV format**
   ```bash
   cd ../scripts
   python extract_xlsb.py ../sample-data/your_file.xlsb > ../sample-data/extracted/your_file.csv
   ```

3. **Update this README**
   - Add file description
   - Document data structure
   - Note any special considerations

4. **Commit with appropriate message**
   ```bash
   git add sample-data/
   git commit -m "Add sample data: [description]"
   ```

### Data Refresh Cycle

Sample data should be refreshed:
- **Quarterly:** Update with recent period data
- **Major releases:** Ensure data reflects current business processes
- **On request:** Business analysts may request specific scenarios

### Quality Checks

Before committing new sample data:
- [ ] Verify data completeness
- [ ] Check for data quality issues
- [ ] Ensure proper masking/anonymization
- [ ] Test with extraction scripts
- [ ] Validate against application requirements
- [ ] Review for sensitive information

## Extraction Instructions

To extract data from XLSB files:

```bash
# Navigate to scripts directory
cd ../scripts

# Extract using pyxlsb (fast, no Excel required)
python extract_xlsb.py ../sample-data/HKG_PL\ review\ Nov\'25.xlsb

# Extract using xlwings (requires Excel, includes formulas)
python extract_live.py ../sample-data/HKG_PL\ review\ Nov\'25.xlsb

# Save to file
python extract_xlsb.py ../sample-data/HKG_TB\ review\ Nov\'25.xlsb > output.json
```

See [Scripts README](../scripts/README.md) for detailed documentation.

## Integration with Application

This sample data is used throughout the Trial Balance application:

### Backend Services
- **Data Import:** Seed development databases
- **Calculation Engine:** Validate balance calculations
- **Workflow Engine:** Test approval processes
- **API Testing:** Provide realistic test payloads

### Frontend Components
- **Dashboard:** Display sample balances
- **Variance Analysis:** Show example calculations
- **Reports:** Generate sample reports
- **Charts:** Populate visualizations

### Testing Frameworks
- **Unit Tests:** Individual component testing
- **Integration Tests:** End-to-end workflows
- **Performance Tests:** Load and stress testing
- **UI Tests:** Automated browser testing

## Related Documentation

- [Parent BusDocs README](../README.md)
- [Extraction Scripts README](../scripts/README.md)
- [Trial Balance Application README](../../README.md)
- [Backend API Documentation](../../backend/docs/API.md)

## Support

For questions about sample data:
- **Data Issues:** Contact data governance team
- **Access Issues:** Contact IT security
- **Technical Issues:** Contact development team
- **Business Questions:** Contact finance operations