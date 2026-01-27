import pandas as pd
import json
import sys
import os
from pyxlsb import open_workbook as open_xlsb

def read_xlsb(file_path):
    try:
        # Read the file
        df = pd.read_excel(file_path, engine='pyxlsb')
        
        # Find the header row
        header_row_idx = None
        for i, row in df.iterrows():
            # Check if the second column (index 1) contains "Assets"
            if len(row) > 1 and str(row.iloc[1]).strip() == "Assets":
                header_row_idx = i
                break
        
        if header_row_idx is not None:
            # Set the header
            df.columns = df.iloc[header_row_idx]
            df = df.iloc[header_row_idx + 1:]
        
        # We need specific columns. based on the previous inspection:
        # Column 1: Description
        # Column 2: Current
        # Column 3: Previous
        # Column 4: Variance Abs
        # Column 5: Variance %
        # Column 6: Comments
        
        # Rename columns by index to be safe
        new_columns = list(df.columns)
        if len(new_columns) >= 7:
            new_columns[1] = 'description'
            new_columns[2] = 'current'
            new_columns[3] = 'previous'
            new_columns[4] = 'variance_abs'
            new_columns[5] = 'variance_pct'
            new_columns[6] = 'comments'
            df.columns = new_columns
            
            # Select only the columns we need
            df = df[['description', 'current', 'previous', 'variance_abs', 'variance_pct', 'comments']]
            
            # Filter out empty descriptions or sub-headers
            df = df[df['description'].notna()]
            df = df[df['description'] != 0] # Filter out 0s
            
            # Clean numeric columns
            def clean_numeric(val):
                if isinstance(val, (int, float)):
                    return val
                try:
                    return float(str(val).replace(',', '').replace('$', '').replace('%', '').strip())
                except:
                    return 0.0

            df['current'] = df['current'].apply(clean_numeric)
            df['previous'] = df['previous'].apply(clean_numeric)
            df['variance_abs'] = df['variance_abs'].apply(clean_numeric)
            # variance_pct might be a string with %
            df['variance_pct'] = df['variance_pct'].apply(clean_numeric)

            # Convert to records
            records = df.to_dict(orient='records')
            return records
        
        return {"error": "Could not identify columns"}

    except Exception as e:
        return {"error": str(e)}

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No file path provided"}))
        sys.exit(1)

    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(json.dumps({"error": f"File not found: {file_path}"}))
        sys.exit(1)

    result = read_xlsb(file_path)
    print(json.dumps(result, default=str))

if __name__ == "__main__":
    main()
