import xlwings as xw
import json
import sys
import os

def extract_with_xlwings(file_path):
    app = None
    try:
        # Launch Excel invisibly
        app = xw.App(visible=False)
        book = app.books.open(file_path)
        
        # Look for the sheet containing "Assets" in the first few rows
        target_sheet = None
        for sheet in book.sheets:
            # Check range A1:C20
            try:
                chunk = sheet.range("A1:C20").value
                for row in chunk:
                    if row and "Assets" in [str(c) for c in row if c]:
                        target_sheet = sheet
                        break
            except:
                continue
            if target_sheet:
                break
        
        if not target_sheet:
            return {"error": "Could not find sheet with 'Assets'"}
        
        # Read the used range as value (Excel calculates it)
        # We assume the structure is similar to what we saw before
        data = target_sheet.used_range.value
        
        # Find header row index
        header_idx = -1
        for i, row in enumerate(data):
            if row and len(row) > 1 and str(row[1]).strip() == "Assets":
                header_idx = i
                break
        
        if header_idx == -1:
            return {"error": "Could not find header row in sheet"}
            
        rows = data[header_idx + 1:]
        
        records = []
        for row in rows:
            # row structure matches column indices (0-based)
            # Description is likely col 1 (B)
            # Current is col 2 (C)
            # Previous is col 3 (D)
            # Variance Abs is col 4 (E)
            # Variance Pct is col 5 (F)
            # Comments is col 6 (G)
            
            if not row or len(row) < 7:
                continue
                
            desc = row[1]
            if desc is None or str(desc).strip() == "" or str(desc) == "0.0":
                continue
                
            record = {
                "description": desc,
                "current": row[2] if isinstance(row[2], (int, float)) else 0.0,
                "previous": row[3] if isinstance(row[3], (int, float)) else 0.0,
                "variance_abs": row[4] if isinstance(row[4], (int, float)) else 0.0,
                "variance_pct": row[5] if isinstance(row[5], (int, float)) else 0.0,
                "comments": row[6]
            }
            records.append(record)
            
        book.close()
        return records

    except Exception as e:
        return {"error": str(e)}
    finally:
        if app:
            app.quit()

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No file path provided"}))
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(json.dumps({"error": f"File not found: {file_path}"}))
        sys.exit(1)

    # Need absolute path for xlwings often
    abs_path = os.path.abspath(file_path)
    result = extract_with_xlwings(abs_path)
    print(json.dumps(result, default=str))

if __name__ == "__main__":
    main()
