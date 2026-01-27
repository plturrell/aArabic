#!/usr/bin/env python3
"""
Load HKG sample data into SQLite development database
Converts CSV files to SQLite INSERT statements
"""

import csv
import sqlite3
import sys
import os
from pathlib import Path
from datetime import datetime
import re

def clean_numeric(value):
    """Clean and convert numeric values"""
    if not value or value == '':
        return 0.0
    # Remove currency symbols, commas, parentheses
    cleaned = re.sub(r'[\$,\(\)]', '', str(value))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def load_raw_tb_data(cursor, csv_file, company_code, period):
    """Load raw trial balance data from CSV"""
    print(f"Loading {csv_file}...")
    
    # Try multiple encodings
    for encoding in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
        try:
            with open(csv_file, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                break
        except UnicodeDecodeError:
            continue
    else:
        print(f"  ⚠ Could not decode {csv_file} with any known encoding, skipping...")
        return
    
    print(f"  Found {len(rows)} rows")
    
    for i, row in enumerate(rows):
        # Skip header or empty rows
        if not row or len(row) < 2:
            continue
        
        # Extract account information from first column
        description = list(row.values())[0] if row else ''
        if not description or description.strip() == '':
            continue
        
        # Try to parse account number from description
        account_match = re.match(r'^(\d+)', description.strip())
        account_num = account_match.group(1) if account_match else f'ACC{i:06d}'
        
        # Extract amounts from subsequent columns
        values = list(row.values())[1:]
        current_amount = clean_numeric(values[0]) if len(values) > 0 else 0.0
        previous_amount = clean_numeric(values[1]) if len(values) > 1 else 0.0
        
        if abs(current_amount) < 0.01:
            continue  # Skip zero balances
        
        # Determine debit/credit indicator
        drcrk = 'S' if current_amount >= 0 else 'H'
        amount = abs(current_amount)
        
        entry_id = f'{company_code}_{period}_{i:06d}'
        
        cursor.execute('''
            INSERT OR IGNORE INTO TB_JOURNAL_ENTRIES 
            (entry_id, mandt, rldnr, rbukrs, gjahr, belnr, buzei, budat, racct, 
             drcrk, hsl, rtcur, rwcur, poper, validated, sgtxt, ifrs_schedule, account_type)
            VALUES (?, '100', '0L', ?, '2025', ?, '001', '2025-11-30', ?, 
                    ?, ?, 'HKD', 'HKD', ?, 1, ?, '1A', 'Asset')
        ''', (
            entry_id,
            company_code,
            f'DOC{i:06d}',
            account_num,
            drcrk,
            amount,
            period,
            description[:50] if len(description) > 50 else description
        ))
    
    print(f"  Loaded {len(rows)} entries")

def load_gl_accounts(cursor, csv_file):
    """Load GL account master data"""
    print(f"Loading GL accounts from {csv_file}...")
    
    # Map of common account patterns to IFRS schedules
    ifrs_mapping = {
        'cash': ('1A', 'Cash & Central Bank', 'Asset'),
        'bank': ('1CA', 'Loans & Advances to Banks', 'Asset'),
        'loan': ('1DA', 'Loans & Advances to Customers', 'Asset'),
        'debt': ('1EA', 'Debt Securities', 'Asset'),
        'equity': ('1FA', 'Equity Shares', 'Asset'),
        'deposit': ('2U', 'Deposits from banks', 'Liability'),
        'customer': ('2V', 'Customer accounts', 'Liability'),
        'interest income': ('3D', 'Interest Income', 'Income'),
        'interest expense': ('3E', 'Interest Expense', 'Expense'),
        'fee': ('3B', 'Fee Income', 'Income'),
        'operating': ('3L', 'Operating Costs', 'Expense'),
    }
    
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            description = list(row.values())[0] if row else ''
            if not description or description.strip() == '':
                continue
            
            # Try to extract account number
            account_match = re.match(r'^(\d+)', description.strip())
            account_num = account_match.group(1) if account_match else f'{100000 + i}'
            
            # Determine IFRS classification based on description
            desc_lower = description.lower()
            ifrs_schedule, ifrs_category, account_type = ('1A', 'Other Assets', 'Asset')
            
            for keyword, mapping in ifrs_mapping.items():
                if keyword in desc_lower:
                    ifrs_schedule, ifrs_category, account_type = mapping
                    break
            
            account_id = f'ACC{i:05d}'
            
            cursor.execute('''
                INSERT OR IGNORE INTO TB_GL_ACCOUNTS 
                (account_id, mandt, saknr, ktopl, txt50, ifrs_schedule, ifrs_category, account_type)
                VALUES (?, '100', ?, 'IFRS', ?, ?, ?, ?)
            ''', (
                account_id,
                account_num,
                description[:50] if len(description) > 50 else description,
                ifrs_schedule,
                ifrs_category,
                account_type
            ))

def load_exchange_rates(cursor):
    """Load standard exchange rates"""
    print("Loading exchange rates...")
    
    rates = [
        ('EUR', 'USD', 1.0845),
        ('GBP', 'USD', 1.2650),
        ('SGD', 'USD', 0.7420),
        ('HKD', 'USD', 0.1283),
        ('JPY', 'USD', 0.0068),
        ('CNY', 'USD', 0.1380),
    ]
    
    for fcurr, tcurr, rate in rates:
        cursor.execute('''
            INSERT OR IGNORE INTO TB_EXCHANGE_RATES 
            (rate_id, mandt, kurst, fcurr, tcurr, gdatu, ukurs, rate_type_desc)
            VALUES (?, '100', 'M', ?, ?, '2025-11-30', ?, 'Month-End Rate')
        ''', (f'FX_{fcurr}_{tcurr}', fcurr, tcurr, rate))
    
    print(f"  Loaded {len(rates)} exchange rates")

def main():
    # Paths
    script_dir = Path(__file__).parent
    sample_data_dir = script_dir.parent / 'sample-data' / 'extracted'
    db_path = script_dir.parent / 'schema' / 'sqlite' / 'trial_balance_dev.db'
    
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        print("Please run init_dev_db.sh first")
        sys.exit(1)
    
    # Connect to database
    print(f"Connecting to {db_path}")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Clear existing sample data
        print("\nClearing existing sample data...")
        cursor.execute("DELETE FROM TB_JOURNAL_ENTRIES WHERE mandt = '100' AND rbukrs IN ('HKG', '1000')")
        cursor.execute("DELETE FROM TB_GL_ACCOUNTS WHERE mandt = '100' AND saknr LIKE 'ACC%'")
        
        # Load GL account names
        names_file = sample_data_dir / "HKG_PL review Nov'25(Names).csv"
        if names_file.exists():
            load_gl_accounts(cursor, names_file)
        
        # Load exchange rates
        load_exchange_rates(cursor)
        
        # Load Raw TB November data
        nov_file = sample_data_dir / "HKG_PL review Nov'25(Raw TB Nov'25).csv"
        if nov_file.exists():
            load_raw_tb_data(cursor, nov_file, 'HKG', '011')
        
        # Load Raw TB October data (for variance comparison)
        oct_file = sample_data_dir / "HKG_PL review Nov'25(Raw TB oct'25).csv"
        if oct_file.exists():
            load_raw_tb_data(cursor, oct_file, 'HKG', '010')
        
        # Commit changes
        conn.commit()
        
        # Verify data
        print("\n" + "="*60)
        print("Data loaded successfully!")
        print("="*60)
        
        cursor.execute("SELECT COUNT(*) FROM TB_GL_ACCOUNTS WHERE mandt = '100'")
        account_count = cursor.fetchone()[0]
        print(f"GL Accounts: {account_count}")
        
        cursor.execute("SELECT COUNT(*) FROM TB_JOURNAL_ENTRIES WHERE mandt = '100'")
        entry_count = cursor.fetchone()[0]
        print(f"Journal Entries: {entry_count}")
        
        cursor.execute("SELECT COUNT(*) FROM TB_EXCHANGE_RATES WHERE mandt = '100'")
        rate_count = cursor.fetchone()[0]
        print(f"Exchange Rates: {rate_count}")
        
        print("\nSample data:")
        cursor.execute("""
            SELECT racct, sgtxt, drcrk, hsl, poper
            FROM TB_JOURNAL_ENTRIES 
            WHERE mandt = '100' 
            LIMIT 5
        """)
        for row in cursor.fetchall():
            print(f"  Account {row[0]}: {row[1][:30]} | {row[2]} | ${row[3]:,.2f} | Period {row[4]}")
        
        print("\n✓ Sample data loaded successfully!")
        print(f"\nDatabase location: {db_path}")
        
    except Exception as e:
        conn.rollback()
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()

if __name__ == '__main__':
    main()