import pandas as pd
import os

EXISTING_FILE = "dataset.csv"
NEW_FILE = "c:/Users/chenn/Downloads/kisan_call_centre_dataset.csv"

def merge_datasets():
    print("Loading datasets...")
    
    # 1. Load Existing
    try:
        df_existing = pd.read_csv(EXISTING_FILE)
        print(f"Existing dataset loaded: {len(df_existing)} rows.")
    except FileNotFoundError:
        print("Existing dataset not found. Creating new...")
        df_existing = pd.DataFrame()

    # 2. Load New
    try:
        df_new = pd.read_csv(NEW_FILE, encoding='latin-1')
        print(f"New dataset loaded: {len(df_new)} rows.")
    except FileNotFoundError:
        print(f"Error: New dataset file not found at {NEW_FILE}")
        return

    # 3. Rename columns in New to match Existing
    # Existing: StateName,DistrictName,BlockName,Season,Sector,Category,Crop,Query Type,QueryText,KccAns
    # New: state,district,block,season,sector,category,crop,query_type,query_text,answer_text,language,source
    
    rename_map = {
        'state': 'StateName',
        'district': 'DistrictName',
        'block': 'BlockName',
        'season': 'Season',
        'sector': 'Sector',
        'category': 'Category',
        'crop': 'Crop',
        'query_type': 'Query Type',
        'query_text': 'QueryText',
        'answer_text': 'KccAns'
    }
    
    df_new_renamed = df_new.rename(columns=rename_map)
    
    # Select only relevant columns
    desired_columns = ['StateName', 'DistrictName', 'BlockName', 'Season', 'Sector', 'Category', 'Crop', 'Query Type', 'QueryText', 'KccAns']
    
    # Filter columns that exist in the renaming (ignore language, source for now if they don't map)
    # Actually, we should only keep columns that are in the desired target schema
    df_new_final = df_new_renamed[desired_columns]
    
    # 4. Concatenate
    df_combined = pd.concat([df_existing, df_new_final], ignore_index=True)
    
    # 5. Deduplicate (optional but good practice)
    # initial_len = len(df_combined)
    # df_combined.drop_duplicates(subset=['QueryText', 'KccAns'], inplace=True)
    # dedup_len = len(df_combined)
    dedup_len = len(df_combined) 
    
    # 6. Save
    df_combined.to_csv(EXISTING_FILE, index=False)
    print(f"Merged successfully. Total rows: {dedup_len}.")

if __name__ == "__main__":
    merge_datasets()
