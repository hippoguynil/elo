import pandas as pd
import requests
import io
import time
import os
import glob

# Constants
START_YEAR = 1993
END_YEAR = 2025
DIVISIONS = ['E0', 'E1', 'E2', 'E3']
DATA_DIR = 'data'

# Expanded columns to keep if available
# We won't strictly enforce usecols in read_csv because older files might miss some columns.
# We will filter AFTER loading.
# Standard columns + Odds + Stats
DESIRED_COLUMNS = [
    'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Season', 'Division',
    'B365H', 'B365D', 'B365A', # Bet365 Odds
    'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR' # Match Stats
]

def download_historical_results(divisions=DIVISIONS, start_year=START_YEAR, end_year=END_YEAR):
    """
    Downloads historical English football results and saves them to a local folder.
    """
    # Generate a list of seasons to scrape
    seasons = [f'{str(year)[2:]}{str(year+1)[2:]}' for year in range(start_year, end_year + 1)]
    
    print(f"Downloading data for seasons from {seasons[0]} to {seasons[-1]}...")
    
    for division in divisions:
        os.makedirs(os.path.join(DATA_DIR, division), exist_ok=True)
        
        for season in seasons:
            url = f'https://www.football-data.co.uk/mmz4281/{season}/{division}.csv'
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                # We read everything first to ensure we don't crash on missing cols
                # But we have to handle potential decoding errors
                content = response.content.decode('utf-8', errors='replace')
                
                df = pd.read_csv(io.StringIO(content))
                df['Season'] = season
                df['Division'] = division
                
                # Check for Date and drop if missing
                if 'Date' in df.columns:
                    df.dropna(subset=['Date'], inplace=True)
                
                output_path = os.path.join(DATA_DIR, division, f"{division}_{season}.csv")
                df.to_csv(output_path, index=False)

                print(f'Successfully downloaded {division} data for season {season}.')
                time.sleep(0.5) # Be nice to the server
            
            except requests.exceptions.RequestException as e:
                print(f'Error downloading {division} data for season {season}: {e}')
                continue
            except Exception as e:
                print(f'Error processing {division} data for season {season}: {e}')
                continue

def combine_data(divisions=DIVISIONS):
    print("Combining data...")
    pd_base = pd.DataFrame()
    
    for division in divisions:
        all_files = glob.glob(os.path.join(DATA_DIR, division, "*.csv"))
        for filename in all_files:
            try:
                df = pd.read_csv(filename)
                df['filename'] = filename
                
                # Normalize columns?
                # We interpret columns broadly. 
                # If desired columns exist, keep them. If not, fill with NaN later?
                # Actually, concating handles missing columns by adding NaNs. 
                # We just need to make sure we don't drop them beforehand.
                
                pd_base = pd.concat([pd_base, df], ignore_index=True)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
    # Normalize Date
    if 'Date' in pd_base.columns:
        # Standard format appears to be dd/mm/yy or dd/mm/yyyy
        pd_base['Date'] = pd.to_datetime(pd_base['Date'], dayfirst=True, errors='coerce')
        pd_base = pd_base.sort_values(by='Date')
        
    # keep only subset of columns + anything else important?
    # Let's keep ALL columns for now in the master file, it's safer.
    # We can filter in the model script if needed.
    
    output_file = os.path.join(DATA_DIR, 'all_data.csv')
    pd_base.to_csv(output_file, index=False)
    print(f"Data combined and saved to {output_file}. Total rows: {len(pd_base)}")
    return pd_base

if __name__ == "__main__":
    download_historical_results()
    combine_data()
