import pandas as pd
import requests
from fredapi import Fred
from google.colab import userdata
import io 

PUBLIC_LIST_ID = 5038
API_KEY = userdata.get('FREDAPI')       # ← put your key here
fred = Fred(api_key=API_KEY)

csv_url = f"https://fredaccount.stlouisfed.org/public/datalist/{PUBLIC_LIST_ID}"


dfs = pd.read_html(csv_url)                                     
df = dfs[1]

# the CSV contains a column usually named "series_id"
series_ids = df["Series ID"].tolist()

print(f"Found {len(series_ids)} series IDs.")
print(series_ids[:10])    # preview

# ---------------------------------------------------------
# 2. DOWNLOAD ALL SERIES FROM FRED API
# ---------------------------------------------------------
all_series = {}

for sid in series_ids:
    try:
        s = fred.get_series(sid)
        all_series[sid] = s
    except Exception as e:
        print(f"Error fetching {sid}: {e}")

# ---------------------------------------------------------
# 3. MERGE INTO ONE DATAFRAME
# ---------------------------------------------------------
df_all = pd.DataFrame(all_series)

print(df_all.head())
df_all.to_csv("fred_5038_all_series.csv")
