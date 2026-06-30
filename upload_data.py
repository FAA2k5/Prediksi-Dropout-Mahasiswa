# import_to_supabase.py
import pandas as pd
from sqlalchemy import create_engine

# 1. Load dataset
url = "https://docs.google.com/spreadsheets/d/1_TPQP7Z-r-DSZFEnV50uJYevCZ07dTKfG3-EPHISgNs/export?format=csv"
df = pd.read_csv(url, sep=',')

print(f"Data loaded: {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")

# 2. Koneksi ke Supabase
DATABASE_URL = "postgresql://postgres.ovfkgdinzcoaincohjoj:Ari527668__@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"

try:
    engine = create_engine(DATABASE_URL)
    
    # 3. Import data ke tabel 'students'
    df.to_sql('students', engine, if_exists='replace', index=False)
    
    print("Data BERHASIL diupload ke Supabase!")
    print(f"📊 Table: students, {len(df)} rows")
    
    # 4. Verifikasi
    print("\nVerifikasi data:")
    import pandas as pd
    query = "SELECT * FROM students LIMIT 5"
    df_sample = pd.read_sql(query, engine)
    print(df_sample.head())
    
except Exception as e:
    print(f"❌ Error: {e}")