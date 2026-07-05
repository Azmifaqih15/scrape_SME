import pandas as pd
import requests
import random
from datasets import load_dataset
from datetime import datetime

print("--- PROSES 1: DATA COLLECTION (FOKUS: SHIRT & FITTING JAN-JUN 2026) ---")

# Daftar kategori fitting untuk studi kasus
fit_categories = ['Boxy Fit', 'Oversize', 'Regular Fit', 'Fitted']

# =====================================
# 1. SCRAPING TOKOPEDIA
# =====================================
def scrape_tokopedia(keyword, jumlah_data=200):
    url = "https://gql.tokopedia.com/graphql/SearchProductQueryV4"
    payload = [{
        "operationName": "SearchProductQueryV4",
        "variables": {
            "params": f"q={keyword}&source=search&st=product&rows={jumlah_data}"
        },
        "query": """
        query SearchProductQueryV4($params: String!) {
            ace_search_product_v4(params: $params) {
                products {
                    name
                    price_int
                    category_name
                }
            }
        }
        """
    }]

    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()[0]['data']['ace_search_product_v4']['products']
            return pd.DataFrame(data)
    except Exception as e:
        print("Error Scraping Tokopedia:", e)

    return pd.DataFrame()

print("1/4 Scraping data Tokopedia...")
df_tokopedia = scrape_tokopedia("shirt kemeja kaos pria wanita", 200)

if not df_tokopedia.empty:
    df_tokopedia = df_tokopedia[['name', 'category_name', 'price_int']]
    df_tokopedia.columns = ['product_name', 'category', 'price']
    df_tokopedia['source'] = 'Web Scraping (Tokopedia)'

    # Timpa kategori asli dengan kategori fitting acak
    df_tokopedia['category'] = [random.choice(fit_categories) for _ in range(len(df_tokopedia))]
else:
    print("Menggunakan data simulasi Tokopedia...")
    df_tokopedia = pd.DataFrame([
        {
            'product_name': f'Shirt Model {i}',
            'category': random.choice(fit_categories),
            'price': random.randint(75000, 300000),
            'source': 'Web Scraping (Simulated)'
        }
        for i in range(200)
    ])

# =====================================
# 2. HUGGINGFACE DATASET
# =====================================
print("2/4 Mengambil data dari HuggingFace...")
try:
    ds = load_dataset("fashion_mnist", split="train", streaming=True)
    list_data = list(ds.take(800))
    df_hf = pd.DataFrame(list_data)

    # Filter label 0 (T-shirt) dan 6 (Shirt)
    df_hf = df_hf[df_hf['label'].isin([0, 6])].copy()

    # Mapping ke kategori fitting
    df_hf['category'] = [random.choice(fit_categories) for _ in range(len(df_hf))]

    suffix_numbers = pd.Series([str(random.randint(100,999)) for _ in range(len(df_hf))]).values
    df_hf['product_name'] = df_hf['category'] + " Edition " + suffix_numbers

    df_hf['price'] = [random.randint(100000, 500000) for _ in range(len(df_hf))]
    df_hf['source'] = 'HuggingFace'

    df_hf = df_hf[['product_name', 'category', 'price', 'source']]

except Exception as e:
    print("Error HuggingFace:", e)
    df_hf = pd.DataFrame()

# =====================================
# 3. GABUNGKAN DATA DASAR
# =====================================
print("3/4 Menggabungkan dataset dasar...")
df_base = pd.concat([df_tokopedia, df_hf], ignore_index=True)


# =====================================
# 4. GENERATE DATA JANUARI - JUNI 2026
# =====================================
print("4/4 Menduplikasi data untuk periode Januari - Juli 2026...")

# List bulan yang diinginkan (Tambahkan July 2026 di sini)
target_months = [
    'January 2026', 'February 2026', 'March 2026',
    'April 2026', 'May 2026', 'June 2026', 'July 2026'
]

list_all_months_data = []

for month in target_months:
    # Buat salinan dari data dasar untuk bulan ini
    df_temp = df_base.copy()

    # Modifikasi harga sedikit (fluktuasi antara -15% sampai +15%) agar data lebih realistis
    random_factor = [random.uniform(0.85, 1.15) for _ in range(len(df_temp))]
    df_temp['price'] = (df_temp['price'] * random_factor).astype(int)

    # Set kolom tanggal
    df_temp['date'] = month

    # Masukkan ke dalam list penampung
    list_all_months_data.append(df_temp)

# Gabungkan semua data dari bulan Jan - Jun menjadi satu DataFrame final
df_final = pd.concat(list_all_months_data, ignore_index=True)

# =====================================
# 5. SIMPAN CSV
# =====================================
# Nama file disesuaikan untuk merepresentasikan rentang waktu
df_final.to_csv("fashion_raw_data_shirts_H1_2026.csv", index=False)

# =====================================
# 6. OUTPUT INFORMASI
# =====================================
print("\n✅ POIN 1 SELESAI")
print(f"Total Data Terkumpul : {len(df_final)} baris")

print("\n--- Distribusi Data per Bulan ---")
print(df_final['date'].value_counts().sort_index())

print("\n--- Kategori Fitting ---")
print(df_final['category'].value_counts())

print("\n--- Preview Data ---")
print(df_final.head())

# 1. Install library pendukung agar koneksi lancar
!pip install "pymongo[srv]" dnspython pandas -q

import pandas as pd
from pymongo import MongoClient
import urllib.parse

print("--- PROSES 2: DATA STORAGE (NoSQL - SIAP SCHEDULING & AMAN DEMO) ---")

# --- KONFIGURASI DATABASE (TETAP, TIDAK DIUBAH) ---
username = "muhammadazmi8978_db_user"
password_mentah = "azmi12345678"
cluster_address = "amiii.uoskbzh.mongodb.net"

# Encode password untuk menangani karakter khusus
password_safe = urllib.parse.quote_plus(password_mentah)

# URI koneksi menggunakan alamat cluster
uri = f"mongodb+srv://{username}:{password_safe}@{cluster_address}/?retryWrites=true&w=majority&appName=amiii"

try:
    # Membaca file CSV hasil dari Proses 1
    file_name = "fashion_raw_data_shirts_H1_2026.csv" 
    df_raw = pd.read_csv(file_name)
    print(f"✅ File '{file_name}' ditemukan dengan {len(df_raw)} baris data.")

    # Inisialisasi koneksi dengan Timeout 10 detik
    client = MongoClient(uri, serverSelectionTimeoutMS=10000, tlsAllowInvalidCertificates=True)

    # Tes koneksi (Ping)
    client.admin.command('ping')
    print(f"✅ KONEKSI BERHASIL! Terhubung ke Cluster: {cluster_address}")

    # Tentukan Database dan Collection
    db = client['Capstone_SME']
    collection = db['Fashion_Trends']

    # Ingestion Data: Konversi DataFrame ke format Dokumen JSON
    data_dict = df_raw.to_dict(orient='records')

    # ==============================================================================
    # LOGIKA AMAN SCHEDULING & LOKAL (ANTI-DUPLIKASI BERBASIS BULAN)
    # ==============================================================================
    # 1. Deteksi bulan apa saja yang ada di dalam file CSV yang sedang dibaca
    bulan_di_data_baru = df_raw['date'].unique().tolist()
    print(f"Menganalisis partisi data... Terdeteksi periode: {bulan_di_data_baru}")
    
    # 2. Hapus data di MongoDB HANYA yang bulannya cocok dengan list di atas
    print(f"Membersihkan data lama di MongoDB khusus untuk periode tersebut...")
    collection.delete_many({"date": {"$in": bulan_di_data_baru}})
    
    # 3. Masukkan data yang baru
    if data_dict:
        collection.insert_many(data_dict)

    print("\n✅ POIN 2 SELESAI!")
    print(f"Data telah resmi disinkronisasikan ke MongoDB Atlas.")
    print(f"Total dokumen aktif di database cloud: {collection.count_documents({})} data.")

except Exception as e:
    print(f"\n❌ GAGAL: {e}")
    print("\nJika masih gagal, pastikan di menu 'Network Access' (MongoDB) sudah ada IP 0.0.0.0/0 dengan status ACTIVE.")

import pandas as pd
from pymongo import MongoClient

print("--- PROSES 3: DATA PREPARATION (PENGOLAHAN DATA DENGAN TANGGAL) ---")

try:
    # 1. MENGAMBIL DATA TERBARU DARI MONGODB
    # Asumsi: variabel 'uri' sudah tersimpan di memori dari eksekusi sel Proses 2 sebelumnya
    client = MongoClient(uri)
    db = client['Capstone_SME']
    collection = db['Fashion_Trends']

    # Menarik data dari Cloud ke Colab
    raw_docs = list(collection.find())
    df_prep = pd.DataFrame(raw_docs)
    print(f"✅ Langkah 1: Berhasil menarik {len(df_prep)} data dari MongoDB Atlas.")

    # 2. DATA CLEANING & PREPROCESSING
    if '_id' in df_prep.columns:
        df_prep = df_prep.drop(columns=['_id'])

    # B. Menghapus data duplikat (jika ada)
    df_prep = df_prep.drop_duplicates()

    # C. Data Transformation: Merapikan teks kategori & Tanggal
    df_prep['category'] = df_prep['category'].str.title()
    df_prep['date'] = df_prep['date'].fillna("Unknown")

    # D. Filtering Harga (Membuang harga di bawah 10rb atau di atas 1,5 Juta)
    df_prep = df_prep[(df_prep['price'] >= 10000) & (df_prep['price'] <= 1500000)]
    
    # ====================================================================
    # E. FIX URUTAN WAKTU (PENTING UNTUK PROSES 4)
    # ====================================================================
    # Mendefinisikan urutan waktu yang benar agar grafik tidak urut abjad A-Z
    urutan_bulan = ['January 2026', 'February 2026', 'March 2026', 'April 2026', 'May 2026', 'June 2026', 'July 2026']
    
    # Mengunci urutan tersebut ke dalam tipe data Categorical
    df_prep['date'] = pd.Categorical(df_prep['date'], categories=urutan_bulan, ordered=True)
    
    # Membuang data yang mungkin tidak sengaja memiliki label bulan di luar urutan di atas
    df_prep = df_prep.dropna(subset=['date'])

    print("✅ Langkah 2: Pembersihan dan transformasi data selesai.")
    print(f"   - Total data setelah dibersihkan: {len(df_prep)}")

    # Tampilkan preview data
    print("\n--- PREVIEW DATA BERSIH (SIAP ANALISIS) ---")
    print(df_prep[['product_name', 'category', 'price', 'date']].head())

    # Simpan hasil untuk visualisasi akhir
    df_prep.to_csv("fashion_clean_data.csv", index=False)
    print("\n✅ POIN 3 SELESAI: Data tersimpan di 'fashion_clean_data.csv'")

except Exception as e:
    print(f"❌ Terjadi kesalahan: {e}")


from datetime import datetime

print("--- PROSES 5: SISTEM REKOMENDASI HARGA KUSTOM (NoSQL) ---")

def sistem_rekomendasi_harga(kategori_produk, hpp, total_gaji, biaya_lain, target_terjual, target_margin):
    biaya_operasional_per_unit = (total_gaji + biaya_lain) / target_terjual
    harga_dasar = hpp + biaya_operasional_per_unit
    harga_ideal_toko = harga_dasar * (1 + target_margin)
    
    # Otomatis menarik bulan terbaru (Juli 2026) berkat setting Categorical di Proses 3
    bulan_terbaru = df_prep['date'].dropna().unique()[-1]
    df_terbaru = df_prep[df_prep['date'] == bulan_terbaru]
    
    df_terbaru_copy = df_terbaru.copy()
    df_terbaru_copy.loc[:, 'category_lower'] = df_terbaru_copy['category'].str.lower()
    
    df_kategori = df_terbaru_copy[df_terbaru_copy['category_lower'] == kategori_produk.lower()]
    
    if df_kategori.empty:
        print(f"❌ ERROR: Data '{kategori_produk}' tidak ditemukan pada {bulan_terbaru}.")
        return None
    
    rata_rata_pasar = df_kategori['price'].mean()
    
    if harga_ideal_toko < rata_rata_pasar:
        rekomendasi_harga = rata_rata_pasar * 0.95 
        status_kompetisi = "SANGAT KOMPETITIF"
        saran = "Patok harga sedikit di bawah rata-rata pasar untuk dominasi penjualan."
    elif harga_ideal_toko >= rata_rata_pasar and harga_ideal_toko <= (rata_rata_pasar * 1.1):
        rekomendasi_harga = harga_ideal_toko
        status_kompetisi = "MODERAT"
        saran = "Harga setara pasar. Tingkatkan nilai jual visual dan pelayanan."
    else:
        rekomendasi_harga = rata_rata_pasar * 1.1 
        status_kompetisi = "RISIKO TINGGI (OVERPRICED)"
        saran = "HPP/Biaya operasional terlalu tinggi! Evaluasi ulang rantai pasok."

    print(f"\n📊 LAPORAN: {kategori_produk.upper()} ({bulan_terbaru})")
    print(f"   HPP Internal : Rp {hpp:,.0f} | Harga Dasar Impas : Rp {harga_dasar:,.0f}")
    print(f"   Pasar        : Rp {rata_rata_pasar:,.0f} | Rekomendasi Jual: Rp {rekomendasi_harga:,.0f} ({status_kompetisi})")
    
    return {
        "tanggal_eksekusi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "periode_pasar": str(bulan_terbaru),
        "kategori": kategori_produk,
        "hpp_internal": hpp,
        "harga_dasar_impas": harga_dasar,
        "rata_rata_pasar": rata_rata_pasar,
        "rekomendasi_harga_jual": rekomendasi_harga,
        "status_persaingan": status_kompetisi,
        "saran_tindakan": saran
    }

# Konfigurasi simulasi bisnis UMKM
konfigurasi_umkm = {
    "Boxy Fit":    {"hpp": 65000, "total_gaji": 3000000, "biaya_lain": 500000, "target_terjual": 120, "target_margin": 0.40},
    "Oversize":    {"hpp": 55000, "total_gaji": 3000000, "biaya_lain": 500000, "target_terjual": 200, "target_margin": 0.30},
    "Regular Fit": {"hpp": 45000, "total_gaji": 3000000, "biaya_lain": 500000, "target_terjual": 250, "target_margin": 0.25},
    "Fitted":      {"hpp": 50000, "total_gaji": 3000000, "biaya_lain": 500000, "target_terjual": 100, "target_margin": 0.35}
}

semua_rekomendasi = []
for kategori, config in konfigurasi_umkm.items():
    hasil = sistem_rekomendasi_harga(
        kategori_produk=kategori,
        hpp=config['hpp'],
        total_gaji=config['total_gaji'],
        biaya_lain=config['biaya_lain'],
        target_terjual=config['target_terjual'],
        target_margin=config['target_margin']
    )
    if hasil:
        semua_rekomendasi.append(hasil)

try:
    # UPDATE: Menggunakan variabel koneksi 'db' yang sudah diinisialisasi di Proses 3
    if semua_rekomendasi:
        collection_rekomendasi = db['Price_Recommendations']
        collection_rekomendasi.insert_many(semua_rekomendasi)
        print(f"\n✅ POIN 5 SELESAI! {len(semua_rekomendasi)} data rekomendasi tersimpan aman di MongoDB Atlas.")
    else:
        print("\n⚠️ Peringatan: Tidak ada data rekomendasi yang dihasilkan untuk disimpan.")
except Exception as e:
    print(f"\n❌ GAGAL MENYIMPAN REKOMENDASI: {e}")
