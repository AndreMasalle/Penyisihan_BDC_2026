# BDC 2026

| Nama Anggota | Asal Universitas   |
|--------------|--------------------|
| Darel Ajni Fahrezi          | Universitas Telkom |
| Ramadhan Abdul Aziz         | Universitas Telkom       |
| Andre Fransiscus Masalle       | Universitas Telkom       |

Repo ini berisi model untuk penyisihan BDC 2026 - Klasifikasi Multiclass gambar sampah

### Struktur Repo
1. `extract_to_csv.ipynb` - Notebook untuk mengekstrak metadata dataset dari folder `train` ke dalam file CSV:
   1. Membaca folder `train`
   2. Mengekstrak metadata setiap gambar
   3. Mengekspor metadata dataset ke file `train.csv`
2. `cleaning.ipynb` - Notebook yang berisi pembersihan dataset:
   1. Cek gambar corrupt
   2. Cek & hapus duplikat
   3. Cek & hapus gambar yang terlalu blur 
3. `eda.ipynb` - Notebook eksplorasi data:
   1. Distribusi Kelas
   2. Analisis rasio ukuran gambar
   3. Analisis intensitas warna per kela
   4. Analisis t-SNE (Dimensionality Reduction)
   5. Saliency & Gradcam pada Model Probe (Resnet18)
4. `preprocess.ipynb` - Notebook berisi alur preprocessing :
   1. Hapus Background & validasi
   2. Filter Dataset
   3. Resize
5. `model.ipynb` - Notebook berisi model yang digunakan dalam training
   1. Pipeline
   2. Resnet50 (Baseline)

