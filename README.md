# Laporan Proyek Sistem Rekomenasi - Andreas Kevin


Pada era teknologi digital, tenologi machine learning menjadi teknologi utama dalam pengolahan dan analisis data. Machine learning membantu dalam banyak kasus. Penerapan machine learning saat bermacam, seperti kasus rekomendasi. Kasus rekomenasi merupakan kasus yang dilakukan dalam memberikan rekomenasi atas berbagai pilihan yang harus dipilih oleh costumers. 

Teknik rekomendasi dalam machine learning telah diadopsi dalam berbagai hal seperti sosial media dalam memberikan postingan yang disukai, marketplace dalam memberikan rekomendasi produk yang diinginkan oleh costumers, hingga dalam skala global seperi dilakukan oleh e-commerce seperti amazone, netflix, dan spotify.

Proyek ini memiliki tujuan untuk pengembangan model rekomendasri dalam membantu dalam menentukan rekomendasi lagu. Fokus utama pengembangan model machine learning ini menggunakan pendekatan Content-Based Filtering dan Collaborative Filtering.

Proyek ini dijadikan sebagai hal  penting diselesaikan karena dengan proyek ini, akan membantu pengguna dalam menentukan lagu yang harus dipilih berdasarkan lagu yang mereka dengar sebelumnya dan pada kasus ini dilakukan perbandingan model terbaik dalam solusi atas permasalahan tersebu. Dengan adanya model rekomendasi yang akurat, pengguna akan terbantu dalam membuat keputusan yang lebih cepat dalam menentukan lagu dan ini akan memberikan manfaat kepada aplikasi dlam memberikan rekomenadi lagu kepada pengguna.
  Beberapa hal yang membuat penelitian ini penting dilakukan dan penelitian sebelumnya mengenai teknik rekomendasi yang didasari pada:
  - [Menakar Preferensi Musik di Kalangan Remaja: Antara Musik Populer Dan Musik Klasik](https://e-journal.hamzanwadi.ac.id/index.php/tmmt/article/view/4365/2192), memaparkan akan fokus berbagai macam musik populer di kalangan remaja.
  - [Pengembangan Sistem Rekomendasi Berbasis Kecerdasan Buatan Untuk Meningkatkan Pengalaman Pengguna Di Platform E-Commerce](https://ejurnal.lkpkaryaprima.id/index.php/juktisi/article/download/145/112), penelitian tersebut berisi pentinya penggunaan AI dalam memberikan pengalaman pengguna terutama dalam E-Commerce.
  - [Pemanfaatan Sistem Rekomendasi Menggunakan Content-Based Filtering pada Hotel di Palangka Raya](https://ojs.uniska-bjm.ac.id/index.php/JIT/article/view/16282), menyampaikan pemanfaatan Content-Bsed Filtering yang dianggap akurat dalam rekomendasi hotel di Palangka Raya.
  - [PREFERENSI MUSIK DI KALANGAN REMAJA](https://journal.isi.ac.id/index.php/promusika/article/download/541/753), yang memparkan bahwa terdpat banyak faktor dalam pemempgaruhi ketertarikan seseorang terhadap genre musik.

---
### ⚠️ Problem Statements
Adapun yang menjadi dasar masalah dalam penelitian ini berupa:
- Bagaimana sistem dapat memberikan rekomendasi lagu yang relevan berdasarkan kemiripan konten (judul lagu) dan kemiripan konten tersebut tercermin dalam skor evaluasi?
- Bagaimana sistem memberikan rekomendasi lagu secara personal kepada pengguna berdasarkan favorite artis sebelumnya, dan seberapa akurat model tersebut dalam memprediksi skor/penilaian terhadap lagu yang belum pernah didengarkan?

---
### ✨ Goals
Berdasarkan problem statements yang telah dipaparkan penulis, adapun yang menjadi tujuan dalam penelitian ini yaitu:
- Menentukan model rekomendasi yang dapat digunakn dalam menentukan solusi terbaik atas rekomendasi lagu.
- Membandingkan dan memberikan solusi atas perbandingan tersebut untuk sistem rekomendasi terbaik.

---
### 💡 Solution Approach
Dalam menyelesaikan permasalahan yang telah diidentifikasi tersebut, penelitian ini menerapkan pendekatan berbasis machine learning dengan memanfaatkan . Penelitian dilakukan dengan menggunakan algoritma machine learning yaitu Content Base Filtering dan Collaborative Filtering. Dalam Collaborative Filtering, digunakan teknik neural network dalam membantu model ini. Dalam meningkatkan pembelajaran yang lebih baik, penelitian dilkaukan dengan beberapa proses yang meliputi pengolahan data, eksplorasi fitur, pemilihan model, hingga evaluasi performa untuk memastikan akurasi yang optimal dalam prediksi diabetes.

## 📑 Data Understanding
Dataset merupakan kumpulan data yang teorganisir. Pada kasus ini digunakan dataset lagu. Dataset berisikan kumpulan lagu yang diperoleh dari kaggle. [Song Dataset](https://www.kaggle.com/datasets/deependraverma13/all-songs-rating-review/data).

### Variabel-variabel yang terdapat pada dataset tersebut yaitu:
- **Name of the song** = Berisikan nama lagu-lagu.
- **Artist** = Berisikan nama artis dari lagu tersebut.
- **Date of Release** = Tanggal dirilis lagu tersebut.
- **Description** = Rincian deskripsi mengenai lagu.
- **Metascore** = Nilai rata-rata berdasarkan kritikus.
- **User Score** = Nilai score dari pengguna mengenai lagu tersebut.

---
### Proses analisis data
Proses pengumpulan data yang telah dilkaukan kemudian dilakukan analisis atas data teresebut. Analisis terebut dilkukan untuk memahami konidisi isi atas data tersebut. Beberapa teknik analisis yang dilakukan berupa: 
- **.info()** untuk meliht informasi atas dataset terserbut berupa jenis tipe data hingga jumlah data.
- **isnull().sum()** digunakan untuk menilai jumlah data yang kosong pada data setelah itu akan dilkaukan **.dropna()** untuk membuang baris yang kosong.
- **.duplicated().sum()** untuk menilai jumlah duplikasi pada data. Pada bagian ini terlihat jumlah duplikasi pada data adalah 0.
- **.duplicated(subset=['Name of the Song']).sum()** untuk melihat jumlah duplikasi data pada atribut Name of the Song. Pada ini terdapat data lagu yang duplicate. Maka dilkaukan **.drop_duplicates(subset=["Name of the Song"])** untuk membuang data yang kosong tersebut.

---
### Visualisasi Data
Beberapa teknik visualisasi digunakan untuk memahami distribusi data antara lain:
- **Histogram**: Digunakan untuk melihat distribusi frekuensi data dalam tiap variabel.
![Histogram](gambar/histogram.png)


## 🛠️ Data Preparation
Data preparation merupakan proses penyiapan data agar dapat diproses lebih lanjut (sebelum dilakukan pelatihan atas model). Pada kasus ini, proses data preparation dapat diuraikan menjadi:
### 1. Data Preparation Content Base Filtering
Data preparation untuk Content Base Filtering didasari pada :
- Membuang atribut **Description, Unnamed: 0, dan Date of Release**. Tujuan dari pembuangan atribut ini untuk mempersiapkan atribut yang akan digunakan pada proses CBF.
- Total data yang siap digunakan untuk content base filtering ini adalah 2537 data dengan atribut nama lagu, artis, metascore, dn user score.

---
### 2. Data Preparation Collaborative Filtering
Data preparation untuk Collaborative Filering didasari pada:
- Penggunaan encoder untuk mengubah data sring menjadi numerik yang terdiri dari atribut **Artist** dan **Name of the Song**.
- Melakukan normalisasi pada data sebagai berikut:
```python
min_rating = min(df_songs['Metascore'])
max_rating = max(df_songs['Metascore'])

x = df_songs[['Artist', 'Name of the Song']].values
y = df_songs['Metascore'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

scaler_x = MinMaxScaler()
x = scaler_x.fit_transform(x)
```
Proses tersebut membagi dataset menjadi x sebagai input dan y sebagai output.
- Proses pembgian dataset menjadi train dan test yang dibagi menjadi 80% sebagai train dan 20% sebagai test.
```python
train_indices = int(0.8 * df_songs.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
```

## 🤖 Modeling
Proses modeling memuat proses perancangan model yang digunakan dalam rekomendasi.

---
### Content Base Filtering
Content Base Filtering yang dibagun didasarkan pada beberapa komponen penting yang berupa:
- TF-IDF digunakan untuk mengubah judul lagu menjadi representasi numerik.
- Cosine similarity dihitung similariity antar judul lagu, menghasilkan matriks similarity.
```python
tf = TfidfVectorizer()
tfidf_matrix = tf.fit_transform(df_songs['Name of the Song']) 

cosine_sim = cosine_similarity(tfidf_matrix) 
```
Proses ini adalah proses inti dalam content base filtering. Content base filtering penting dalam menghitung similarity hal ini dikarenakan content base filtering mengukur kemiripan antara lagu-lagu.
- Fungsi Rekomenadasi pada 
```python
  closest = similarity_data.columns[index[-1:-(k+2):-1]]
```
yang pada proses ini mengambil index dari nilai similarity tertinggi dan mengeluarkan rekomendasi lagu yang paling relevan.

---
### Collaborative Filtering
Beberapa hal yang mendasari pada collaborative filtering adalah sebagai berikut:
- Collaborative filtering dibangun dengan model neural network dengan bantuan TensorFlow dan keras. Proses ini dilakukan dengan membuat vektor berdimensi embendding untuk artist dan lagu untuk menangkap hubungan antar indentitas.
- Dot product (hasil dari 2 vektor yang menghasilkan angka skalar (digunakan dalam sistem rekomendasi berbasis embedding)) antara embending artis dan lagu diikuti dengan penambahan bias dalam menghasilkan prediksi rating yang dinormalisasi melalui fungsi sigmoid.
```python
# Dot product antara artist dan song
dot_artist_song = tf.reduce_sum(artist_vector * song_vector, axis=1, keepdims=True)

# Menambahkan bias
x = dot_artist_song + artist_bias + song_bias
x = self.dropout(x)

# Aktivasi sigmoid untuk output antara 0 dan 1
return tf.nn.sigmoid(x)
```
- Model dilakukan komplikasi menggunakan MSE sebagai loss function dan RMSE sebagai metrik evaluasi.
```python
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),  # karena output sigmoid (0-1)
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```
- Proses callback dilakukan agar proses latih dihentikan ketika mencapai nilai yang diinginkan.
- Proses kemudian dilakukan latih dengan fungsi **.fit(...)**
```python
history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 64,
    epochs = 100,
    callbacks=[callbacks, early_stop, reduce_lr], 
    validation_data = (x_val, y_val),
    verbose=1 
)
```
proses tersebut meliputi pelatihan, hingga selesainya yang dimaksimalkan dengan x dari x_train dan y dari y_train, dengan ukuran data setiap pelatihan yaitu 64, 100 iterasi maksimal, batas callbacks, dan validasi data berdasarkan x_val dan y_val yang telah dibangun, serta menampilkan proses bar selama pelatihan dengan verbose=1.

---
### Kelebihan dan kekurangan CBF dan CF
#### Content Base Filtering
**Kelebihan:**
- Tidak memerlukan data interaksi pengguna (user rating), cukup dari konten seperti judul lagu.
- Dapat memberikan rekomendasi meskipun hanya berdasarkan satu lagu (cocok untuk cold start user).
- Proses implementasi relatif lebih sederhana dan cepat.

**Kekurangan:**
- Merekomedasikan lagu yang mirip dengan yang sudah diketahui, sehingga tidak bisa menangkap selera musik yang lebih luas.
- Berdasarkan model yang dibangun, CBS bergantung sepenuhnya pada judul lagu.


#### Colaborative Filtering
**Kelebihan:**
- Mampu menangkap pola selera pengguna berdasarkan interaksi (artis dan lagu dengan rating).
- Memberikan rekomendasi yang lebih personal karena mempertmbangkan relasi antara artis dan lagu.
- Bisa memberikan rekomendasi yang tidak memiliki kemiripan konten.

**Kekurangan:**
- Butuh data interaksi dalam jumlah cukup besar untuk bekerja optimal.
- Rumit dalam hal pemodelan dan pelatihan (butuh arsitektur neural network, dll.).


## 🔍 Evaluation
### Evaluasi Content Based Filtering
Evaluasi dilakukan dengan menghitung rata-rata cosine similarity. Ini dilakukan dengan perintah
```python
def avg_cosine_similarity(k=3):
    total_sim = 0
    count = 0

    for idx, row in df_songs.iterrows():
        target_song = row['Name of the Song']

        if target_song not in cosine_sim_df.columns:
            continue
        try:
            recs = song_recomendations(target_song, k=k)
        except:
            continue

        # Ambil similarity score antara target dan rekomendasi
        for rec_song in recs['Name of the Song']:
            sim_score = cosine_sim_df.loc[target_song, rec_song]
            total_sim += sim_score
            count += 1

    avg_sim = total_sim / count if count else 0
    print(f"Avg Cosine Similarity@{k}: {avg_sim:.4f}")
    return avg_sim

avg_cosine_similarity(3)
```
proses tersebut meliputi 
- **def avg_cosine_similarity(k=3):** merupakan fungsi untuk menghitung rata-rata nilai kemiripan (cosine similarity) antara lagu target dan hasil rekomendasinya.
- **for idx, row in df_songs.iterrows():** digunakan untuk melakukan iterasi ke setiap lagu dalam dataset.
- **if target_song not in cosine_sim_df.columns:** mengecek apakah lagu target ada di dalam data similarity, jika tidak maka dilewati.
- **song_recommendations(target_song, k=k)** dipanggil untuk mendapatkan rekomendasi lagu berdasarkan lagu target.
- **cosine_sim_df.loc[target_song, rec_song]** mengambil nilai cosine similarity antara lagu target dan lagu hasil rekomendasi.
- **total_sim += sim_score** menjumlahkan semua nilai similarity dari seluruh lagu dan rekomendasinya.
- **avg_sim = total_sim / count if count else 0** menghitung rata-rata similarity hanya jika ada data yang valid, jika tidak maka hasilnya 0.
- Hasil evaluasi diperoleh Cosine similarity sebesar 0,29. Hasil ini dapat dikatakan cukup mirip dengan konten, hal ini dikarenakan dataset yang sangat besar dan beragam. 

---
### Evaluasi Collaborative Filtering
Evaluasi atas collaborative Filltering meliputi penggunaan RMSE berdasarkan perbandingan dari y_value dan y_prediksi.
![RMSE](gambar/rmse.png)
```python
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
```
- RMSE cocok digunakan dalam kasus collaborative filtering dikarenakan mengukur jarak asli dengan hasil prediksi sehingga metode evaluasi ini dinilai cocok. 
- Hasil evaluasi diperoleh nilai RMSE 0,1946 yang merupakan baik dikarenakan RMSE < 0,2. Hal ini dengan mempertimbangkan dataset yang besar dan bervariasi sehingga Collaborative Filtering dianggap mampu untuk menebak score lagu.

## 🎯 Kesimpulan
- Content-Base Filtering lebih mudah dalam penerapan karena hanya menggunakan TF-IDF dan cosine similarity berdasarkan kemiripan judul lagu dengan hasil average cosine similarity adalah 0,29.
- Collavorative filtering lebih sulit dalam hal implementasi karena membutuhkan model neural network, dll dalam membantu model ini. Tetapi hasil testing pada collaborative filtering mampu mencapai nilai RMSE 0,1946 yang dianggap baik karena nilai RMSE < 0,2.
- Berdasarkan hasil Content-Base Filtering dan Collaborative Filtering, Collaborative filtering lebih unggul dalam memberikan saran rekomendasi lagu. dikarenakan nilai RMSE yang baik.

#   s i s t e m r e k o m e n d a s i  
 #   S i s t e m r e k o m e n d a s i _ C B F _ C F  
 