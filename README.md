# Clustering
Implementasi tiga cara clustering yaitu Agglomerative, DBScan, dan K-Means

**Kontributor**:

13516045 - Dion Saputra


13516079 - Harry Setiawan Hamjaya


13516155 - Restu Wahyu Kartiko

## Petunjuk Penggunaan Program

Eksekusi tester.py
```
python3 tester.py
```

## Agglomerative
- Pertama melakukan inisiasi group sebanyak jumlah data.
- Menyimpan semua euclidean distance dari data dari setiap group yang terdefenisi sesuai dengan linkage yang digunakan.
- Langkah selanjutnya adalah melakukan join dengan group lain, perlu diperhatikan bahwa ketika join, nilai dari euclidean distance juga perlu diupdate karena jumlha dari group akan berkurang, melainkan ada jumlah data yang bertambah pada suatu group.
- Ulangi proses join sampai nilai group sama dengan nilai dari *k* yang merupakan banyak cluster yang ingin dibentuk.
- Ketika nilai *k* sama dengan banyak group maka simpan kedalam cluster, misalkan saja cluster mulai dari 1.

## DBScan
- Pertama melakukan inisiasi label cluster dengan nilai 0 untuk setiap data train, menandakan item belum di-*cluster*-kan
- Iterasi setiap data train. Apabila item data belum di-*cluster*-kan, hitung jumlah tetangga yang *reachable* dari item. Apabila jumlah tetangga < `min_pts`, labeli sementara item sebagai outlier.
- Apabila jumlah tetangga >= `min_pts`, buat sebuah cluster baru dan ekspansi cluster tersebut.
- Saat ekspansi cluster, cari semua tetangga yang *reachable* dari item saat ini. Untuk setiap tetangga apabila ia telah dilabeli sebagai *outlier* ubah labelnya menjadi label *cluster* saat ini. Untuk item yang belum dilabeli lakukan pencarian secara BFS tetangga-tetangga lainnya.

## KMeans
- Pertama dilakukan insiasi jumlah centroid sesuai input dan maksimal iterasi
- Tentukan sebanyak K centroid dari data dengan menggunakan algoritma random
- Lakukan iterasi sebanyak maksimal iterasi
- Untuk setiap iterasinya, akan di tentukan tiap titik akan masuk ke cluster mana sesuai dengan jarak centroid terdekat.
- Pada akhir iterasi, centroid-centroid tersebut akan di update. Nilai centroid menjadi nilai rata rata dari semual titik yang masuk cluster centroid tersebut