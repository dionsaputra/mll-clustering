# Clustering
Implementasi tiga cara clustering yaitu Agglomerative, DBScan, dan K-Means

**Kontributor**:
13516079 - Harry Setiawan Hamjaya

## Petunjuk Penggunaan Program

Eksekusi tester.py
```
python3 tester.py
```

## Agglometarive
1. Pertama melakukan inisiasi group sebanyak jumlah data.
2. Menyimpan semua euclidean distance dari data dari setiap group yang terdefenisi sesuai dengan linkage yang digunakan.
3. Langkah selanjutnya adalah melakukan join dengan group lain, perlu diperhatikan bahwa ketika join, nilai dari euclidean distance juga perlu diupdate karena jumlha dari group akan berkurang, melainkan ada jumlah data yang bertambah pada suatu group.
4. Ulangi proses join sampai nilai group sama dengan nilai dari *k* yang merupakan banyak cluster yang ingin dibentuk.
5. Ketika nilai *k* sama dengan banyak group maka simpan kedalam cluster, misalkan saja cluster mulai dari 1.
6. Untuk mencetak hasil cluster, maka ada fungsi ```get_all()```