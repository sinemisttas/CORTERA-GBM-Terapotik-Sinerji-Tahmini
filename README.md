# CORTERA – Glioblastoma (GBM) Hibrit Yapay Zekâ ve Sinerji Tahmin Algoritması

Bu repository, Glioblastoma (GBM) tedavisinde terapötik kombinasyonların sinerji skorunu, belirsizlik düzeylerini ve biyofiziksel özelliklerini tahmin eden hibrit yapay zekâ modeline (CatBoost + TabNet + Deep Ensemble + Ridge Stacking) ve 12 akademik görselleştirme analizine ait tüm Python kodlarını içermektedir.

---

## 🧬 Proje ve Model Mimarisi

Geliştirilen hibrit yapay zekâ mimarisi 4 temel bileşenden oluşmaktadır:

1. **CatBoost Regresyonu:** Kategori ve sayısal özellikler üzerinde yüksek performanslı gradient boosting tahmini.
2. **PyTorch TabNet:** Tabüler veriler için dikkat (attention) mekanizmalı derin öğrenme mimarisi.
3. **Deep Ensemble (Random Forest + Extra Trees Topluluğu):** Model tahminlerindeki epistemik belirsizliği ($\sigma$) ve güven aralıklarını hesaplayan 10 üyeli ensemble mimari.
4. **Meta-Model (Ridge Stacking):** CatBoost, TabNet ve Deep Ensemble tahminlerini Ridge regresyonu ile harmanlayan meta-model katmanı.

---

## 📊 12 Akademik Görselleştirme ve Analiz Modülü

Kod scripti otomatik olarak aşağıdaki 12 grafik ve akademik analizi üretir:
- **Görsel 1:** Modellerin Performans/Hata Karşılaştırması (R², RMSE, MAE)
- **Görsel 2:** Literatür Modelleri vs Önerilen Hibrit Mimari (Random Forest, XGBoost, LightGBM)
- **Görsel 3:** Gerçek vs Tahmin Korelasyon Analizi
- **Görsel 4:** Belirsizlik Güven Aralığı Dağılımı ($\pm 2\sigma$)
- **Görsel 5:** Lider Adaylar İçin Hata Çubukları ($\pm\sigma$)
- **Görsel 6:** En İyi 3 Adayın Çok Boyutlu Radar Profili
- **Görsel 7:** SHAP Karar Şelalesi (XAI Açıklanabilir Yapay Zekâ)
- **Görsel 8:** Derin Öğrenme / TabNet Biyofiziksel Özellik Dikkat Ağırlıkları
- **Görsel 9:** Kenetlenme Skoru ve Yolak Tamamlayıcılığı Bubble Plot (Plotly)
- **Görsel 10:** Kan-Beyin Bariyeri (BBB) Geçiş Potansiyeli Boxplot Analizi
- **Görsel 11:** En Kritik Biyofiziksel Özelliklerin Korelasyon Matrisi (Heatmap)
- **Görsel 12:** Terapötik Adayların 3 Boyutlu Kümeleme Uzayı (3D PCA Plotly)

---

## 📁 Dosya Yapısı

```
CORTERA_Github_Paketi/
├── cortera_pipeline.py     # Tam kod scripti (Veri işleme, model eğitimi, 12 görselleştirme)
├── requirements.txt        # Gerekli Python paketleri
├── README.md               # Proje ve model dokümantasyonu
└── data/                   # Excel veri setleri (CORTERA_Terapotik_Sinerji_Veri_Seti.xlsx)
```

---

## 🚀 Kurulum ve Çalıştırma

```bash
# 1. Kütüphaneleri Yükleyin
pip install -r requirements.txt

# 2. Modeli Çalıştırın
python cortera_pipeline.py
```

---

## 📌 Gizlilik

Bu repository **özel (private)** bir proje arşivi olup yalnızca TEKNOFEST ve jüri değerlendirme kurulunun incelemesine sunulmuştur.
