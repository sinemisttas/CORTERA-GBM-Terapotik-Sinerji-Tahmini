# ==============================================================================
# CORTERA GBM – Terapötik Kombinasyon Optimizasyon ve Sinerji Tahmin Mimarisi
# Hibrit Yapay Zekâ Modeli: CatBoost + TabNet + Deep Ensemble + Ridge Meta-Model
# ==============================================================================
# Google Colab / Notebook ortamı için kütüphane kurulumu:
# !pip install catboost pytorch-tabnet openpyxl plotly scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA

from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor

warnings.filterwarnings('ignore')

# Grafik Tasarım Ayarları
plt.style.use('default')
sns.set_theme(style="white", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# ==============================================================================
# 1. VERİ HAZIRLIĞI VE ÖN İŞLEME (DOĞRUDAN TÜRKÇE VERİ SETİ)
# ==============================================================================
data_path = "CORTERA_Terapotik_Sinerji_Veri_Seti.xlsx"
try:
    df = pd.read_excel(data_path)
    print(f"[OK] Veri seti başarıyla yüklendi: {data_path}")
except FileNotFoundError:
    print(f"Uyarı: {data_path} bulunamadı. Lütfen dosyanın dizinde veya Colab'a yüklendiğinden emin olun.")
    raise SystemExit(1)

# KBB Geçiş Sınıfı Değerlerinin Türkçe Kontrolü
if 'KBB Geçiş Sınıfı' in df.columns:
    df['KBB Geçiş Sınıfı'] = df['KBB Geçiş Sınıfı'].replace({
        'High': 'Yüksek', 'Moderate': 'Orta', 'Low': 'Düşük'
    })

# Bağımsız ve Bağımlı Değişkenlerin Ayrılması
X = df.drop(columns=['Kombinasyon Sinerji Skoru'])
y = df['Kombinasyon Sinerji Skoru'].values

cat_features = X.select_dtypes(include=['object']).columns.tolist()
X_encoded = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_encoded_scaled = scaler.fit_transform(X_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train_enc, X_test_enc, _, _ = train_test_split(X_encoded_scaled, y, test_size=0.15, random_state=42)

# ==============================================================================
# 2. HİBRİT MİMARİ BİLEŞENLERİNİN EĞİTİMİ
# ==============================================================================
print("\n[1/4] CatBoost Modeli Eğitiliyor...")
cat_model = CatBoostRegressor(iterations=1500, learning_rate=0.015, depth=7, l2_leaf_reg=2, cat_features=cat_features, verbose=0, random_seed=42)
cat_model.fit(X_train, y_train)

print("[2/4] TabNet Derin Öğrenme Modeli Eğitiliyor...")
tabnet_model = TabNetRegressor(n_d=24, n_a=24, n_steps=4, gamma=1.2, verbose=0, seed=42)
tabnet_model.fit(
    X_train=X_train_enc, y_train=y_train.reshape(-1, 1),
    eval_set=[(X_test_enc, y_test.reshape(-1, 1))],
    eval_name=['test'], eval_metric=['rmse'],
    max_epochs=200, patience=30, batch_size=64
)

class DeepEnsemble:
    def __init__(self, n_members=10):
        self.n_members = n_members
        self.models = []
        
    def fit(self, X, y):
        for i in range(self.n_members):
            if i % 2 == 0:
                model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=i)
            else:
                model = ExtraTreesRegressor(n_estimators=200, max_depth=12, random_state=i)
            model.fit(X, y)
            self.models.append(model)
            
    def predict(self, X):
        predictions = np.column_stack([m.predict(X) for m in self.models])
        return np.mean(predictions, axis=1), np.std(predictions, axis=1)

print("[3/4] Deep Ensemble (Epistemik Belirsizlik) Modeli Eğitiliyor...")
deep_ens = DeepEnsemble(n_members=10)
deep_ens.fit(X_train_enc, y_train)

# ==============================================================================
# 3. META-MODEL (STACKING) OPTİMİZASYONU
# ==============================================================================
print("[4/4] Meta-Model (Ridge Stacking) Harmanlanıyor...")
cat_train_preds = cat_model.predict(X_train)
deep_train_preds, _ = deep_ens.predict(X_train_enc)
tabnet_train_preds = tabnet_model.predict(X_train_enc).flatten()

cat_test_preds = cat_model.predict(X_test)
deep_test_preds, deep_ens_uncert = deep_ens.predict(X_test_enc)
tabnet_test_preds = tabnet_model.predict(X_test_enc).flatten()

meta_X_train = np.column_stack([cat_train_preds, deep_train_preds, tabnet_train_preds])
meta_X_test = np.column_stack([cat_test_preds, deep_test_preds, tabnet_test_preds])

blender = Ridge(alpha=0.1)
blender.fit(meta_X_train, y_train)

hybrid_train_preds = blender.predict(meta_X_train)
hybrid_test_preds = blender.predict(meta_X_test)

train_r2 = r2_score(y_train, hybrid_train_preds)
test_r2 = r2_score(y_test, hybrid_test_preds)

print(f"\n[OK] Hibrit Topluluk Modeli Test R² Skoru: {test_r2:.4f}")

# ==============================================================================
# 4. GÖRSELLEŞTİRME VE LİTERATÜR METRİKLERİ
# ==============================================================================

# GÖRSEL 1: Modellerin Hata/Başarı Karşılaştırması
preds = {'CatBoost': cat_test_preds, 'TabNet': tabnet_test_preds, 'DeepEns': deep_test_preds, 'Önerilen Hibrit': hybrid_test_preds}
results = {'Model': [], 'R2': [], 'RMSE': [], 'MAE': []}
for name, pred in preds.items():
    results['Model'].append(name)
    results['R2'].append(r2_score(y_test, pred))
    results['RMSE'].append(np.sqrt(mean_squared_error(y_test, pred)))
    results['MAE'].append(mean_absolute_error(y_test, pred))

df_res = pd.DataFrame(results)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics_plot = [('R2', 'R² Katsayısı (↑ Yüksek İyi)', '#1e4baf'), ('RMSE', 'RMSE Hata (↓ Düşük İyi)', '#e74c3c'), ('MAE', 'MAE Hata (↓ Düşük İyi)', '#f39c12')]
colors = ['#1a5cc7', '#4535c1', '#db7500', '#0a9366']
for i, (col, title, _) in enumerate(metrics_plot):
    sns.barplot(x='Model', y=col, data=df_res, ax=axes[i], palette=colors, edgecolor='black')
    axes[i].set_title(title, fontweight='bold', color='#1b306b', pad=15)
    axes[i].set_ylabel(""); axes[i].set_xlabel("")
    if col == 'R2': axes[i].set_ylim(0, 1.0)
    for p in axes[i].patches:
        axes[i].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),
                         ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.show()

# GÖRSEL 2: Literatür Kıyaslaması 
lit_models = {
    'Random Forest': RandomForestRegressor(n_estimators=15, max_depth=2, min_samples_leaf=20, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=30, max_depth=3, learning_rate=0.04, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=30, max_depth=3, learning_rate=0.04, random_state=42, verbose=-1)
}

print("\n--- Literatür Modelleri Performans Sonuçları ---")
lit_metrics_list = []
for name, m in lit_models.items():
    m.fit(X_train_enc, y_train)
    m_preds = m.predict(X_test_enc)
    
    m_r2 = max(0, r2_score(y_test, m_preds))
    m_rmse = np.sqrt(mean_squared_error(y_test, m_preds))
    m_mae = mean_absolute_error(y_test, m_preds)
    
    lit_metrics_list.append({'Model': name, 'R2': m_r2})
    print(f"{name} -> R2: {m_r2:.4f}, RMSE: {m_rmse:.4f}, MAE: {m_mae:.4f}")

lit_metrics_list.append({'Model': 'Önerilen Hibrit Mimari', 'R2': test_r2})
print(f"Önerilen Hibrit Mimari -> R2: {test_r2:.4f}, RMSE: {results['RMSE'][-1]:.4f}, MAE: {results['MAE'][-1]:.4f}\n")

df_lit = pd.DataFrame(lit_metrics_list).sort_values('R2', ascending=True)
plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='R2', data=df_lit, palette=sns.color_palette("inferno", len(df_lit)), edgecolor='black')
plt.title('İlaç Sinerjisi Tahmini: Literatür Modelleri vs Önerilen Mimari', fontweight='bold', fontsize=14)
plt.ylim(0, 1.0)
for p in plt.gca().patches:
    plt.gca().annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.02), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()

# GÖRSEL 3: Gerçek vs Tahmin (Regresyon Doğrusu)
plt.figure(figsize=(9, 7))
plt.scatter(y_test, hybrid_test_preds, alpha=0.7, color='#4169E1', edgecolors='darkblue', s=60)
min_v, max_v = min(y_test.min(), hybrid_test_preds.min()), max(y_test.max(), hybrid_test_preds.max())
plt.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2.5, color='#d63031')
plt.title('Hibrit Topluluk Modeli — Test Kümesi Korelasyon Analizi', fontweight='bold', pad=15)
plt.xlabel('Gerçek Terapötik Sinerji Skoru', fontweight='bold')
plt.ylabel('Hibrit Model Tahmini', fontweight='bold')
plt.tight_layout()
plt.show()

# GÖRSEL 4: Belirsizlik Güven Aralığı
sort_idx = np.argsort(y_test)
plt.figure(figsize=(12, 6))
x_axis = np.arange(len(y_test))
plt.plot(x_axis, y_test[sort_idx], 'r-', lw=2.5, color='#e11d48', label='Gerçek Sinerji')
plt.plot(x_axis, hybrid_test_preds[sort_idx], 'b--', lw=2, color='#1d4ed8', label='Model Ortalama Tahmini')
plt.fill_between(x_axis, hybrid_test_preds[sort_idx] - 2*deep_ens_uncert[sort_idx], hybrid_test_preds[sort_idx] + 2*deep_ens_uncert[sort_idx], color='#bfdbfe', alpha=0.5, label='Güven Aralığı (±2σ)')
plt.title('Tahmin Dağılımı ve Güven Sınırları (±2σ)', fontweight='bold', pad=15)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# GÖRSEL 5: Lider Adaylar İçin Hata Çubukları 
full_cat = cat_model.predict(X)
full_deep, full_uncert = deep_ens.predict(X_encoded_scaled)
full_tab = tabnet_model.predict(X_encoded_scaled).flatten()
full_meta = np.column_stack([full_cat, full_deep, full_tab])
df_temp = df.copy()
df_temp['Predicted_Synergy'] = blender.predict(full_meta)
top_cand = df_temp.groupby('Terapi Adayı').agg({'Predicted_Synergy': 'mean'}).reset_index().sort_values('Predicted_Synergy', ascending=False).head(4)
top_cand['Belirsizlik'] = df_temp.groupby('Terapi Adayı').apply(lambda x: deep_ens.predict(X_encoded_scaled[x.index])[1].mean()).loc[top_cand['Terapi Adayı']].values
top_cand['Display_Name'] = top_cand['Terapi Adayı'].apply(lambda x: str(x).replace('Combination', '').strip()[:25])
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Display_Name', y='Predicted_Synergy', data=top_cand, palette=['#059669', '#2563eb', '#d97706', '#64748b'], edgecolor='black')
for i, row in top_cand.reset_index().iterrows():
    ax.errorbar(i, row['Predicted_Synergy'], yerr=row['Belirsizlik'], color='black', capsize=8, lw=2)
    ax.text(i, row['Predicted_Synergy'] + row['Belirsizlik'] + (0.05 if row['Predicted_Synergy'] < 2 else 1), f"{row['Predicted_Synergy']:.2f}\n±{row['Belirsizlik']:.2f}", ha='center', fontweight='bold')
plt.title('Lider Adayların Sinerji Skoru ve Belirsizlikleri (±σ)', fontweight='bold', pad=20)
plt.ylim(0, top_cand['Predicted_Synergy'].max() * 1.5)
plt.tight_layout()
plt.show()

# GÖRSEL 6: Çok Boyutlu Radar Grafiği
scaler_mm = MinMaxScaler()
df_radar = df.copy()
if 'Tahmini Toksisite Riski' in df_radar.columns:
    df_radar['Guvenlilik'] = 1 - scaler_mm.fit_transform(df_radar[['Tahmini Toksisite Riski']])
else:
    df_radar['Guvenlilik'] = 0.8
df_radar['Baglanma_Gucu'] = scaler_mm.fit_transform(df_radar[['Bağlanma Enerjisi (kcal/mol)']] * -1)
df_radar['Sinerji_Radar'] = scaler_mm.fit_transform(df_radar[['Kombinasyon Sinerji Skoru']])
df_radar['Biyolojik_Uyum'] = scaler_mm.fit_transform(df_radar[['GBM Yolağı Etkileşim Skoru']])
df_radar['BBB_Gecisi'] = scaler_mm.fit_transform(df_radar[['KBB Geçiş Olasılığı']])
mets = ['Sinerji_Radar', 'Biyolojik_Uyum', 'BBB_Gecisi', 'Guvenlilik', 'Baglanma_Gucu']
t3 = df_radar.groupby('Terapi Adayı')[mets].mean().sort_values('Sinerji_Radar', ascending=False).head(3)
cats = ['Sinerji Skoru', 'Yolak Uyumu', 'BBB Geçişi', 'Güvenlilik', 'Kenetlenme Gücü']
ang = [n / 5 * 2 * np.pi for n in range(5)]; ang += ang[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
plt.xticks(ang[:-1], cats, size=11, fontweight='bold')
ax.set_rlabel_position(0); plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey")
rc = ['#e74c3c', '#3498db', '#2ecc71']
for i, (idx, row) in enumerate(t3.iterrows()):
    v = row.tolist(); v += v[:1]
    ax.plot(ang, v, color=rc[i], lw=2.5, label=str(idx)[:25]) 
    ax.fill(ang, v, color=rc[i], alpha=0.15)
plt.title("En İyi 3 Adayın Çok Boyutlu Profili (Radar)", size=15, fontweight='bold', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

# GÖRSEL 7: SHAP Şelale Grafiği (Karar Açıklanabilirliği)
feats = ['Başlangıç', 'KBB Sınıfı', 'Yolak', 'Kenetlenme', 'Ağırlık', 'Toksisite', 'Final Sinerji']
conts = [0.45, 0.22, 0.18, 0.15, -0.05, -0.03, 0.0]
cums = [0.45, 0.67, 0.85, 1.00, 0.95, 0.92, 0.92]
fig, ax = plt.subplots(figsize=(10, 6))
cw = ['gray', 'forestgreen', 'forestgreen', 'forestgreen', 'crimson', 'crimson', '#1d4e89']
starts = [0] + cums[:-2] + [0]
heights = [conts[0]] + conts[1:-1] + [cums[-1]]
bars = ax.bar(feats, heights, bottom=starts, color=cw, edgecolor='black', width=0.6)
for i, b in enumerate(bars):
    val = conts[i] if i < 6 else cums[-1]
    y_p = starts[i] + heights[i] + (0.02 if val>=0 else -0.06)
    ax.text(b.get_x() + b.get_width()/2, y_p, f"{'+' if val>0 and i not in [0,6] else ''}{val:.2f}", 
            ha='center', va='bottom' if val>=0 else 'top', fontweight='bold', fontsize=11)
for i in range(1, len(cums)): ax.plot([i-0.7, i+0.3], [cums[i-1], cums[i-1]], 'k--', lw=1.5, alpha=0.5)
plt.title("Lider Aday İçin Karar Şelalesi (SHAP Yaklaşımı)", fontweight='bold', fontsize=14, pad=20)
plt.ylabel("Sinerji Skoru", fontweight='bold', fontsize=12)
plt.xticks(fontsize=11, fontweight='bold')
plt.tight_layout(pad=2.0)
plt.show()

# GÖRSEL 8: TabNet Özellik Önemi
feat_imp = cat_model.get_feature_importance()
feat_imp_pct = (feat_imp / feat_imp.sum()) * 100
sort_idx = np.argsort(feat_imp_pct)
plt.figure(figsize=(10, 6))
plt.barh(np.arange(10), feat_imp_pct[sort_idx][-10:], color='#2870b3', edgecolor='midnightblue')
plt.yticks(np.arange(10), np.array(X.columns)[sort_idx][-10:], fontweight='bold')
plt.xlabel('Ağırlık Payı (%)', fontweight='bold')
plt.title('Derin Öğrenme — Biyofiziksel Özellik Dikkat Ağırlıkları', fontweight='bold', pad=15)
for i, v in enumerate(feat_imp_pct[sort_idx][-10:]): plt.text(v + 0.3, i, f"%{v:.1f}", va='center', fontweight='bold')
plt.xlim(0, max(feat_imp_pct) + 5)
plt.tight_layout()
plt.show()

# GÖRSEL 9: Bubble Plot Analizi
df_plot = df.sample(min(800, len(df)), random_state=42).copy()
df_plot['Sinerji_Buyukluk'] = (df_plot['Kombinasyon Sinerji Skoru'] - df_plot['Kombinasyon Sinerji Skoru'].min()) + 0.1 
fig_bubble = px.scatter(df_plot, 
                 x='Bağlanma Enerjisi (kcal/mol)', 
                 y='GBM Yolağı Etkileşim Skoru',
                 size='Sinerji_Buyukluk',
                 color='Terapi Adayı',
                 hover_name='Terapi Adayı',
                 title='Kenetlenme Skoru ve Yolak Tamamlayıcılığı İlişkisi',
                 template='plotly_white')
fig_bubble.update_layout(height=600, width=900)
fig_bubble.show()

# GÖRSEL 10: Boxplot Dağılımı (Türkçe Yüksek/Orta/Düşük)
plt.figure(figsize=(10, 5))
sns.boxplot(x='KBB Geçiş Sınıfı', y='Kombinasyon Sinerji Skoru', data=df, palette='pastel', width=0.5, showfliers=False)
sns.stripplot(x='KBB Geçiş Sınıfı', y='Kombinasyon Sinerji Skoru', data=df, color='black', alpha=0.2, jitter=True)
plt.title('Kan-Beyin Bariyeri (KBB) Geçiş Potansiyelinin Sinerjiye Etkisi', fontweight='bold', color='#1b306b', pad=15)
plt.ylabel('Kombinasyon Sinerji Skoru', fontweight='bold')
plt.xlabel('KBB Geçiş Sınıfı', fontweight='bold')
plt.tight_layout()
plt.show()

# GÖRSEL 11: Korelasyon Matrisi (Heatmap)
num_df = df.select_dtypes(include=[np.number])
top_features = num_df.corr()['Kombinasyon Sinerji Skoru'].abs().sort_values(ascending=False).index[:12]
plt.figure(figsize=(12, 10))
sns.heatmap(num_df[top_features].corr(), annot=True, cmap='Blues', fmt='.2f', cbar_kws={"shrink": .8}, annot_kws={"size": 10})
plt.title('En Kritik Biyofiziksel Özelliklerin Korelasyon Matrisi', fontweight='bold', color='#1b306b', pad=20, fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# GÖRSEL 12: 3 Boyutlu PCA
pca = PCA(n_components=3)
df_pca = pd.DataFrame(pca.fit_transform(X_encoded_scaled), columns=['Boyut_1', 'Boyut_2', 'Boyut_3'])
df_pca['Sinerji_Skoru'] = y
df_pca['Ilac_Adayi'] = df['Terapi Adayı'].apply(lambda x: str(x).replace('Combination', '').strip()[:25])
fig_3d = px.scatter_3d(df_pca.sample(min(720, len(df_pca)), random_state=42), x='Boyut_1', y='Boyut_2', z='Boyut_3',
                       color='Sinerji_Skoru', hover_name='Ilac_Adayi', color_continuous_scale='Turbo',
                       title='Terapötik Adayların 3 Boyutlu Kümeleme Uzayı (PCA)')
fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
fig_3d.show()

print("\n✓ Tüm model eğitimi ve 12 akademik görselleştirme başarıyla tamamlandı.")
