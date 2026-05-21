import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

from fuzzy_module import run_fuzzy_experiments
from ml_module import run_ml_experiments

warnings.filterwarnings('ignore')

# ==============================================
# 1. PRZYGOTOWANIE DANYCH
# ==============================================
print("Wczytywanie i przygotowywanie danych...")
df = pd.read_csv('diabetes.csv')

# Zastąpienie zer (medycznych błędów) medianą
for col in ['Glucose', 'BMI', 'BloodPressure', 'Insulin', 'SkinThickness']:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tworzymy ostateczną tabelę testową z przyciętymi skrajnościami dla systemu rozmytego
X_test_clean = X_test.copy()
X_test_clean['BMI_c'] = X_test_clean['BMI'].clip(0, 70)
X_test_clean['Age_c'] = X_test_clean['Age'].clip(0, 80)

print(f"Dane gotowe. Zbiór testowy: {len(X_test_clean)} pacjentów.\n")

# ==============================================
# 2. URUCHOMIENIE EKSPERYMENTÓW
# ==============================================
# Zbieramy wyniki od systemu rozmytego
exp1, exp2, exp3 = run_fuzzy_experiments(X_test_clean, y_test)

# Zbieramy wyniki od uczenia maszynowego
ml_res = run_ml_experiments(X_train, X_test, y_train, y_test)


# ==============================================
# 3. TABELA ZBIORCZA W KONSOLI
# ==============================================
print("=" * 70)
print(f"{'System':<28} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'ms':>8}")
print("-" * 70)

all_results = {
    '── FUZZY: 1 atrybut':       exp1['1 atrybut\n(glukoza)'],
    '── FUZZY: 2 atrybuty':      exp1['2 atrybuty\n(glukoza+BMI)'],
    '── FUZZY: 3 atrybuty':      exp1['3 atrybuty\n(glukoza+BMI+wiek)'],
    '── FUZZY: 2 MF':            exp2['2 MF\n(binarne)'],
    '── FUZZY: 4 MF':            exp2['4 MF\n(granularne)'],
    '── FUZZY: trimf':           exp3['trimf'],
    '── FUZZY: trapmf':          exp3['trapmf'],
    '── FUZZY: gaussmf':         exp3['gaussmf'],
    '── ML: Logistic Regression':ml_res['Logistic\nRegression'],
    '── ML: Decision Tree':      ml_res['Decision\nTree'],
    '── ML: Random Forest':      ml_res['Random\nForest'],
}

for name, m in all_results.items():
    print(f"{name:<28} {m['accuracy']:>7.3f} {m['precision']:>7.3f} "
          f"{m['recall']:>7.3f} {m['f1']:>7.3f} {m['time_ms']:>8.1f}")
print("=" * 70)


# ==============================================
# 4. WIZUALIZACJA I ZAPIS WYKRESÓW
# ==============================================
print("\nGenerowanie wykresów...")

PALETTE = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0']

def bar_comparison(ax, results, metric, title, colors=None):
    names = list(results.keys())
    vals  = [results[k][metric] for k in names]
    if colors is None:
        colors = PALETTE[:len(names)]
    bars = ax.bar(names, vals, color=colors, edgecolor='white', linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(metric.capitalize())
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Wyniki eksperymentów — system rozmyty i metody ML', fontsize=15, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

# Wiersz 1
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
bar_comparison(ax1, exp1, 'accuracy', 'Exp 1: Liczba atrybutów\nDokładność')
bar_comparison(ax2, exp1, 'f1', 'Exp 1: Liczba atrybutów\nF1-score')

names1 = list(exp1.keys())
times1 = [exp1[k]['time_ms'] for k in names1]
ax3.bar(names1, times1, color=PALETTE[:len(names1)], edgecolor='white')
ax3.set_title('Exp 1: Liczba atrybutów\nCzas klasyfikacji [ms]', fontsize=11, fontweight='bold')
ax3.set_ylabel('Czas [ms]')
ax3.spines[['top', 'right']].set_visible(False)
ax3.tick_params(axis='x', labelsize=8)

# Wiersz 2
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])
bar_comparison(ax4, exp2, 'accuracy', 'Exp 2: Liczba MF\nDokładność')
bar_comparison(ax5, exp2, 'f1', 'Exp 2: Liczba MF\nF1-score')
bar_comparison(ax6, exp3, 'accuracy', 'Exp 3: Typ funkcji MF\nDokładność', colors=['#2196F3', '#FF5722', '#009688'])

# Wiersz 3
ax7 = fig.add_subplot(gs[2, 0])
ax8 = fig.add_subplot(gs[2, 1])
ax9 = fig.add_subplot(gs[2, 2])
bar_comparison(ax7, exp3, 'f1', 'Exp 3: Typ funkcji MF\nF1-score', colors=['#2196F3', '#FF5722', '#009688'])
bar_comparison(ax8, ml_res, 'accuracy', 'ML: Dokładność')
bar_comparison(ax9, ml_res, 'f1', 'ML: F1-score')

plt.savefig('fuzzy_experiments.png', dpi=130, bbox_inches='tight')
plt.close()
print("Gotowe! Wykres zapisano w pliku 'fuzzy_experiments.png'.")