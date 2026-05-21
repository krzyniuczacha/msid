import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split


df = pd.read_csv('diabetes.csv')

# Zera w Glucose i BMI to brakujące dane (medycznie niemożliwe)
# Zastępujemy je medianą kolumny, żeby nie psuły systemu
for col in ['Glucose', 'BMI']:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Zbiór testowy: {len(X_test)} pacjentów  "
      f"(zdrowi: {(y_test==0).sum()}, chorzy: {(y_test==1).sum()})\n")


glukoza_zakres = np.arange(0, 201, 1)   # 0–200 mg/dL
bmi_zakres     = np.arange(0, 71,  1)   # 0–70 kg/m²
ryzyko_zakres  = np.arange(0, 1.01, 0.01)  # 0–1 (0 = zdrowy, 1 = chory)

glucose = ctrl.Antecedent(glukoza_zakres, 'glucose')
bmi_var = ctrl.Antecedent(bmi_zakres,    'bmi')
risk    = ctrl.Consequent(ryzyko_zakres, 'risk')

# FUNKCJE PRZYNALEŻNOŚCI
# Progi oparte na wytycznych klinicznych ADA (glukoza) i WHO (BMI)

#GLUKOZA
# low:    < 100 mg/dL  -> normalna glikemia (ADA: poniżej progu stanu przedcukrzycowego)
# normal: 80–126 mg/dL -> stan przedcukrzycowy (ADA: 100–125) z marginesem rozmycia
# high:   > 110 mg/dL  -> cukrzyca (ADA: ≥ 126) z marginesem rozmycia
glucose['low']    = fuzz.trimf(glukoza_zakres, [0,   0,   100])
glucose['normal'] = fuzz.trimf(glukoza_zakres, [80,  110, 126])
glucose['high']   = fuzz.trimf(glukoza_zakres, [110, 200, 200])

#BMI
# normal:     < 25     -> zdrowa masa ciała (WHO)
# overweight: 22–32    -> nadwaga (WHO: 25–29.9) z marginesem rozmycia
# obese:      > 28     -> otyłość (WHO: ≥ 30) z marginesem rozmycia
bmi_var['normal']     = fuzz.trimf(bmi_zakres, [0,   0,   25])
bmi_var['overweight'] = fuzz.trimf(bmi_zakres, [22,  27,  32])
bmi_var['obese']      = fuzz.trimf(bmi_zakres, [28,  70,  70])

#RYZYKO CUKRZYCY (wyjście)
risk['low']    = fuzz.trimf(ryzyko_zakres, [0.0, 0.0, 0.4])
risk['medium'] = fuzz.trimf(ryzyko_zakres, [0.2, 0.5, 0.8])
risk['high']   = fuzz.trimf(ryzyko_zakres, [0.6, 1.0, 1.0])

# WIZUALIZACJA FUNKCJI PRZYNALEŻNOŚCI - wykresy dla glukozy, BMI i ryzyka do membership_functions.png

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

#GLUKOZA
for label, mf in glucose.terms.items():
    axes[0].plot(glukoza_zakres, mf.mf, label=label)
axes[0].set_title('Glukoza [mg/dL]')
axes[0].set_xlabel('Poziom glukozy')
axes[0].set_ylabel('Przynależność μ')
axes[0].legend()

#BMI
for label, mf in bmi_var.terms.items():
    axes[1].plot(bmi_zakres, mf.mf, label=label)
axes[1].set_title('BMI [kg/m²]')
axes[1].set_xlabel('Wartość BMI')
axes[1].set_ylabel('Przynależność μ')
axes[1].legend()

#RYZYKO
for label, mf in risk.terms.items():
    axes[2].plot(ryzyko_zakres, mf.mf, label=label)
axes[2].set_title('Ryzyko cukrzycy')
axes[2].set_xlabel('Poziom ryzyka')
axes[2].set_ylabel('Przynależność μ')
axes[2].legend()

plt.suptitle('Funkcje przynależności - system rozmyty', fontsize=13)
plt.tight_layout()
plt.savefig('membership_functions.png', dpi=120, bbox_inches='tight')
plt.show()
plt.close()

#REGUŁY ROZMYTE
#Każda reguła: IF (warunek) THEN (wniosek)
rule1 = ctrl.Rule(glucose['high']   & bmi_var['obese'],      risk['high'])
rule2 = ctrl.Rule(glucose['high']   & bmi_var['overweight'], risk['high'])
rule3 = ctrl.Rule(glucose['high']   & bmi_var['normal'],     risk['medium'])
rule4 = ctrl.Rule(glucose['normal'] & bmi_var['obese'],      risk['medium'])
rule5 = ctrl.Rule(glucose['normal'] & bmi_var['overweight'], risk['medium'])
rule6 = ctrl.Rule(glucose['normal'] & bmi_var['normal'],     risk['low'])
rule7 = ctrl.Rule(glucose['low'],                            risk['low'])

print("Baza reguł:")
print("  R1: IF glukoza=wysoka   AND bmi=otyłość   THEN ryzyko=wysokie")
print("  R2: IF glukoza=wysoka   AND bmi=nadwaga   THEN ryzyko=wysokie")
print("  R3: IF glukoza=wysoka   AND bmi=normalne  THEN ryzyko=średnie")
print("  R4: IF glukoza=normalna AND bmi=otyłość   THEN ryzyko=średnie")
print("  R5: IF glukoza=normalna AND bmi=nadwaga   THEN ryzyko=średnie")
print("  R6: IF glukoza=normalna AND bmi=normalne  THEN ryzyko=niskie")
print("  R7: IF glukoza=niska                      THEN ryzyko=niskie")
print()

# budowa systemu rozmytego i symulacja
system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
sim = ctrl.ControlSystemSimulation(system)

def predict_risk(glucose_val: float, bmi_val: float) -> float:
    """
    Oblicza ryzyko cukrzycy dla danego pacjenta.
    Zwraca wartość z zakresu [0, 1].
    W razie błędu (np. brak aktywnych reguł) zwraca 0.5 (niepewność).
    """
    try:
        sim.input['glucose'] = float(np.clip(glucose_val, 0, 200))
        sim.input['bmi'] = float(np.clip(bmi_val, 0, 70))
        sim.compute()
        return sim.output['risk']
    except Exception:
        return 0.5  # neutralna wartość gdy reguły nie pokrywają przypadku


#klasyfikacja na podstawie surowych wyników ryzyka: > 0.5 -> cukrzyca, ≤ 0.5 -> zdrowy
THRESHOLD = 0.7  # próg decyzyjny: ryzyko > 0.7 -> cukrzyca

start = time.time()
raw_scores = [predict_risk(row['Glucose'], row['BMI'])
              for _, row in X_test.iterrows()]
elapsed_ms = (time.time() - start) * 1000

predictions = [1 if r > THRESHOLD else 0 for r in raw_scores]

# metryki oceny
acc = accuracy_score(y_test, predictions)
prec = precision_score(y_test, predictions, zero_division=0)
rec = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
cm = confusion_matrix(y_test, predictions)

print("=" * 50)
print("        WYNIKI - SYSTEM ROZMYTY")
print("=" * 50)
print(f"  Dokładność  (accuracy) : {acc:.4f}  ({acc * 100:.1f}%)")
print(f"  Precyzja   (precision) : {prec:.4f}")
print(f"  Czułość       (recall) : {rec:.4f}")
print(f"  F1-score               : {f1:.4f}")
print(f"  Czas klasyfikacji      : {elapsed_ms:.1f} ms  "
      f"({elapsed_ms / len(X_test):.2f} ms/pacjent)")
print()
print("Macierz pomyłek:")
print(f"  Prawdziwie Zdrowi (TN) : {cm[0, 0]}")
print(f"  Fałszywie Chorzy  (FP) : {cm[0, 1]}")
print(f"  Fałszywie Zdrowi  (FN) : {cm[1, 0]}")
print(f"  Prawdziwie Chorzy (TP) : {cm[1, 1]}")
print()
print("Pełny raport:")
print(classification_report(y_test, predictions,
                            target_names=['Zdrowy (0)', 'Cukrzyca (1)']))


#WIZUALIZACJA WYNIKÓW - histogram surowych wyników ryzyka
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Histogram surowych wartości ryzyka z podziałem na faktyczne klasy
scores_arr = np.array(raw_scores)
labels_arr = np.array(y_test)

ax1.hist(scores_arr[labels_arr == 0], bins=20, alpha=0.7,
         color='steelblue', label='Faktycznie zdrowi')
ax1.hist(scores_arr[labels_arr == 1], bins=20, alpha=0.7,
         color='tomato', label='Faktycznie chorzy')
ax1.axvline(THRESHOLD, color='black', linestyle='--', label=f'Próg = {THRESHOLD}')
ax1.set_xlabel('Wyjście systemu rozmytego (ryzyko)')
ax1.set_ylabel('Liczba pacjentów')
ax1.set_title('Rozkład wyników ryzyka na zbiorze testowym')
ax1.legend()

# Macierz pomyłek jako mapa ciepła
im = ax2.imshow(cm, cmap='Blues')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Pred: Zdrowy', 'Pred: Chory'])
ax2.set_yticklabels(['Fakt: Zdrowy', 'Fakt: Chory'])
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm[i, j]), ha='center', va='center',
                 fontsize=16, color='white' if cm[i, j] > cm.max() / 2 else 'black')
ax2.set_title('Macierz pomyłek')
plt.colorbar(im, ax=ax2)

plt.suptitle('Ewaluacja systemu rozmytego — cukrzyca', fontsize=13)
plt.tight_layout()
plt.savefig('fuzzy_results.png', dpi=120, bbox_inches='tight')
plt.close()
print("Wykres wyników zapisany -> fuzzy_results.png")