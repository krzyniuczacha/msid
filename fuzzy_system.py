import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
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

# --- GLUKOZA ---
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

# --- GLUKOZA ---
for label, mf in glucose.terms.items():
    axes[0].plot(glukoza_zakres, mf.mf, label=label)
axes[0].set_title('Glukoza [mg/dL]')
axes[0].set_xlabel('Poziom glukozy')
axes[0].set_ylabel('Przynależność μ')
axes[0].legend()

# --- BMI ---
for label, mf in bmi_var.terms.items():
    axes[1].plot(bmi_zakres, mf.mf, label=label)
axes[1].set_title('BMI [kg/m²]')
axes[1].set_xlabel('Wartość BMI')
axes[1].set_ylabel('Przynależność μ')
axes[1].legend()

# --- RYZYKO ---
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
