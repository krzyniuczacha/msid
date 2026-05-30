import time
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def metrics(y_true, y_pred, y_prob, elapsed_ms):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_prob),
        'time_ms': elapsed_ms,
    }


def build_and_evaluate(rules, inputs_dict, X_test_clean, y_test, threshold=0.5):
    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)

    def predict(row):
        try:
            for var_name, col_name in inputs_dict.items():
                sim.input[var_name] = float(row[col_name])
            sim.compute()
            return sim.output['risk']
        except Exception:
            return 0.5

    t0 = time.time()
    raw = [predict(row) for _, row in X_test_clean.iterrows()]
    elapsed = (time.time() - t0) * 1000

    preds = [1 if r > threshold else 0 for r in raw]
    m = metrics(y_test, preds, raw, elapsed)
    m['y_pred'] = preds
    m['y_prob'] = raw
    return m


def run_fuzzy_experiments(X_test_clean, y_test):
    print("Uruchamianie systemów rozmytych (Fuzzy Logic)...")

    g_rng = np.arange(0, 201, 1)
    b_rng = np.arange(0, 71, 1)
    a_rng = np.arange(0, 82, 1)
    r_rng = np.arange(0, 1.01, 0.01)

    exp1_results = {}
    exp2_results = {}
    exp3_results = {}

    # ================= EKSPERYMENT 1 =================
    # Wariant A (1 atrybut)
    g1 = ctrl.Antecedent(g_rng, 'glucose')
    r1 = ctrl.Consequent(r_rng, 'risk')
    g1['low'] = fuzz.trimf(g_rng, [0, 0, 100])
    g1['normal'] = fuzz.trimf(g_rng, [80, 110, 126])
    g1['high'] = fuzz.trimf(g_rng, [110, 200, 200])
    r1['low'] = fuzz.trimf(r_rng, [0.0, 0.0, 0.4])
    r1['medium'] = fuzz.trimf(r_rng, [0.2, 0.5, 0.8])
    r1['high'] = fuzz.trimf(r_rng, [0.6, 1.0, 1.0])
    rules_1 = [
        ctrl.Rule(g1['high'], r1['high']),
        ctrl.Rule(g1['normal'], r1['medium']),
        ctrl.Rule(g1['low'], r1['low']),
    ]
    exp1_results['1 atrybut\n(glukoza)'] = build_and_evaluate(rules_1, {'glucose': 'Glucose'}, X_test_clean, y_test,
                                                              0.5)

    # Wariant B (2 atrybuty)
    g2 = ctrl.Antecedent(g_rng, 'glucose')
    b2 = ctrl.Antecedent(b_rng, 'bmi')
    r2 = ctrl.Consequent(r_rng, 'risk')
    g2['low'] = fuzz.trimf(g_rng, [0, 0, 100])
    g2['normal'] = fuzz.trimf(g_rng, [80, 110, 126])
    g2['high'] = fuzz.trimf(g_rng, [110, 200, 200])
    b2['normal'] = fuzz.trimf(b_rng, [0, 0, 25])
    b2['overweight'] = fuzz.trimf(b_rng, [22, 27, 32])
    b2['obese'] = fuzz.trimf(b_rng, [28, 70, 70])
    r2['low'] = fuzz.trimf(r_rng, [0.0, 0.0, 0.4])
    r2['medium'] = fuzz.trimf(r_rng, [0.2, 0.5, 0.8])
    r2['high'] = fuzz.trimf(r_rng, [0.6, 1.0, 1.0])
    rules_2 = [
        ctrl.Rule(g2['high'] & b2['obese'], r2['high']),
        ctrl.Rule(g2['high'] & b2['overweight'], r2['high']),
        ctrl.Rule(g2['high'] & b2['normal'], r2['medium']),
        ctrl.Rule(g2['normal'] & b2['obese'], r2['medium']),
        ctrl.Rule(g2['normal'] & b2['overweight'], r2['medium']),
        ctrl.Rule(g2['normal'] & b2['normal'], r2['low']),
        ctrl.Rule(g2['low'], r2['low']),
    ]
    exp1_results['2 atrybuty\n(glukoza+BMI)'] = build_and_evaluate(rules_2, {'glucose': 'Glucose', 'bmi': 'BMI_c'},
                                                                   X_test_clean, y_test, 0.7)

    # Wariant C (3 atrybuty)
    g3 = ctrl.Antecedent(g_rng, 'glucose')
    b3 = ctrl.Antecedent(b_rng, 'bmi')
    a3 = ctrl.Antecedent(a_rng, 'age')
    r3 = ctrl.Consequent(r_rng, 'risk')
    g3['low'] = fuzz.trimf(g_rng, [0, 0, 100])
    g3['normal'] = fuzz.trimf(g_rng, [80, 110, 126])
    g3['high'] = fuzz.trimf(g_rng, [110, 200, 200])
    b3['normal'] = fuzz.trimf(b_rng, [0, 0, 25])
    b3['overweight'] = fuzz.trimf(b_rng, [22, 27, 32])
    b3['obese'] = fuzz.trimf(b_rng, [28, 70, 70])
    a3['young'] = fuzz.trimf(a_rng, [0, 0, 35])
    a3['middle'] = fuzz.trimf(a_rng, [25, 45, 60])
    a3['old'] = fuzz.trimf(a_rng, [50, 80, 80])
    r3['low'] = fuzz.trimf(r_rng, [0.0, 0.0, 0.4])
    r3['medium'] = fuzz.trimf(r_rng, [0.2, 0.5, 0.8])
    r3['high'] = fuzz.trimf(r_rng, [0.6, 1.0, 1.0])
    rules_3 = [
        ctrl.Rule(g3['high'] & b3['obese'] & a3['old'], r3['high']),
        ctrl.Rule(g3['high'] & b3['obese'] & a3['middle'], r3['high']),
        ctrl.Rule(g3['high'] & b3['overweight'] & a3['old'], r3['high']),
        ctrl.Rule(g3['high'] & b3['overweight'] & a3['middle'], r3['high']),
        ctrl.Rule(g3['high'] & b3['overweight'] & a3['young'], r3['medium']),
        ctrl.Rule(g3['high'] & b3['normal'] & a3['old'], r3['medium']),
        ctrl.Rule(g3['high'] & b3['normal'] & a3['middle'], r3['medium']),
        ctrl.Rule(g3['high'] & b3['normal'] & a3['young'], r3['low']),
        ctrl.Rule(g3['high'] & b3['obese'] & a3['young'], r3['medium']),
        ctrl.Rule(g3['normal'] & b3['obese'] & a3['old'], r3['high']),
        ctrl.Rule(g3['normal'] & b3['obese'] & a3['middle'], r3['medium']),
        ctrl.Rule(g3['normal'] & b3['obese'] & a3['young'], r3['medium']),
        ctrl.Rule(g3['normal'] & b3['overweight'] & a3['old'], r3['medium']),
        ctrl.Rule(g3['normal'] & b3['overweight'] & a3['middle'], r3['medium']),
        ctrl.Rule(g3['normal'] & b3['overweight'] & a3['young'], r3['low']),
        ctrl.Rule(g3['normal'] & b3['normal'], r3['low']),
        ctrl.Rule(g3['low'], r3['low']),
    ]
    exp1_results['3 atrybuty\n(glukoza+BMI+wiek)'] = build_and_evaluate(rules_3, {'glucose': 'Glucose', 'bmi': 'BMI_c',
                                                                                  'age': 'Age_c'}, X_test_clean, y_test,
                                                                        0.5)

    # ================= EKSPERYMENT 2 =================
    # 2 MF
    g_bin = ctrl.Antecedent(g_rng, 'glucose')
    b_bin = ctrl.Antecedent(b_rng, 'bmi')
    r_bin = ctrl.Consequent(r_rng, 'risk')
    g_bin['normal'] = fuzz.trimf(g_rng, [0, 0, 126])
    g_bin['high'] = fuzz.trimf(g_rng, [100, 200, 200])
    b_bin['normal'] = fuzz.trimf(b_rng, [0, 0, 30])
    b_bin['obese'] = fuzz.trimf(b_rng, [25, 70, 70])
    r_bin['low'] = fuzz.trimf(r_rng, [0.0, 0.0, 0.5])
    r_bin['high'] = fuzz.trimf(r_rng, [0.5, 1.0, 1.0])
    rules_2mf = [
        ctrl.Rule(g_bin['high'] & b_bin['obese'], r_bin['high']),
        ctrl.Rule(g_bin['high'] & b_bin['normal'], r_bin['high']),
        ctrl.Rule(g_bin['normal'] & b_bin['obese'], r_bin['high']),
        ctrl.Rule(g_bin['normal'] & b_bin['normal'], r_bin['low']),
    ]
    exp2_results['2 MF\n(binarne)'] = build_and_evaluate(rules_2mf, {'glucose': 'Glucose', 'bmi': 'BMI_c'},
                                                         X_test_clean, y_test, 0.5)

    # 3 MF (Kopiujemy wynik z exp 1)
    exp2_results['3 MF\n(bazowy)'] = exp1_results['2 atrybuty\n(glukoza+BMI)']

    # 4 MF
    g4 = ctrl.Antecedent(g_rng, 'glucose')
    b4 = ctrl.Antecedent(b_rng, 'bmi')
    r4 = ctrl.Consequent(r_rng, 'risk')
    g4['low'] = fuzz.trimf(g_rng, [0, 0, 90])
    g4['borderline'] = fuzz.trimf(g_rng, [70, 100, 126])
    g4['high'] = fuzz.trimf(g_rng, [110, 155, 180])
    g4['very_high'] = fuzz.trimf(g_rng, [160, 200, 200])
    b4['normal'] = fuzz.trimf(b_rng, [0, 0, 25])
    b4['overweight'] = fuzz.trimf(b_rng, [22, 27, 30])
    b4['obese'] = fuzz.trimf(b_rng, [28, 35, 40])
    b4['morbid_obese'] = fuzz.trimf(b_rng, [38, 70, 70])
    r4['very_low'] = fuzz.trimf(r_rng, [0.0, 0.0, 0.25])
    r4['low'] = fuzz.trimf(r_rng, [0.1, 0.3, 0.5])
    r4['high'] = fuzz.trimf(r_rng, [0.4, 0.65, 0.8])
    r4['very_high'] = fuzz.trimf(r_rng, [0.7, 1.0, 1.0])
    rules_4mf = [
        ctrl.Rule(g4['very_high'] & b4['morbid_obese'], r4['very_high']),
        ctrl.Rule(g4['very_high'] & b4['obese'], r4['very_high']),
        ctrl.Rule(g4['very_high'] & b4['overweight'], r4['high']),
        ctrl.Rule(g4['very_high'] & b4['normal'], r4['high']),
        ctrl.Rule(g4['high'] & b4['morbid_obese'], r4['very_high']),
        ctrl.Rule(g4['high'] & b4['obese'], r4['high']),
        ctrl.Rule(g4['high'] & b4['overweight'], r4['high']),
        ctrl.Rule(g4['high'] & b4['normal'], r4['low']),
        ctrl.Rule(g4['borderline'] & b4['morbid_obese'], r4['high']),
        ctrl.Rule(g4['borderline'] & b4['obese'], r4['high']),
        ctrl.Rule(g4['borderline'] & b4['overweight'], r4['low']),
        ctrl.Rule(g4['borderline'] & b4['normal'], r4['low']),
        ctrl.Rule(g4['low'], r4['very_low']),
    ]
    exp2_results['4 MF\n(granularne)'] = build_and_evaluate(rules_4mf, {'glucose': 'Glucose', 'bmi': 'BMI_c'},
                                                            X_test_clean, y_test, 0.5)

    # ================= EKSPERYMENT 3 =================
    def build_mf_type(mf_type, threshold):
        g = ctrl.Antecedent(g_rng, 'glucose')
        b = ctrl.Antecedent(b_rng, 'bmi')
        r = ctrl.Consequent(r_rng, 'risk')

        if mf_type == 'trimf':
            g['low'] = fuzz.trimf(g_rng, [0, 0, 100])
            g['normal'] = fuzz.trimf(g_rng, [80, 110, 126])
            g['high'] = fuzz.trimf(g_rng, [110, 200, 200])
            b['normal'] = fuzz.trimf(b_rng, [0, 0, 25])
            b['overweight'] = fuzz.trimf(b_rng, [22, 27, 32])
            b['obese'] = fuzz.trimf(b_rng, [28, 70, 70])
            r['low'] = fuzz.trimf(r_rng, [0.0, 0.0, 0.4])
            r['medium'] = fuzz.trimf(r_rng, [0.2, 0.5, 0.8])
            r['high'] = fuzz.trimf(r_rng, [0.6, 1.0, 1.0])
        elif mf_type == 'trapmf':
            g['low'] = fuzz.trapmf(g_rng, [0, 0, 80, 100])
            g['normal'] = fuzz.trapmf(g_rng, [80, 100, 115, 126])
            g['high'] = fuzz.trapmf(g_rng, [110, 130, 200, 200])
            b['normal'] = fuzz.trapmf(b_rng, [0, 0, 20, 25])
            b['overweight'] = fuzz.trapmf(b_rng, [20, 25, 28, 32])
            b['obese'] = fuzz.trapmf(b_rng, [28, 33, 70, 70])
            r['low'] = fuzz.trapmf(r_rng, [0.0, 0.0, 0.2, 0.4])
            r['medium'] = fuzz.trapmf(r_rng, [0.2, 0.35, 0.6, 0.75])
            r['high'] = fuzz.trapmf(r_rng, [0.6, 0.75, 1.0, 1.0])
        elif mf_type == 'gaussmf':
            g['low'] = fuzz.gaussmf(g_rng, 50, 30)
            g['normal'] = fuzz.gaussmf(g_rng, 110, 15)
            g['high'] = fuzz.gaussmf(g_rng, 170, 30)
            b['normal'] = fuzz.gaussmf(b_rng, 18, 5)
            b['overweight'] = fuzz.gaussmf(b_rng, 27, 3)
            b['obese'] = fuzz.gaussmf(b_rng, 40, 10)
            r['low'] = fuzz.gaussmf(r_rng, 0.1, 0.12)
            r['medium'] = fuzz.gaussmf(r_rng, 0.5, 0.15)
            r['high'] = fuzz.gaussmf(r_rng, 0.9, 0.12)

        rules = [
            ctrl.Rule(g['high'] & b['obese'], r['high']),
            ctrl.Rule(g['high'] & b['overweight'], r['high']),
            ctrl.Rule(g['high'] & b['normal'], r['medium']),
            ctrl.Rule(g['normal'] & b['obese'], r['medium']),
            ctrl.Rule(g['normal'] & b['overweight'], r['medium']),
            ctrl.Rule(g['normal'] & b['normal'], r['low']),
            ctrl.Rule(g['low'], r['low']),
        ]
        return build_and_evaluate(rules, {'glucose': 'Glucose', 'bmi': 'BMI_c'}, X_test_clean, y_test, threshold)

    for mf_name, thr in [('trimf', 0.7), ('trapmf', 0.5), ('gaussmf', 0.5)]:
        exp3_results[mf_name] = build_mf_type(mf_name, thr)

    print("Systemy rozmyte przeliczone!\n")
    return exp1_results, exp2_results, exp3_results