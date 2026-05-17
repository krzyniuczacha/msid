import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Wymuszenie wyświetlania wszystkich kolumn (None oznacza brak limitu)
pd.set_option('display.max_columns', None)

# Zwiększenie maksymalnej szerokości linii w terminalu, żeby kolumny się nie rozjeżdżały
pd.set_option('display.width', 1000)

df = pd.read_csv('diabetes.csv')

#info o datasecie
print("Informacje o zbiorze danych:")
df.info()
print()

print("pierwsze 5 wierszy zbioru danych:")
print(df.head())

#statystyki opisowe
print("Statystyki opisowe:")
print(df.describe())

#histogram z linią trendu (kde=True) dla glukozy
sns.histplot(data=df, x='Glucose', kde=True, color='blue')

#tytuł i opisy osi, żeby wykres był czytelny w raporcie
plt.title('Rozkład poziomu glukozy u pacjentów')
plt.xlabel('Poziom glukozy')
plt.ylabel('Liczba pacjentów')

#Ta funkcja fizycznie otwiera okienko z wykresem na ekranie
#plt.show()

# --- PRZYGOTOWANIE CECH I CELU ---
# X to wszystkie kolumny OPRÓCZ kolumny wynikowej (zrzucamy ją za pomocą .drop)
X = df.drop(columns=['Outcome'])

# y to TYLKO i wyłącznie kolumna wynikowa
y = df['Outcome']

# --- PODZIAŁ NA ZBIÓR TRENINGOWY I TESTOWY ---
# test_size=0.2 oznacza, że 20% danych idzie do testu, a 80% do nauki
# random_state=42 blokuje losowość, żeby wyniki były zawsze takie same
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sprawdzamy, czy podział się udał (funkcja .shape pokaże nam liczbę wierszy i kolumn)
print("\nPodział zakończony sukcesem!")
print(f"Liczba pacjentów w zbiorze treningowym (nauka dla ML): {X_train.shape[0]}")
print(f"Liczba pacjentów w zbiorze testowym (egzamin dla ML i Fuzzy): {X_test.shape[0]}")

