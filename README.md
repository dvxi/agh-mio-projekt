# Analiza Sentymentu Tweetów Trump'a z wykorzystaniem SHAP

## Opis projektu

Ten projekt realizuje zadanie analizy sentymentu tweetów z wykorzystaniem zbioru danych Trump Tweets z platformy Kaggle. Głównym celem jest zbudowanie modelu klasyfikacji sentymentu oraz przeprowadzenie analizy SHAP (SHapley Additive exPlanations) w celu wyjaśnienia decyzji modelu.

## Cele projektu

1. **Budowa modelu sentymentu** - wykorzystanie metod uczenia maszynowego do klasyfikacji sentymentu tweetów
2. **Analiza SHAP** - wyjaśnienie predykcji modelu za pomocą wartości Shapley'a
3. **Interpretacja wyników** - dyskusja nad najważniejszymi cechami wpływającymi na decyzje modelu
4. **Wizualizacja** - stworzenie wykresów i wizualizacji pomagających w zrozumieniu danych i modelu

## Struktura projektu

```
project/
├── README.md                      # Ten plik
├── requirements.txt               # Wymagane biblioteki
├── trump_sentiment_analysis.py    # Główny skrypt analizy
├── data_downloader.py            # Skrypt do pobierania danych
├── sentiment_analysis_notebook.ipynb  # Interaktywny notebook Jupyter
└── data/                         # Folder na dane (tworzony automatycznie)
    └── trump_tweets_sample.csv   # Przykładowe dane
```

## Dataset

Projekt wykorzystuje zbiór danych "Trump Tweets" dostępny na platformie Kaggle:
- **URL:** https://www.kaggle.com/datasets/austinreese/trump-tweets
- **Opis:** Kolekcja tweetów opublikowanych przez Donald Trump
- **Format:** CSV z polami takimi jak content, date, retweets, favorites

## Wymagania systemowe

### Biblioteki Python
- pandas >= 2.0.3
- numpy >= 1.24.3
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.2
- seaborn >= 0.12.2
- nltk >= 3.8.1
- textblob >= 0.17.1
- wordcloud >= 1.9.2
- shap >= 0.42.1
- jupyter >= 1.0.0
- kaggle >= 1.5.16

### Instalacja zależności
```bash
pip install -r requirements.txt
```

## Instrukcja uruchomienia

### Opcja 1: Użycie gotowych przykładowych danych
```bash
python trump_sentiment_analysis.py
```

### Opcja 2: Pobieranie prawdziwych danych z Kaggle
1. Załóż konto na https://www.kaggle.com/
2. Wygeneruj token API w ustawieniach konta
3. Umieść plik `kaggle.json` w folderze `~/.kaggle/`
4. Uruchom skrypt pobierania danych:
```bash
python data_downloader.py
```
5. Uruchom główną analizę:
```bash
python trump_sentiment_analysis.py
```

### Opcja 3: Interaktywny Jupyter Notebook
```bash
jupyter notebook sentiment_analysis_notebook.ipynb
```

## Metodologia

### 1. Przetwarzanie danych
- Czyszczenie tekstu (usuwanie URL, znaków specjalnych)
- Normalizacja (małe litery, usuwanie nadmiarowych spacji)
- Tokenizacja i usuwanie słów stop

### 2. Etykietowanie sentymentu
- Wykorzystanie biblioteki TextBlob do automatycznego etykietowania
- Klasyfikacja na trzy kategorie: positive, negative, neutral
- Podział na podstawie wartości polarności

### 3. Modelowanie
- **Wektoryzacja:** TF-IDF z n-gramami (1,2)
- **Model:** Regresja logistyczna z regularyzacją
- **Ewaluacja:** Dokładność, macierz pomyłek, raport klasyfikacji

### 4. Analiza SHAP
- **Eksplanator:** LinearExplainer dla regresji logistycznej
- **Analiza globalna:** Ważność cech na całym zbiorze
- **Analiza lokalna:** Wyjaśnienie indywidualnych predykcji
- **Wizualizacje:** Summary plots, feature importance

## Oczekiwane wyniki

### Pliki wyjściowe
- `model_evaluation.png` - Wykresy ewaluacji modelu
- `shap_summary.png` - Wykres podsumowujący SHAP
- `word_clouds.png` - Chmury słów dla różnych sentymentów

### Metryki wydajności
- Dokładność modelu: ~0.75-0.85 (zależnie od danych)
- Precision/Recall dla każdej klasy
- Macierz pomyłek z wizualizacją

### Wnioski z analizy SHAP
- Identyfikacja najważniejszych słów/fraz
- Zrozumienie bias modelu
- Wyjaśnienie indywidualnych decyzji

## Struktura kodu

### `trump_sentiment_analysis.py`
Główna klasa `TrumpTweetsSentimentAnalyzer` zawiera:
- `load_data()` - ładowanie i przygotowanie danych
- `preprocess_text()` - czyszczenie tekstu
- `train_model()` - trenowanie modelu
- `evaluate_model()` - ewaluacja wydajności
- `setup_shap_analysis()` - przygotowanie analizy SHAP
- `analyze_with_shap()` - obliczanie wartości SHAP
- `analyze_individual_predictions()` - analiza indywidualnych przypadków

### `data_downloader.py`
Narzędzia do pobierania danych:
- Konfiguracja API Kaggle
- Pobieranie zbioru danych
- Tworzenie przykładowych danych (fallback)

## Interpretacja wyników SHAP

### Wartości SHAP
- **Pozytywne wartości:** cechy zwiększające prawdopodobieństwo danej klasy
- **Negatywne wartości:** cechy zmniejszające prawdopodobieństwo
- **Wartość bezwzględna:** siła wpływu cechy

### Typy analiz
1. **Summary Plot:** globalna ważność cech
2. **Individual Analysis:** wyjaśnienie konkretnych predykcji
3. **Feature Importance:** ranking najważniejszych słów

## Ograniczenia i dalszy rozwój

### Obecne ograniczenia
- Automatyczne etykietowanie może być niedoskonałe
- Brak uwzględnienia kontekstu temporalnego
- Ograniczona interpretacja ironii i sarkazmu

### Możliwe usprawnienia
1. Użycie pre-trained models (BERT, RoBERTa)
2. Analiza temporalna zmian sentymentu
3. Włączenie metadanych (retweets, likes)
4. Walidacja krzyżowa z różnymi algorytmami

## Przykłady użycia

### Podstawowa analiza
```python
from trump_sentiment_analysis import TrumpTweetsSentimentAnalyzer

analyzer = TrumpTweetsSentimentAnalyzer()
analyzer.load_data()
analyzer.prepare_data()
analyzer.train_model()
analyzer.evaluate_model()
```

### Analiza SHAP
```python
analyzer.setup_shap_analysis()
analyzer.analyze_with_shap()
analyzer.analyze_individual_predictions([0, 1, 2])
```

## Wymagania sprawozdania

Zgodnie z wymaganiami projektu, sprawozdanie powinno zawierać:
1. **Opis zadania** (1-2 strony)
2. **Metodologia** (1-2 strony) 
3. **Wyniki eksperymentów** (2-3 strony)
4. **Analiza SHAP** (2-3 strony)
5. **Wnioski i dyskusja** (1 strona)

## Autor

Szymon Gwóźdź

## Licencja

Projekt edukacyjny - wykorzystanie w celach akademickich.

---