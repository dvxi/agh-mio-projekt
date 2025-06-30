#!/usr/bin/env python3
"""
Runner Script for Trump Tweets Sentiment Analysis
================================================

This script demonstrates the complete workflow:
1. Data preparation
2. Model training
3. Evaluation
4. SHAP analysis

Usage:
    python run_analysis.py
"""

import os
import sys
from trump_sentiment_analysis import TrumpTweetsSentimentAnalyzer

def main():
    """Run the complete sentiment analysis pipeline."""
    
    print("🚀 Uruchamianie analizy sentymentu tweetów Trump'a")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        print("📊 Inicjalizacja analizatora...")
        analyzer = TrumpTweetsSentimentAnalyzer()
        
        # Load data
        print("📁 Ładowanie danych...")
        analyzer.load_data()
        
        # Prepare data
        print("🧹 Przygotowywanie danych...")
        analyzer.prepare_data()
        
        # Train model
        print("🤖 Trenowanie modelu...")
        analyzer.train_model('logistic_regression')
        
        # Evaluate model
        print("📈 Ewaluacja modelu...")
        accuracy, _ = analyzer.evaluate_model()
        
        # SHAP analysis
        print("🔍 Analiza SHAP...")
        analyzer.setup_shap_analysis()
        analyzer.analyze_with_shap(sample_size=30)
        
        # Individual predictions analysis
        print("🎯 Analiza indywidualnych predykcji...")
        analyzer.analyze_individual_predictions([0, 1, 2])
        
        # Create visualizations
        print("📊 Tworzenie wizualizacji...")
        analyzer.create_word_clouds()
        
        # Generate final report
        print("📋 Generowanie raportu...")
        analyzer.generate_report()
        
        print("\n" + "=" * 60)
        print("✅ Analiza zakończona pomyślnie!")
        print(f"📊 Dokładność modelu: {accuracy:.4f}")
        print("\n📁 Wygenerowane pliki:")
        
        expected_files = [
            'model_evaluation.png',
            'shap_summary.png', 
            'word_clouds.png'
        ]
        
        for file in expected_files:
            if os.path.exists(file):
                print(f"   ✓ {file}")
            else:
                print(f"   ⚠ {file} (nie wygenerowano)")
        
        print("\n🎓 Projekt gotowy do oceny!")
        print("📝 Nie zapomnij stworzyć sprawozdania (5-10 stron) zawierającego:")
        print("   - Opis metodologii")
        print("   - Wyniki eksperymentów")
        print("   - Analizę SHAP")
        print("   - Wnioski i dyskusję")
        
    except Exception as e:
        print(f"❌ Błąd podczas analizy: {e}")
        print("💡 Spróbuj uruchomić ponownie lub sprawdź logi błędów")
        sys.exit(1)

if __name__ == "__main__":
    main() 