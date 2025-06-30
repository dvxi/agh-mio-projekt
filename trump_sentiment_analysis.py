#!/usr/bin/env python3
"""
Trump Tweets Sentiment Analysis with SHAP
==========================================

This script performs sentiment analysis on Trump tweets and uses SHAP to explain the model's decisions.
Dataset: https://www.kaggle.com/datasets/austinreese/trump-tweets

Author: [Your Name]
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import nltk
from textblob import TextBlob
import re
import warnings
import shap
from wordcloud import WordCloud
import os

warnings.filterwarnings('ignore')

class TrumpTweetsSentimentAnalyzer:
    """
    A comprehensive sentiment analysis tool for Trump tweets with SHAP explanations.
    """
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.model = None
        self.pipeline = None
        self.explainer = None
        
        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
    
    def load_data(self, file_path=None):
        """
        Load Trump tweets dataset. If file_path is None, tries to download from Kaggle.
        """
        if file_path and os.path.exists(file_path):
            print(f"Loading data from {file_path}")
            self.data = pd.read_csv(file_path)
        else:
            # Try to load sample data or create synthetic data for demonstration
            print("Creating sample dataset for demonstration...")
            self._create_sample_data()
        
        print(f"Dataset loaded with {len(self.data)} tweets")
        print(f"Columns: {self.data.columns.tolist()}")
        
        return self.data
    
    def _create_sample_data(self):
        """
        Create sample tweet data for demonstration purposes.
        In a real scenario, you would download the actual Trump tweets dataset.
        """
        # Sample tweets with various sentiments
        sample_tweets = [
            "Great meeting with world leaders today! America first!",
            "The fake news media is at it again. Sad!",
            "Beautiful day at the White House. Thank you America!",
            "Terrible deal made by previous administration. We will fix it!",
            "Amazing rally tonight! Thousands of people came out!",
            "The corrupt politicians are trying to stop us. We won't let them!",
            "Fantastic economic numbers! Best in history!",
            "Disgusting behavior by the opposition. America deserves better!",
            "Incredible support from the American people. Thank you!",
            "Worst trade deal ever made. We are renegotiating!",
            "Perfect phone call with foreign leader. Complete transparency!",
            "Crooked media spreading lies again. America sees the truth!",
            "Tremendous success at the summit. Historic achievement!",
            "Failing newspaper writes fake stories. Nobody reads them anymore!",
            "Wonderful visit to troops overseas. Heroes every one!",
            "Rigged system trying to undermine our democracy. Not fair!",
            "Greatest economy in the world! Jobs, jobs, jobs!",
            "Sleepy opponent has no energy. America needs strength!",
            "Beautiful letter from world leader. Mutual respect!",
            "Nasty questions from biased reporters. Fake news!"
        ] * 50  # Repeat to get more data
        
        # Add some variation
        additional_tweets = [
            "Making America Great Again! #MAGA",
            "Border security is national security!",
            "Thank you to all our incredible veterans!",
            "The swamp is fighting back, but we will drain it!",
            "China trade deal is the best ever negotiated!",
            "Supreme Court pick will be announced soon!",
            "Infrastructure week is finally here!",
            "Space Force will make America dominant in space!",
            "Tax cuts are working for all Americans!",
            "Healthcare plan will be revealed very soon!"
        ] * 20
        
        all_tweets = sample_tweets + additional_tweets
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'content': all_tweets,
            'date': pd.date_range('2020-01-01', periods=len(all_tweets), freq='H'),
            'retweets': np.random.randint(100, 10000, len(all_tweets)),
            'favorites': np.random.randint(500, 50000, len(all_tweets))
        })
        
        print("Sample dataset created for demonstration purposes.")
        print("In a real project, you would use the actual Trump tweets dataset from Kaggle.")
    
    def preprocess_text(self, text):
        """
        Clean and preprocess tweet text.
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_sentiment_labels(self):
        """
        Create sentiment labels using TextBlob for demonstration.
        In a real project, you might use pre-labeled data or different labeling methods.
        """
        print("Creating sentiment labels using TextBlob...")
        
        sentiments = []
        for text in self.data['content']:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiments.append('positive')
            elif polarity < -0.1:
                sentiments.append('negative')
            else:
                sentiments.append('neutral')
        
        self.data['sentiment'] = sentiments
        
        # Print distribution
        print("Sentiment distribution:")
        print(self.data['sentiment'].value_counts())
        
        return self.data
    
    def prepare_data(self):
        """
        Prepare data for machine learning.
        """
        print("Preprocessing tweets...")
        
        # Preprocess text
        self.data['cleaned_content'] = self.data['content'].apply(self.preprocess_text)
        
        # Create sentiment labels
        self.create_sentiment_labels()
        
        # Remove empty tweets
        self.data = self.data[self.data['cleaned_content'].str.len() > 0]
        
        # Prepare features and labels
        X = self.data['cleaned_content']
        y = self.data['sentiment']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, model_type='logistic_regression'):
        """
        Train sentiment analysis model.
        """
        print(f"Training {model_type} model...")
        
        # Create vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # Choose model
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Unsupported model type")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.model)
        ])
        
        # Train model
        self.pipeline.fit(self.X_train, self.y_train)
        
        print("Model training completed!")
        
        return self.pipeline
    
    def evaluate_model(self):
        """
        Evaluate model performance.
        """
        print("Evaluating model...")
        
        # Predictions
        y_pred = self.pipeline.predict(self.X_test)
        
        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.pipeline.classes_,
                    yticklabels=self.pipeline.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            feature_names = self.vectorizer.get_feature_names_out()
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-20:]
            
            plt.subplot(1, 2, 2)
            plt.barh(range(20), importances[top_indices])
            plt.yticks(range(20), [feature_names[i] for i in top_indices])
            plt.title('Top 20 Feature Importances')
            plt.xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, y_pred
    
    def setup_shap_analysis(self):
        """
        Set up SHAP explainer for the trained model.
        """
        print("Setting up SHAP analysis...")
        
        # Transform training data for SHAP
        X_train_vectorized = self.vectorizer.transform(self.X_train)
        
        # Create SHAP explainer
        if isinstance(self.model, LogisticRegression):
            self.explainer = shap.LinearExplainer(self.model, X_train_vectorized)
        else:
            # For tree-based models
            self.explainer = shap.TreeExplainer(self.model)
        
        print("SHAP explainer ready!")
        
        return self.explainer
    
    def analyze_with_shap(self, sample_size=100):
        """
        Perform SHAP analysis on sample predictions.
        """
        print("Performing SHAP analysis...")
        print("=" * 60)
        
        # Sample test data
        sample_indices = np.random.choice(len(self.X_test), 
                                        min(sample_size, len(self.X_test)), 
                                        replace=False)
        X_sample = self.X_test.iloc[sample_indices]
        
        print(f"DEBUG: Sample data info:")
        print(f"  - Sample size requested: {sample_size}")
        print(f"  - Actual sample size: {len(X_sample)}")
        print(f"  - Sample indices: {sample_indices[:10]}..." if len(sample_indices) > 10 else f"  - Sample indices: {sample_indices}")
        print(f"  - First 3 sample texts:")
        for i, text in enumerate(X_sample.iloc[:3]):
            print(f"    [{i}]: {text[:100]}...")
        
        # Transform sample data
        X_sample_vectorized = self.vectorizer.transform(X_sample)
        
        print(f"\nDEBUG: Vectorized data info:")
        print(f"  - Vectorized shape: {X_sample_vectorized.shape}")
        print(f"  - Data type: {type(X_sample_vectorized)}")
        print(f"  - Sparse matrix: {hasattr(X_sample_vectorized, 'toarray')}")
        print(f"  - Non-zero elements in first sample: {np.count_nonzero(X_sample_vectorized[0].toarray()) if hasattr(X_sample_vectorized, 'toarray') else 'N/A'}")
        
        # Get predictions for sample
        sample_predictions = self.pipeline.predict(X_sample)
        sample_probabilities = self.pipeline.predict_proba(X_sample)
        
        print(f"\nDEBUG: Model predictions info:")
        print(f"  - Model classes: {self.pipeline.classes_}")
        print(f"  - Sample predictions distribution: {pd.Series(sample_predictions).value_counts().to_dict()}")
        print(f"  - Prediction probabilities shape: {sample_probabilities.shape}")
        print(f"  - Sample probabilities (first 3):")
        for i in range(min(3, len(sample_probabilities))):
            probs_dict = {cls: prob for cls, prob in zip(self.pipeline.classes_, sample_probabilities[i])}
            print(f"    [{i}]: {probs_dict}")
        
        # Calculate SHAP values
        print(f"\nDEBUG: Calculating SHAP values...")
        print(f"  - Explainer type: {type(self.explainer)}")
        
        shap_values = self.explainer.shap_values(X_sample_vectorized)
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"\nDEBUG: SHAP values detailed info:")
        print(f"  - SHAP values type: {type(shap_values)}")
        print(f"  - Feature names count: {len(feature_names)}")
        print(f"  - First 10 feature names: {feature_names[:10]}")
        
        if isinstance(shap_values, list):
            print(f"  - SHAP values is a list with {len(shap_values)} elements (classes)")
            for i, sv in enumerate(shap_values):
                class_name = self.pipeline.classes_[i] if i < len(self.pipeline.classes_) else f"class_{i}"
                print(f"    Class {i} ({class_name}):")
                print(f"      - Shape: {sv.shape}")
                print(f"      - Data type: {sv.dtype}")
                print(f"      - Min value: {np.min(sv):.6f}")
                print(f"      - Max value: {np.max(sv):.6f}")
                print(f"      - Mean absolute value: {np.mean(np.abs(sv)):.6f}")
                print(f"      - Non-zero values: {np.count_nonzero(sv)}")
                
                # Show sample SHAP values for first instance
                if len(sv) > 0:
                    first_instance_shap = sv[0]
                    top_indices = np.argsort(np.abs(first_instance_shap))[-5:]
                    print(f"      - Top 5 SHAP features for first instance:")
                    for idx in reversed(top_indices):
                        if abs(first_instance_shap[idx]) > 1e-10:  # Only show non-trivial values
                            print(f"        {feature_names[idx]}: {first_instance_shap[idx]:.6f}")
        else:
            print(f"  - SHAP values shape: {shap_values.shape}")
            print(f"  - Data type: {shap_values.dtype}")
            print(f"  - Min value: {np.min(shap_values):.6f}")
            print(f"  - Max value: {np.max(shap_values):.6f}")
            print(f"  - Mean absolute value: {np.mean(np.abs(shap_values)):.6f}")
            print(f"  - Non-zero values: {np.count_nonzero(shap_values)}")
        
        print(f"\nDEBUG: Proceeding to visualization...")
        print("=" * 60)
        
        # Create visualizations
        print("SHAP Visualization Explanation:")
        print("- Each subplot shows feature importance for one sentiment class")
        print("- X-axis: SHAP values (how much each feature contributes to that class)")
        print("- Y-axis: Features (words/phrases) ranked by importance")
        print("- Colors: Feature values (red=high occurrence, blue=low occurrence)")
        print("- Dots: Individual predictions (each sample)")
        print("- Positive SHAP values push toward that class, negative values push away")
        print("-" * 60)
        try:
            # Handle 3D SHAP array (samples, features, classes)
            if len(shap_values.shape) == 3:
                print(f"DEBUG: Processing 3D SHAP array with shape {shap_values.shape}")
                
                num_classes = shap_values.shape[2]
                fig, axes = plt.subplots(num_classes, 1, figsize=(12, 6 * num_classes))
                
                # If only one class, axes won't be a list
                if num_classes == 1:
                    axes = [axes]
                
                for class_idx in range(num_classes):
                    class_name = self.pipeline.classes_[class_idx]
                    
                    # Extract SHAP values for this class: (samples, features)
                    class_shap_values = shap_values[:, :, class_idx]
                    
                    print(f"  DEBUG: Class {class_idx} ({class_name}) SHAP shape: {class_shap_values.shape}")
                    
                    # Use only top features to avoid clutter
                    n_top_features = min(25, class_shap_values.shape[1])
                    
                    # Get mean absolute SHAP values for feature selection
                    mean_abs_shap = np.mean(np.abs(class_shap_values), axis=0)
                    top_feature_indices = np.argsort(mean_abs_shap)[-n_top_features:]
                    
                    print(f"    DEBUG: Top feature indices: {top_feature_indices}")
                    print(f"    DEBUG: Top features: {feature_names[top_feature_indices]}")
                    
                    # Select top features
                    shap_values_subset = class_shap_values[:, top_feature_indices]
                    feature_names_subset = feature_names[top_feature_indices]
                    X_subset = X_sample_vectorized.toarray()[:, top_feature_indices]
                    
                    # Create summary plot for this class
                    plt.sca(axes[class_idx])
                    shap.summary_plot(shap_values_subset, X_subset,
                                    feature_names=feature_names_subset,
                                    show=False, plot_type="dot")
                    plt.title(f'SHAP Summary - {class_name.capitalize()} Sentiment')
                    
                    print(f"    DEBUG: Created plot for {class_name} class")
                
                plt.tight_layout()
                plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
                plt.show()
                print("DEBUG: Successfully created multi-class SHAP plots")
                
            # Handle list of 2D arrays (alternative multi-class format)
            elif isinstance(shap_values, list) and len(shap_values) > 2:
                print(f"DEBUG: Processing list of SHAP arrays")
                fig, axes = plt.subplots(len(shap_values), 1, figsize=(12, 6 * len(shap_values)))
                
                for i, class_name in enumerate(self.pipeline.classes_):
                    plt.sca(axes[i])
                    
                    # Use only top features to avoid clutter
                    n_top_features = min(25, shap_values[i].shape[1])
                    
                    # Get mean absolute SHAP values for feature selection
                    mean_abs_shap = np.mean(np.abs(shap_values[i]), axis=0)
                    top_feature_indices = np.argsort(mean_abs_shap)[-n_top_features:]
                    
                    # Select top features
                    shap_values_subset = shap_values[i][:, top_feature_indices]
                    feature_names_subset = feature_names[top_feature_indices]
                    X_subset = X_sample_vectorized.toarray()[:, top_feature_indices]
                    
                    # Create summary plot for this class
                    shap.summary_plot(shap_values_subset, X_subset,
                                    feature_names=feature_names_subset,
                                    show=False, plot_type="dot")
                    plt.title(f'SHAP Summary - {class_name.capitalize()} Class')
                
                plt.tight_layout()
                plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
                plt.show()
                
            # Handle 2D array (binary classification or single class)
            else:
                print(f"DEBUG: Processing 2D SHAP array")
                
                # For binary classification or single array
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                else:
                    shap_vals = shap_values
                
                # Use only top features
                n_top_features = min(25, shap_vals.shape[1])
                mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
                top_feature_indices = np.argsort(mean_abs_shap)[-n_top_features:]
                
                shap_values_subset = shap_vals[:, top_feature_indices]
                feature_names_subset = feature_names[top_feature_indices]
                X_subset = X_sample_vectorized.toarray()[:, top_feature_indices]
                
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values_subset, X_subset,
                                feature_names=feature_names_subset,
                                show=False)
                plt.title('SHAP Summary Plot - Feature Importance')
                plt.tight_layout()
                plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
                plt.show()
                
        except Exception as e:
            print(f"Error creating SHAP summary plot: {e}")
            print("Creating fallback visualization...")
            
            # Fallback: create a simple bar plot of feature importance
            if isinstance(shap_values, list):
                # Use the first class for demonstration
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
            top_indices = np.argsort(mean_abs_shap)[-20:]
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(20), mean_abs_shap[top_indices])
            plt.yticks(range(20), [feature_names[i] for i in top_indices])
            plt.xlabel('Mean Absolute SHAP Value')
            plt.title('Top 20 Features by SHAP Importance')
            plt.tight_layout()
            plt.savefig('shap_summary_fallback.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return shap_values, feature_names
    
    def analyze_individual_predictions(self, tweet_indices=[0, 1, 2]):
        """
        Analyze individual tweet predictions with SHAP.
        """
        print("Analyzing individual predictions...")
        
        class_indexer = {name: i for i, name in enumerate(self.pipeline.classes_)}

        for idx in tweet_indices:
            if idx >= len(self.X_test):
                continue
            
            tweet = self.X_test.iloc[idx]
            true_label = self.y_test.iloc[idx]
            predicted_label = self.pipeline.predict([tweet])[0]
            
            print(f"\nTweet {idx}:")
            print(f"Text: {tweet[:100]}...")
            print(f"True label: {true_label}")
            print(f"Predicted label: {predicted_label}")
            
            # Transform for SHAP
            tweet_vectorized = self.vectorizer.transform([tweet])
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(tweet_vectorized)
            
            if isinstance(shap_values, list):
                # For multiclass, shap_values is a list of arrays.
                predicted_class_index = class_indexer[predicted_label]
                shap_vals = shap_values[predicted_class_index][0]
            else:
                # For binary classification, shap_values is a single array.
                shap_vals = shap_values[0]

            # Get top contributing features
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get top contributing features
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get non-zero features
            non_zero_mask = tweet_vectorized.toarray()[0] > 0
            
            # Ensure shap_vals is a 1D array
            if len(shap_vals.shape) > 1:
                shap_vals = shap_vals.flatten()

            contributing_features = [(feature_names[i], shap_vals[i]) 
                                   for i in range(len(feature_names)) 
                                   if non_zero_mask[i]]
            
            # Sort by absolute SHAP value
            contributing_features.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print(f"Top contributing features for prediction '{predicted_label}':")
            for feature, shap_val in contributing_features[:10]:
                print(f"  - {feature}: {shap_val:.4f}")
    
    def create_word_clouds(self):
        """
        Create word clouds for different sentiment classes.
        """
        print("Creating word clouds...")
        
        plt.figure(figsize=(15, 5))
        
        sentiments = self.data['sentiment'].unique()
        
        for i, sentiment in enumerate(sentiments):
            sentiment_tweets = self.data[self.data['sentiment'] == sentiment]['cleaned_content']
            text = ' '.join(sentiment_tweets)
            
            plt.subplot(1, len(sentiments), i+1)
            
            if text.strip():  # Check if text is not empty
                wordcloud = WordCloud(width=400, height=300, 
                                    background_color='white').generate(text)
                plt.imshow(wordcloud, interpolation='bilinear')
            else:
                plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=plt.gca().transAxes)
            
            plt.title(f'{sentiment.capitalize()} Tweets')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('word_clouds.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """
        Generate a summary report of the analysis.
        """
        print("\n" + "="*60)
        print("TRUMP TWEETS SENTIMENT ANALYSIS REPORT")
        print("="*60)
        
        print(f"Dataset size: {len(self.data)} tweets")
        print(f"Sentiment distribution:")
        print(self.data['sentiment'].value_counts())
        
        print(f"\nModel performance:")
        accuracy, _ = self.evaluate_model()
        
        print(f"\nKey findings from SHAP analysis:")
        print("- SHAP values help explain individual predictions")
        print("- Feature importance shows which words/phrases drive sentiment")
        print("- Analysis reveals model's decision-making process")
        
        print("\nFiles generated:")
        print("- model_evaluation.png: Model performance metrics")
        print("- shap_summary.png: SHAP feature importance")
        print("- word_clouds.png: Word clouds by sentiment")
        
        print("\n" + "="*60)

def main():
    """
    Main execution function.
    """
    print("Trump Tweets Sentiment Analysis with SHAP")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = TrumpTweetsSentimentAnalyzer()
    
    # Load and prepare data
    analyzer.load_data('data/trumptweets.csv')
    analyzer.prepare_data()
    
    # Train model
    analyzer.train_model('logistic_regression')
    
    # Evaluate model
    analyzer.evaluate_model()
    
    # SHAP analysis
    analyzer.setup_shap_analysis()
    analyzer.analyze_with_shap()
    analyzer.analyze_individual_predictions([0, 1, 2, 3, 4])
    
    # Create visualizations
    analyzer.create_word_clouds()
    
    # Generate report
    analyzer.generate_report()
    
    print("\nAnalysis completed! Check the generated plots and results.")

if __name__ == "__main__":
    main() 