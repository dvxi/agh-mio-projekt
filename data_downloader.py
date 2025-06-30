#!/usr/bin/env python3
"""
Trump Tweets Dataset Downloader
==============================

This script helps download the Trump tweets dataset from Kaggle.
You need to have a Kaggle account and API token set up.

Instructions:
1. Create a Kaggle account at https://www.kaggle.com/
2. Go to Account settings and create a new API token
3. Place the kaggle.json file in ~/.kaggle/ directory
4. Run this script to download the dataset

Dataset: https://www.kaggle.com/datasets/austinreese/trump-tweets
"""

import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import pandas as pd

def setup_kaggle_api():
    """
    Set up Kaggle API authentication.
    """
    try:
        api = KaggleApi()
        api.authenticate()
        print("Kaggle API authentication successful!")
        return api
    except Exception as e:
        print(f"Kaggle API authentication failed: {e}")
        print("\nTo use Kaggle API:")
        print("1. Create account at https://www.kaggle.com/")
        print("2. Go to Account settings and create API token")
        print("3. Place kaggle.json in ~/.kaggle/ directory")
        print("4. Make sure the file permissions are correct: chmod 600 ~/.kaggle/kaggle.json")
        return None

def download_dataset():
    """
    Download the Trump tweets dataset from Kaggle.
    """
    api = setup_kaggle_api()
    
    if api is None:
        print("Cannot proceed without Kaggle API authentication.")
        return None
    
    dataset_name = "austinreese/trump-tweets"
    download_path = "./data"
    
    try:
        print(f"Downloading dataset: {dataset_name}")
        
        # Create data directory if it doesn't exist
        os.makedirs(download_path, exist_ok=True)
        
        # Download dataset
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        
        print(f"Dataset downloaded to: {download_path}")
        
        # List downloaded files
        files = os.listdir(download_path)
        print(f"Downloaded files: {files}")
        
        # Try to load and inspect the data
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(download_path, file)
                print(f"\nInspecting {file}:")
                
                df = pd.read_csv(file_path)
                print(f"Shape: {df.shape}")
                print(f"Columns: {df.columns.tolist()}")
                print(f"First few rows:")
                print(df.head())
                
                return file_path
        
        return download_path
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def create_sample_dataset():
    """
    Create a sample dataset if Kaggle download fails.
    """
    print("Creating sample dataset for demonstration...")
    
    # Create data directory
    os.makedirs("./data", exist_ok=True)
    
    # Sample Trump-style tweets (for educational purposes)
    sample_data = {
        'content': [
            "Great meeting today with world leaders. America First!",
            "The fake news media is at it again. Very unfair coverage!",
            "Beautiful ceremony at the White House. Thank you to all who attended!",
            "Terrible deal made by previous administration. We will renegotiate!",
            "Amazing rally last night! Thousands of incredible people!",
            "The corrupt establishment is trying to stop our movement. They won't succeed!",
            "Fantastic jobs numbers! Best economy in history!",
            "Disgusting behavior by the radical left. America deserves better!",
            "Incredible support from the American people. Together we will win!",
            "Worst trade deal ever negotiated. We are fixing it!",
            "Perfect call with foreign leader. Complete and total transparency!",
            "Crooked media spreading fake news again. Sad!",
            "Tremendous success at the international summit. Historic achievement!",
            "Failing newspaper continues to write false stories. Nobody believes them!",
            "Wonderful visit with our brave troops. Heroes, every single one!",
            "Rigged system trying to undermine democracy. We will not let this happen!",
            "Greatest economy the world has ever seen! Jobs, jobs, jobs!",
            "Sleepy Joe has no energy or vision for America. We need strength!",
            "Beautiful letter received from respected world leader. Mutual respect!",
            "Nasty questions from very biased reporters. The people see through it!"
        ] * 25,  # Repeat to create more samples
        'date': pd.date_range('2020-01-01', periods=500, freq='6H'),
        'retweets': [1000, 500, 2000, 800, 1500, 300, 3000, 600, 1200, 400] * 50,
        'favorites': [5000, 2500, 10000, 4000, 7500, 1500, 15000, 3000, 6000, 2000] * 50,
        'isRetweet': [False] * 500
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save to CSV
    file_path = "./data/trump_tweets_sample.csv"
    df.to_csv(file_path, index=False)
    
    print(f"Sample dataset created: {file_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return file_path

def main():
    """
    Main function to download or create dataset.
    """
    print("Trump Tweets Dataset Downloader")
    print("=" * 40)
    
    # Try to download real dataset first
    dataset_path = download_dataset()
    
    if dataset_path is None:
        print("\nFalling back to sample dataset...")
        dataset_path = create_sample_dataset()
    
    print(f"\nDataset ready at: {dataset_path}")
    print("\nYou can now run trump_sentiment_analysis.py with this dataset!")

if __name__ == "__main__":
    main() 