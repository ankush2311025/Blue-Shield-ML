import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import re

class DisasterModelTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def load_data(self, tweet_path, youtube_comment_path):
        """Load data from both sources"""
        print("📂 Loading data...")
        
        tweets_df = pd.read_csv(tweet_path) if tweet_path else pd.DataFrame()
        yt_comments_df = pd.read_csv(youtube_comment_path) if youtube_comment_path else pd.DataFrame()
        
        if not tweets_df.empty:
            tweets_df['source'] = 'twitter'
            tweets_df['text'] = tweets_df['text'].astype(str)
            print(f"   Loaded {len(tweets_df)} tweets")
        
        if not yt_comments_df.empty:
            yt_comments_df['source'] = 'youtube'
            yt_comments_df['text'] = yt_comments_df['text'].astype(str)
            print(f"   Loaded {len(yt_comments_df)} YouTube comments")
        
        # Combine data
        combined_df = pd.concat([
            tweets_df[['text', 'source']] if not tweets_df.empty else pd.DataFrame(),
            yt_comments_df[['text', 'source']] if not yt_comments_df.empty else pd.DataFrame()
        ], ignore_index=True)
        
        print(f"📊 Total samples: {len(combined_df)}")
        return combined_df
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()
    
    def prepare_training_data(self, df):
        """Prepare data for training with automatic labeling"""
        print("🔧 Preprocessing text...")
        
        df['cleaned_text'] = df['text'].apply(self.preprocess_text)
        
        # Automatic labeling based on disaster keywords
        disaster_keywords = [
            'flood', 'cyclone', 'tsunami', 'storm', 'surge', 'wave',
            'disaster', 'emergency', 'warning', 'alert', 'evacuate',
            'damage', 'destroy', 'rescue', 'help', 'emergency',
            'rain', 'water', 'coastal', 'erosion', 'landslide'
        ]
        
        df['label'] = df['cleaned_text'].apply(
            lambda x: 1 if any(keyword in x for keyword in disaster_keywords) else 0
        )
        
        print(f"📈 Disaster samples: {df['label'].sum()}, Normal: {len(df) - df['label'].sum()}")
        return df
    
    def train_model(self, df, test_size=0.2):
        """Train the disaster classification model"""
        print("🎯 Training model...")
        
        X = self.vectorizer.fit_transform(df['cleaned_text'])
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("\n📊 Model Performance:")
        print(classification_report(y_test, y_pred))
        
        return self.model, self.vectorizer
    
    def predict_disaster(self, text):
        """Predict if text is disaster-related"""
        cleaned_text = self.preprocess_text(text)
        vectorized = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(vectorized)
        probability = self.model.predict_proba(vectorized)[0][1]
        return prediction[0], probability
    
    def test_regional_disasters(self):
        """Test with regional disaster scenarios"""
        print("\n🌊 Testing Regional Disaster Scenarios:")
        print("=" * 50)
        
        # Regional disaster test cases
        test_cases = [
            # Cyclones
            ("Cyclone alert for Odisha coast. Evacuation in Puri and Bhubaneswar.", "Odisha Cyclone"),
            ("Severe cyclonic storm warning for Andhra Pradesh coastline.", "Andhra Cyclone"),
            
            # Floods
            ("Major flooding in Chennai after heavy rains. Areas waterlogged.", "Chennai Flood"),
            ("Mumbai streets flooded due to high tide and rainfall.", "Mumbai Flood"),
            ("Kerala floods: Rescue operations ongoing in Kochi and Alappuzha.", "Kerala Flood"),
            
            # Tsunamis
            ("Tsunami warning for Andaman Islands after earthquake.", "Andaman Tsunami"),
            ("Coastal Tamil Nadu on high alert after undersea earthquake.", "Tamil Nadu Tsunami"),
            
            # Storm Surges
            ("Storm surge warning for West Bengal coast. Sea water entering villages.", "West Bengal Storm"),
            ("High waves and storm surge expected in Goa coastal areas.", "Goa Storm Surge"),
            
            # Non-disaster (should return 0)
            ("Beautiful weather at Goa beach today. Perfect for vacation!", "Normal"),
            ("Enjoying seafood at Mumbai's Gateway of India. Amazing view!", "Normal")
        ]
        
        results = []
        for text, scenario in test_cases:
            prediction, confidence = self.predict_disaster(text)
            results.append({
                'scenario': scenario,
                'prediction': 'DISASTER' if prediction else 'NORMAL',
                'confidence': f"{confidence:.1%}",
                'correct': (prediction == 1) if 'Normal' not in scenario else (prediction == 0)
            })
            
            status = "✅" if results[-1]['correct'] else "❌"
            print(f"{status} {scenario}: {results[-1]['prediction']} ({confidence:.1%})")
        
        # Calculate accuracy
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        print(f"\n📊 Test Accuracy: {accuracy:.1%}")
        
        return results

    def save_model(self, model_path='models/disaster_classifier.pkl',
                  vectorizer_path='models/vectorizer.pkl'):
        """Save trained model and vectorizer"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"💾 Model saved to {model_path}")

# Main execution
if __name__ == "__main__":
    trainer = DisasterModelTrainer()
    
    # Load data
    df = trainer.load_data('data/raw_tweets.csv', 'data/youtube_comments.csv')
    
    if not df.empty:
        # Prepare and train
        df = trainer.prepare_training_data(df)
        trainer.train_model(df)
        
        # Test with regional scenarios
        trainer.test_regional_disasters()
        
        # Save model
        trainer.save_model()
    else:
        print("❌ No data available for training")