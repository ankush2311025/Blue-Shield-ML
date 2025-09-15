# main.py
from utils.model_trainer import DisasterModelTrainer
from utils.visualizer import DisasterVisualizer
import os

def main():
    print("🚀 Starting Disaster Monitoring ML Pipeline")
    print("=" * 50)
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Train Model
    print("\n1. 🎯 Training Disaster Classification Model")
    trainer = DisasterModelTrainer()
    
    # CORRECTED: Use load_data instead of load_and_combine_data
    df = trainer.load_data(
        'data/coastal_tweets_data.csv',
        'data/indian_coastal_comments.csv'
    )
    
    if not df.empty:
        # Prepare and train
        df = trainer.prepare_training_data(df)
        trainer.train_model(df)
        trainer.save_model()
        
        print("✅ Model training completed successfully!")
    else:
        print("❌ No data available for training")
    
    # Step 2: Create Visualizations
    print("\n2. 📊 Creating Trend Visualizations")
    visualizer = DisasterVisualizer()
    visualizer.load_data('data/coastal_tweets_data.csv', 'data/indian_coastal_comments.csv')
    visualizer.create_trend_dashboard()
    
    print("✅ Visualization completed successfully!")
    print("\n🎉 Pipeline execution complete!")
    print("📁 Check 'models/' for trained models and root directory for trends dashboard")

if __name__ == "__main__":
    main()