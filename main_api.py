
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from datetime import datetime
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import cv2
import joblib
import io
import os
from utils.ocean_disaster_predict import predict_image, predict_video
from utils.ocean_disaster_predict import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ocean_disaster_model.pth")


# Load CNN model for images
try:
    cnn_model = load_model(MODEL_PATH)
    if cnn_model:
        cnn_model.eval()
        print("✅ CNN model loaded successfully")
    else:
        print("❌ CNN model is None after loading")
except Exception as e:
    print(f"❌ Failed to load CNN model: {e}")
    cnn_model = None

# Import your visualizer
from utils.visualizer import DisasterVisualizer

app = FastAPI()

# Root
@app.get("/")
def root():
    return {"message": "Disaster Monitoring API", "status": "active"}

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load ML models
try:
    model = joblib.load('models/disaster_classifier.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    model = None
    vectorizer = None


# Pydantic schema
class PredictionRequest(BaseModel):
    text: str
@app.post("/api/predict")
async def predict_disaster(request: PredictionRequest):
    """
    Predict if text is disaster-related and assign specific disaster type
    using the same keyword logic as the visualizer.
    """
    try:
        text_lower = str(request.text).lower()
        disaster_types = ['flood', 'cyclone', 'tsunami', 'storm','storm surges', 'high waves', 'swell surges', 'earthquake', 'landslide']
        
        # Check text for disaster keywords
        detected_types = [d for d in disaster_types if d in text_lower]
        
        if detected_types:
            predicted_type = ", ".join(detected_types)
            is_disaster = True
        else:
            predicted_type = "none"
            is_disaster = False

        # Optional: Use ML model to get confidence if available
        confidence = 1.0
        if model is not None and vectorizer is not None:
            try:
                X = vectorizer.transform([request.text])
                if hasattr(model, "predict_proba"):
                    confidence = max(model.predict_proba(X)[0])
            except:
                confidence = 1.0

        return {
            "text": request.text,
            "predicted_type": predicted_type,
            "is_disaster": is_disaster,
            "confidence": float(confidence),
            "timestamp": datetime.now()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# image prediction
@app.post("/api/predict-image")
async def predict_disaster_image(file: UploadFile = File(...)):
    if cnn_model is None:
        raise HTTPException(status_code=500, detail="CNN model not loaded")
    
    # Save uploaded file temporarily
    image_path = f"temp_{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())

    # Run prediction using your function
    predicted_class, confidence = predict_image(image_path, cnn_model)
    
    # Remove temp file
    os.remove(image_path)
    
    return JSONResponse({
        "predicted_class": predicted_class,
        "confidence": confidence
    })


# Video prediction endpoint
@app.post("/api/predict-video")
async def predict_disaster_video(file: UploadFile = File(...)):
    if cnn_model is None:
        raise HTTPException(status_code=500, detail="CNN model not loaded")
    
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())
    
    # Use the imported predict_video function with the correct signature
    predicted_class, confidence = predict_video(video_path, cnn_model)
    
    # Remove temp file
    os.remove(video_path)

    return JSONResponse({
        "predicted_class": predicted_class,
        "confidence": confidence
    })

# Initialize visualizer
visualizer = DisasterVisualizer()
DATA_TWEETS = "data/coastal_tweets_data.csv"
DATA_COMMENTS = "data/indian_coastal_comments.csv"
try:
    visualizer.load_data(tweet_path=DATA_TWEETS, youtube_comment_path=DATA_COMMENTS)
except Exception as e:
    print(f"❌ Failed to load visualizer data: {e}")

@app.get("/api/trends-dashboard")
def trends_dashboard():
    """
    Return the full trends dashboard image
    """
    try:
        output_path = "trends_dashboard.png"
        visualizer.create_trend_dashboard(output_path=output_path)
        
        if os.path.exists(output_path):
            return FileResponse(output_path, media_type="image/png")
        else:
            raise HTTPException(status_code=500, detail="Dashboard image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating dashboard: {str(e)}")

@app.get("/api/daily-trends-plot")
def daily_trends_plot():
    """
    Return only the daily trends plot
    """
    try:
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig, ax = plt.subplots(figsize=(8,5))
        visualizer.plot_daily_trends(ax)
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating plot: {str(e)}")

@app.get("/api/source-distribution-plot")
def source_distribution_plot():
    try:
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig, ax = plt.subplots(figsize=(6,6))
        visualizer.plot_source_distribution(ax)
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating plot: {str(e)}")

@app.get("/api/disaster-types-plot")
def disaster_types_plot():
    try:
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig, ax = plt.subplots(figsize=(8,5))
        visualizer.plot_disaster_types(ax)
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating plot: {str(e)}")

@app.get("/api/regional-distribution-plot")
def regional_distribution_plot():
    try:
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig, ax = plt.subplots(figsize=(8,5))
        visualizer.plot_regional_distribution(ax)
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating plot: {str(e)}")

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)