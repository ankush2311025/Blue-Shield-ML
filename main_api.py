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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ocean_disaster_model.pth")


# Import your visualizer
from utils.visualizer import DisasterVisualizer

app = FastAPI()

# ----------------------------
# Root
# ----------------------------
@app.get("/")
def root():
    return {"message": "Disaster Monitoring API", "status": "active"}

# ----------------------------
# CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Load ML models
# ----------------------------
try:
    model = joblib.load('models/disaster_classifier.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Failed to load models: {e}")

# ----------------------------
# Pydantic schema
# ----------------------------
class PredictionRequest(BaseModel):
    text: str

# ----------------------------
# Predict endpoint
# ----------------------------
@app.post("/api/predict")
async def predict_disaster(request: PredictionRequest):
    """
    Predict if text is disaster-related and assign specific disaster type
    using the same keyword logic as the visualizer.
    """
    try:
        text_lower = str(request.text).lower()
        disaster_types = ['flood', 'cyclone', 'tsunami', 'storm','storm surges,' 'high waves', 'swell surges' ,'earthquake', 'landslide']
        
        # Check text for disaster keywords
        detected_types = [d for d in disaster_types if d in text_lower]
        
        if detected_types:
            predicted_type = ", ".join(detected_types)
            is_disaster = True
        else:
            predicted_type = "none"
            is_disaster = False

        # Optional: Use ML model to get confidence if available
        try:
            X = vectorizer.transform([request.text])
            if hasattr(model, "predict_proba"):
                confidence = max(model.predict_proba(X)[0])
            else:
                confidence = 1.0
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
    # Save uploaded file temporarily
    image_path = f"temp_{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())
    
    # Run prediction using your function
    predicted_class, confidence = predict_image(image_path)
    
    # Remove temp file
    os.remove(image_path)
    
    return JSONResponse({
        "predicted_class": predicted_class,
        "confidence": confidence
    })


# ----------------------------
# Video prediction endpoint
# ----------------------------
@app.post("/api/predict-video")
async def predict_disaster_video(file: UploadFile = File(...)):
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())
    
    cap = cv2.VideoCapture(video_path)
    frame_predictions = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, frame_count // 10)  # Predict on ~10 frames for efficiency

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % sample_rate == 0:
            # Save temporary frame
            frame_path = f"temp_frame_{count}.jpg"
            cv2.imwrite(frame_path, frame)
            predicted_class, confidence = predict_video(frame_path)
            frame_predictions.append((predicted_class, confidence))
            os.remove(frame_path)
        count += 1
    
    cap.release()
    os.remove(video_path)

    # Aggregate predictions (majority vote)
    from collections import Counter
    if frame_predictions:
        classes = [p[0] for p in frame_predictions]
        most_common_class, _ = Counter(classes).most_common(1)[0]
        avg_confidence = sum([p[1] for p in frame_predictions if p[0]==most_common_class]) / classes.count(most_common_class)
    else:
        most_common_class = "none"
        avg_confidence = 0.0

    return JSONResponse({
        "predicted_class": most_common_class,
        "confidence": avg_confidence
    })



# ----------------------------
# Visualization endpoints
# ----------------------------
# Initialize visualizer
visualizer = DisasterVisualizer()
DATA_TWEETS = "data/coastal_tweets_data.csv"
DATA_COMMENTS = "data/indian_coastal_comments.csv"
visualizer.load_data(tweet_path=DATA_TWEETS, youtube_comment_path=DATA_COMMENTS)

@app.get("/api/trends-dashboard")
def trends_dashboard():
    """
    Return the full trends dashboard image
    """
    output_path = "trends_dashboard.png"
    visualizer.create_trend_dashboard(output_path=output_path)
    
    if os.path.exists(output_path):
        return FileResponse(output_path, media_type="image/png")
    else:
        raise HTTPException(status_code=500, detail="Dashboard image not found")

@app.get("/api/daily-trends-plot")
def daily_trends_plot():
    """
    Return only the daily trends plot
    """
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(8,5))
    visualizer.plot_daily_trends(ax)
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/api/source-distribution-plot")
def source_distribution_plot():
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(6,6))
    visualizer.plot_source_distribution(ax)
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/api/disaster-types-plot")
def disaster_types_plot():
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(8,5))
    visualizer.plot_disaster_types(ax)
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/api/regional-distribution-plot")
def regional_distribution_plot():
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(8,5))
    visualizer.plot_regional_distribution(ax)
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
