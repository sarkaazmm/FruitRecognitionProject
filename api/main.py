from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

from fastapi.middleware.cors import CORSMiddleware

# Global variable to store the loaded model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and cleanup on shutdown"""
    global model
    try:
        # Load YOLOv8 model
        model = YOLO("../model/last.pt")
        print(f"Model loaded successfully")
        print(f"Model type: {type(model)}")
        yield
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    finally:
        # Cleanup if needed
        model = None

app = FastAPI(
    title="YOLO Fruit Detection API",
    description="API for fruit detection using YOLOv8",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "model_type": "YOLOv8 Detection"
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
):
    """
    Detect fruits in an uploaded image
    
    Parameters:
    - file: Image file (jpg, png, etc.)
    - conf_threshold: Confidence threshold for detections (0.0-1.0)
    - iou_threshold: IoU threshold for NMS (0.0-1.0)
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read and open image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Run inference
        results = model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "class_id": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    }
                }
                detections.append(detection)
        
        return {
            "success": True,
            "image_size": {
                "width": image.width,
                "height": image.height
            },
            "detections_count": len(detections),
            "detections": detections
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/detailed")
async def predict_detailed(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25
):
    """
    Get detailed detection results with additional metrics
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        results = model.predict(image, conf=conf_threshold, verbose=False)
        
        response = {
            "success": True,
            "detections": []
        }
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Calculate center point and dimensions
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1
                center_x = x1 + width / 2
                center_y = y1 + height / 2
                
                detection = {
                    "class_id": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "width": float(width),
                        "height": float(height),
                        "center_x": float(center_x),
                        "center_y": float(center_y)
                    }
                }
                response["detections"].append(detection)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        return {
            "model_type": "YOLOv8",
            "task": model.task,
            "class_names": model.names,
            "num_classes": len(model.names)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/classes")
async def get_classes():
    """Get list of detectable classes"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "classes": model.names
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)