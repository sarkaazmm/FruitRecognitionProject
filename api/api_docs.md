# YOLO Fruit Detection API Documentation

## Overview

This API provides fruit detection capabilities using a YOLOv8 deep learning model. It accepts image uploads and returns detected fruits with their locations, confidence scores, and classifications.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required.

---

## Endpoints

### 1. Health Check

Check if the API and model are running properly.

**Endpoint:** `GET /`

**Response:**
```json
{
  "status": "online",
  "model_loaded": true,
  "model_type": "YOLOv8 Detection"
}
```

**Status Codes:**
- `200 OK` - Service is running

---

### 2. Basic Fruit Detection

Detect fruits in an uploaded image with basic information.

**Endpoint:** `POST /predict`

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| file | File | Yes | - | Image file (JPG, PNG, etc.) |
| conf_threshold | float | No | 0.25 | Confidence threshold (0.0-1.0) |
| iou_threshold | float | No | 0.45 | IoU threshold for NMS (0.0-1.0) |

**Request Example (cURL):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@apple.jpg" \
  -F "conf_threshold=0.25" \
  -F "iou_threshold=0.45"
```

**Request Example (Python):**
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("apple.jpg", "rb")}
params = {
    "conf_threshold": 0.25,
    "iou_threshold": 0.45
}

response = requests.post(url, files=files, params=params)
print(response.json())
```

**Response:**
```json
{
  "success": true,
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "detections_count": 2,
  "detections": [
    {
      "class_id": 0,
      "class_name": "apple",
      "confidence": 0.89,
      "bbox": {
        "x1": 100.5,
        "y1": 150.2,
        "x2": 250.8,
        "y2": 300.4
      }
    },
    {
      "class_id": 1,
      "class_name": "banana",
      "confidence": 0.76,
      "bbox": {
        "x1": 300.0,
        "y1": 200.5,
        "x2": 450.3,
        "y2": 350.8
      }
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Detection successful
- `500 Internal Server Error` - Model not loaded or prediction error

---

### 3. Detailed Fruit Detection

Get detailed detection results with additional metrics including center points and dimensions.

**Endpoint:** `POST /predict/detailed`

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| file | File | Yes | - | Image file (JPG, PNG, etc.) |
| conf_threshold | float | No | 0.25 | Confidence threshold (0.0-1.0) |

**Request Example (cURL):**
```bash
curl -X POST "http://localhost:8000/predict/detailed" \
  -F "file=@orange.jpg" \
  -F "conf_threshold=0.3"
```

**Request Example (JavaScript/Fetch):**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/predict/detailed?conf_threshold=0.3', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log(data);
```

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "class_id": 2,
      "class_name": "orange",
      "confidence": 0.92,
      "bbox": {
        "x1": 150.0,
        "y1": 200.0,
        "x2": 300.0,
        "y2": 350.0,
        "width": 150.0,
        "height": 150.0,
        "center_x": 225.0,
        "center_y": 275.0
      }
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Detection successful
- `500 Internal Server Error` - Model not loaded or prediction error

---

### 4. Model Information

Get information about the loaded YOLO model.

**Endpoint:** `GET /model/info`

**Response:**
```json
{
  "model_type": "YOLOv8",
  "task": "detect",
  "class_names": {
    "0": "Apple",
    "1": "Banana",
    "2": "Grape",
    "3": "Mango",
    "4": "Strawberry"
  },
  "num_classes": 5
}
```

**Status Codes:**
- `200 OK` - Information retrieved successfully
- `500 Internal Server Error` - Model not loaded

---

### 5. Get Detectable Classes

Get a list of all fruit classes the model can detect.

**Endpoint:** `GET /classes`

**Response:**
```json
{
  "classes": {
    "0": "Apple",
    "1": "Banana",
    "2": "Grape",
    "3": "Mango",
    "4": "Strawberry"
  }
}
```

**Status Codes:**
- `200 OK` - Classes retrieved successfully
- `500 Internal Server Error` - Model not loaded

---

## Response Objects

### Detection Object

| Field | Type | Description |
|-------|------|-------------|
| class_id | integer | Numeric identifier of the detected fruit class |
| class_name | string | Name of the detected fruit |
| confidence | float | Detection confidence score (0.0-1.0) |
| bbox | object | Bounding box coordinates |

### Bounding Box Object (Basic)

| Field | Type | Description |
|-------|------|-------------|
| x1 | float | Left coordinate |
| y1 | float | Top coordinate |
| x2 | float | Right coordinate |
| y2 | float | Bottom coordinate |

### Bounding Box Object (Detailed)

Includes all basic fields plus:

| Field | Type | Description |
|-------|------|-------------|
| width | float | Width of bounding box |
| height | float | Height of bounding box |
| center_x | float | X coordinate of center point |
| center_y | float | Y coordinate of center point |

---

## Error Handling

All endpoints return error responses in the following format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Codes

- `400 Bad Request` - Invalid input parameters
- `500 Internal Server Error` - Model not loaded or processing error

---

## Installation & Setup

### Requirements

```txt
fastapi
uvicorn[standard]
ultralytics
pillow
numpy
python-multipart
```

### Installation Steps

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your `last.pt` model file is in the correct path

3. Run the API:
```bash
python3 main.py
```


### Interactive Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- WebP (.webp)
- TIFF (.tiff)

---

## Troubleshooting

### Model Not Loading
- Verify `last.pt` file path is correct
- Ensure ultralytics is properly installed
- Check Python version compatibility (3.8+)

### Low Detection Accuracy
- Try adjusting `conf_threshold` (default: 0.25)
- Ensure image quality is good
- Check if fruits are clearly visible in the image

### Slow Response Times
- Consider using GPU if available
- Reduce image resolution before uploading
- Adjust `iou_threshold` for faster NMS
