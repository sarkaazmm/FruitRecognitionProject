import os
from pathlib import Path
from select_photo import select_image_and_predict
import neptune
from ultralytics import YOLO, settings
from dotenv import load_dotenv
import yaml
import torch

load_dotenv()
NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")
NEPTUNE_PROJECT = "saienko.alona/yolo-fruits"

if NEPTUNE_API_TOKEN:
    try:
        os.environ['NEPTUNE_API_TOKEN'] = NEPTUNE_API_TOKEN
        os.environ['NEPTUNE_PROJECT'] = NEPTUNE_PROJECT
        
        settings.update({'neptune': True})
        
        USE_NEPTUNE = True
    except Exception as e:
        print(f"Neptune setup failed: {e}")
        USE_NEPTUNE = False
else:
    print("NEPTUNE_API_TOKEN not found")
    USE_NEPTUNE = False

try:
    with open('/home/alona/універ/4курс/FruitRecognitionProject/data/fruits_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        num_classes = config.get('nc', 'N/A')
        class_names = config.get('names', [])
        print(f"Dataset: {num_classes} classes")

except Exception as e:
    print(f"Config load error: {e}")
    num_classes = 'N/A'
    class_names = []

checkpoint_path = Path('yolo-fruits/exp_final/weights/last.pt')
resume_training = checkpoint_path.exists()

if resume_training:
    print(f"\nFound checkpoint: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        last_epoch = ckpt.get('epoch', -1)
        total_epochs = ckpt.get('best_fitness', {})
        
    except Exception as e:
        print(f"Could not read checkpoint info: {e}")

if resume_training:
    print("\nResuming training from checkpoint...")
    model = YOLO(str(checkpoint_path))

    results = model.train(resume=True)
    
else:
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='/home/alona/універ/4курс/FruitRecognitionProject/data/fruits_config.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        save=True,
        device=0,            
        project='yolo-fruits',
        name='exp_final',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',
        verbose=True,
        seed=42,
        deterministic=True,
        cos_lr=True,
        amp=True,
    )

metrics = model.val()

print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

if USE_NEPTUNE:

    results_dir = Path('yolo-fruits/exp_final')
