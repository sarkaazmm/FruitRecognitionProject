import os
import random
import shutil
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import kagglehub
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split


class FruitsDatasetPreparer:
    def __init__(self, kaggle_dataset_path: str, output_path: str = 'fruits_dataset', val_ratio: float = 0.2):
        self.kaggle_path = Path(kaggle_dataset_path)
        self.output_path = Path(output_path)
        self.val_ratio = val_ratio

    def explore_dataset(self) -> Dict[str, List[Path]]:
        """
        Collect image paths and their class labels.
        Automatically supports Fruits-360 folder structure from KaggleHub.
        """
        print("📊 Exploring dataset structure...")

        # Try to find the real base folder containing 'Training'
        train_path = None
        for root, dirs, files in os.walk(self.kaggle_path):
            for d in dirs:
                if d.lower() in ['training', 'train']:
                    train_path = Path(root) / d
                    break
            if train_path:
                break

        if not train_path or not train_path.exists():
            raise FileNotFoundError("❌ Could not find 'Training' or 'train' folder inside the dataset.")

        # Check for optional Test folder
        test_path = train_path.parent / 'Test'
        all_images = []

        for base_path in [train_path, test_path] if test_path.exists() else [train_path]:
            for class_dir in tqdm(sorted(base_path.iterdir()), desc=f"Scanning {base_path.name}"):
                if class_dir.is_dir():
                    for img_path in class_dir.glob('*.jpg'):
                        all_images.append((img_path, class_dir.name))

        print(f"✅ Found {len(all_images)} images across {len(set(label for _, label in all_images))} classes.")
        return {"images": all_images}


    def visualize_eda(self, dataset_info: Dict[str, List[Tuple[Path, str]]]):
        """
        Perform EDA: class distribution, image shapes, and example visualization.
        """
        print("📈 Running EDA...")

        image_paths, labels = zip(*dataset_info['images'])
        label_counts = Counter(labels)

        df = pd.DataFrame({
            'label': labels,
            'path': image_paths
        })

        # Class distribution
        plt.figure(figsize=(14, 6))
        sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()))
        plt.xticks(rotation=90)
        plt.title("Class Distribution in Fruits-360 Dataset")
        plt.tight_layout()
        plt.savefig("class_distribution.png")
        plt.close()

        # Image size distribution (sample 100)
        sample_paths = random.sample(image_paths, min(100, len(image_paths)))
        sizes = [cv2.imread(str(p)).shape[:2] for p in sample_paths]
        heights, widths = zip(*sizes)
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=widths, y=heights)
        plt.title("Sampled Image Size Distribution")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.tight_layout()
        plt.savefig("image_sizes.png")
        plt.close()

        # Show some example images per class (optional)
        print("📸 Saving sample images per class...")
        sample_output = Path("samples")
        sample_output.mkdir(exist_ok=True)

        for label in random.sample(list(label_counts.keys()), min(10, len(label_counts))):
            class_images = [p for p, l in dataset_info['images'] if l == label]
            sample_img = cv2.imread(str(random.choice(class_images)))
            cv2.imwrite(str(sample_output / f"{label.replace(' ', '_')}.jpg"), sample_img)

        print("✅ EDA completed. Saved: class_distribution.png, image_sizes.png, /samples")

    def split_and_save_dataset(self, dataset_info: Dict[str, List[Tuple[Path, str]]]):
        """
        Split dataset into train/val preserving class ratios and build YOLO structure.
        """
        print("📦 Splitting dataset...")

        image_paths, labels = zip(*dataset_info['images'])
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=self.val_ratio, stratify=labels, random_state=42
        )

        # YOLO structure
        for subset in ['train', 'val']:
            for label in set(labels):
                (self.output_path / subset / label).mkdir(parents=True, exist_ok=True)

        print("🚚 Copying images into YOLOv8 structure...")
        for src, label in tqdm(zip(train_paths, train_labels), total=len(train_paths), desc="Train"):
            shutil.copy(src, self.output_path / 'train' / label / src.name)
        for src, label in tqdm(zip(val_paths, val_labels), total=len(val_paths), desc="Validation"):
            shutil.copy(src, self.output_path / 'val' / label / src.name)

        print(f"✅ Split complete: {len(train_paths)} train / {len(val_paths)} val")

    def create_yolo_config(self, dataset_info: Dict[str, List[Tuple[Path, str]]]):
        """
        Generate a YOLOv8 classification config YAML file.
        """
        labels = sorted(list(set(label for _, label in dataset_info['images'])))
        yaml_data = {
            'path': str(self.output_path),
            'train': 'train',
            'val': 'val',
            'nc': len(labels),
            'names': labels
        }

        config_path = self.output_path / 'fruits_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(yaml_data, f)

        print(f"✅ YOLO config created at {config_path}")

    def generate_report(self, dataset_info: Dict[str, List[Tuple[Path, str]]]):
        """
        Generate dataset summary and statistics.
        """
        labels = [label for _, label in dataset_info['images']]
        counter = Counter(labels)
        print("\n📊 Dataset Summary Report:")
        print(f"Total images: {len(labels)}")
        print(f"Classes: {len(counter)}")
        print(f"Top 10 classes:")
        for name, count in counter.most_common(10):
            print(f"  {name}: {count}")

    def run(self):
        info = self.explore_dataset()
        self.visualize_eda(info)
        self.split_and_save_dataset(info)
        self.create_yolo_config(info)
        self.generate_report(info)


# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":
    # Download latest version of Fruits-360
    print("⬇️ Downloading Fruits-360 via kagglehub...")
    path = kagglehub.dataset_download("moltean/fruits")
    print("Path to dataset files:", path)

    # Prepare dataset for YOLOv8 classification
    preparer = FruitsDatasetPreparer(
        kaggle_dataset_path=path,
        output_path='fruits_yolo_dataset',
        val_ratio=0.2
    )
    preparer.run()
