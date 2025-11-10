"""
Скрипт для завантаження та підготовки датасету фруктів/овочів з Kaggle
Датасет: https://www.kaggle.com/datasets/kvnpatel/fruits-vegetable-detection-for-yolov4
Запуск: python prepare_yolo_fruits_dataset.py
"""

import os
import shutil
import yaml
import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import kagglehub

# Налаштування
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class YOLOFruitsDatasetPreparer:
    def __init__(self, output_path='fruits_vegetable_dataset'):
        """
        Args:
            output_path: шлях для підготовленого YOLO датасету
        """
        self.kaggle_path = None
        self.output_path = Path(output_path)
        self.train_split = 0.8
        self.val_split = 0.2
        
    def download_dataset(self):
        """Завантажує датасет з Kaggle через kagglehub"""
        print("=" * 60)
        print("КРОК 1: Завантаження датасету з Kaggle")
        print("=" * 60)
        
        try:
            print("\n🔄 Завантаження датасету...")
            path = kagglehub.dataset_download("kvnpatel/fruits-vegetable-detection-for-yolov4")
            self.kaggle_path = Path(path)
            print(f"✓ Датасет завантажено у: {self.kaggle_path}")
            
            # Виводимо структуру датасету
            print(f"\n📁 Структура датасету:")
            for item in sorted(self.kaggle_path.rglob('*'))[:20]:
                if item.is_file():
                    rel_path = item.relative_to(self.kaggle_path)
                    print(f"  {rel_path}")
            
            return True
        except Exception as e:
            print(f"❌ Помилка завантаження датасету: {e}")
            return False
        
    def explore_dataset(self):
        """Проводить EDA датасету YOLO"""
        print("\n" + "=" * 60)
        print("КРОК 2: Exploratory Data Analysis (EDA)")
        print("=" * 60)
        
        # Пошук YOLO структури (images та labels)
        images_dirs = list(self.kaggle_path.rglob('images'))
        labels_dirs = list(self.kaggle_path.rglob('labels'))
        
        print(f"\n🔍 Пошук структури YOLO...")
        print(f"  Знайдено папок 'images': {len(images_dirs)}")
        print(f"  Знайдено папок 'labels': {len(labels_dirs)}")
        
        # Збираємо всі зображення та анотації
        all_images = []
        all_labels = []
        class_names_set = set()
        
        # Варіант 1: Якщо є структура images/labels
        if images_dirs and labels_dirs:
            print("\n✓ Знайдено YOLO структуру")
            
            for images_dir in images_dirs:
                # Шукаємо відповідну папку labels
                labels_dir = images_dir.parent / 'labels' / images_dir.name
                if not labels_dir.exists():
                    labels_dir = images_dir.parent.parent / 'labels' / images_dir.name
                
                if labels_dir.exists():
                    print(f"\n  Обробка: {images_dir}")
                    images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
                    
                    for img_path in images:
                        label_path = labels_dir / (img_path.stem + '.txt')
                        if label_path.exists():
                            all_images.append(str(img_path))
                            all_labels.append(str(label_path))
        
        # Варіант 2: Пошук всіх зображень та label файлів
        else:
            print("\n🔍 Пошук всіх зображень та анотацій...")
            
            for img_path in self.kaggle_path.rglob('*.jpg'):
                # Шукаємо відповідний label файл
                possible_label_paths = [
                    img_path.parent / (img_path.stem + '.txt'),
                    img_path.parent.parent / 'labels' / img_path.parent.name / (img_path.stem + '.txt'),
                ]
                
                for label_path in possible_label_paths:
                    if label_path.exists():
                        all_images.append(str(img_path))
                        all_labels.append(str(label_path))
                        break
        
        if not all_images:
            print("\n❌ Не знайдено зображень з анотаціями!")
            return None, None, None
        
        print(f"\n✓ Знайдено {len(all_images)} зображень з анотаціями")
        
        # Аналіз класів та bbox
        class_counts = Counter()
        bbox_stats = []
        image_sizes = []
        objects_per_image = []
        
        print("\n📊 Аналіз анотацій...")
        for img_path, label_path in tqdm(zip(all_images, all_labels), total=len(all_images)):
            try:
                # Читаємо розмір зображення
                with Image.open(img_path) as img:
                    image_sizes.append(img.size)
                
                # Читаємо анотації
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    objects_per_image.append(len(lines))
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            class_counts[class_id] += 1
                            bbox_stats.append({
                                'class_id': class_id,
                                'width': width,
                                'height': height,
                                'area': width * height
                            })
                            class_names_set.add(class_id)
            except Exception as e:
                print(f"Помилка обробки {img_path}: {e}")
        
        # Створюємо список класів (якщо немає yaml файлу)
        classes = sorted(list(class_names_set))
        class_to_name = {i: f"class_{i}" for i in classes}
        
        # Спроба знайти names.txt або data.yaml
        names_file = self.kaggle_path / 'names.txt'
        yaml_file = self.kaggle_path / 'data.yaml'
        
        if yaml_file.exists():
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'names' in data:
                        class_to_name = {i: name for i, name in enumerate(data['names'])}
                        print(f"\n✓ Завантажено назви класів з data.yaml")
            except:
                pass
        elif names_file.exists():
            try:
                with open(names_file, 'r') as f:
                    names = [line.strip() for line in f.readlines()]
                    class_to_name = {i: name for i, name in enumerate(names)}
                    print(f"\n✓ Завантажено назви класів з names.txt")
            except:
                pass
        
        # Створення графіків EDA
        self._create_eda_plots(class_counts, class_to_name, image_sizes, bbox_stats, objects_per_image)
        
        # Статистика
        print(f"\n📈 Загальна статистика:")
        print(f"  Всього зображень: {len(all_images)}")
        print(f"  Всього об'єктів: {sum(class_counts.values())}")
        print(f"  Унікальних класів: {len(classes)}")
        print(f"  Середня кількість об'єктів на зображенні: {np.mean(objects_per_image):.1f}")
        
        if image_sizes:
            widths = [s[0] for s in image_sizes]
            heights = [s[1] for s in image_sizes]
            print(f"\n📏 Розміри зображень:")
            print(f"  Ширина: {np.mean(widths):.0f}±{np.std(widths):.0f} px (мін: {min(widths)}, макс: {max(widths)})")
            print(f"  Висота: {np.mean(heights):.0f}±{np.std(heights):.0f} px (мін: {min(heights)}, макс: {max(heights)})")
        
        print(f"\n📋 Розподіл по класах:")
        for class_id in sorted(class_counts.keys()):
            class_name = class_to_name.get(class_id, f"class_{class_id}")
            count = class_counts[class_id]
            print(f"  {class_id}: {class_name:20s} - {count:5d} об'єктів ({count/sum(class_counts.values())*100:.1f}%)")
        
        return list(zip(all_images, all_labels)), classes, class_to_name
        
    def _create_eda_plots(self, class_counts, class_to_name, image_sizes, bbox_stats, objects_per_image):
        """Створює графіки для EDA"""
        
        # 1️⃣ Розподіл об'єктів по класах
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        class_ids = [x[0] for x in sorted_classes]
        class_names = [class_to_name.get(x[0], f"class_{x[0]}") for x in sorted_classes]
        counts = [x[1] for x in sorted_classes]

        fig1, ax1 = plt.subplots(figsize=(12, max(6, len(class_names) * 0.3)))
        colors = sns.color_palette("viridis", len(class_names))
        bars = ax1.barh(range(len(class_names)), counts, color=colors)
        ax1.set_yticks(range(len(class_names)))
        ax1.set_yticklabels(class_names)
        ax1.set_xlabel("Кількість об'єктів", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Клас", fontsize=12, fontweight='bold')
        ax1.set_title("Розподіл об'єктів по класах", fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        # Додаємо значення на стовпчики
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax1.text(count, i, f' {count}', va='center', fontsize=9)
        
        plt.tight_layout()
        fig1.savefig('eda_class_distribution.png', dpi=150, bbox_inches='tight')
        plt.close(fig1)

        # 2️⃣ Розподіл розмірів зображень
        if image_sizes:
            widths = [s[0] for s in image_sizes]
            heights = [s[1] for s in image_sizes]

            fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Scatter plot
            ax2a.scatter(widths, heights, alpha=0.5, s=20, color='coral')
            ax2a.set_xlabel("Ширина (px)", fontsize=12)
            ax2a.set_ylabel("Висота (px)", fontsize=12)
            ax2a.set_title("Розподіл розмірів зображень", fontsize=13, fontweight='bold')
            ax2a.grid(True, alpha=0.3)
            
            # Гістограми
            ax2b.hist(widths, bins=30, alpha=0.6, label='Ширина', color='skyblue', edgecolor='black')
            ax2b.hist(heights, bins=30, alpha=0.6, label='Висота', color='lightcoral', edgecolor='black')
            ax2b.set_xlabel("Розмір (px)", fontsize=12)
            ax2b.set_ylabel("Частота", fontsize=12)
            ax2b.set_title("Гістограма розмірів", fontsize=13, fontweight='bold')
            ax2b.legend()
            ax2b.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig2.savefig('eda_image_sizes.png', dpi=150, bbox_inches='tight')
            plt.close(fig2)

        # 3️⃣ Статистика bbox
        if bbox_stats:
            fig3, axes3 = plt.subplots(2, 2, figsize=(15, 12))
            
            # Розподіл розмірів bbox
            widths_bbox = [b['width'] for b in bbox_stats]
            heights_bbox = [b['height'] for b in bbox_stats]
            areas = [b['area'] for b in bbox_stats]
            
            # Width distribution
            axes3[0, 0].hist(widths_bbox, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            axes3[0, 0].set_xlabel("Ширина bbox (нормалізована)", fontsize=11)
            axes3[0, 0].set_ylabel("Частота", fontsize=11)
            axes3[0, 0].set_title("Розподіл ширини bbox", fontsize=12, fontweight='bold')
            axes3[0, 0].grid(True, alpha=0.3)
            
            # Height distribution
            axes3[0, 1].hist(heights_bbox, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
            axes3[0, 1].set_xlabel("Висота bbox (нормалізована)", fontsize=11)
            axes3[0, 1].set_ylabel("Частота", fontsize=11)
            axes3[0, 1].set_title("Розподіл висоти bbox", fontsize=12, fontweight='bold')
            axes3[0, 1].grid(True, alpha=0.3)
            
            # Area distribution
            axes3[1, 0].hist(areas, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
            axes3[1, 0].set_xlabel("Площа bbox (нормалізована)", fontsize=11)
            axes3[1, 0].set_ylabel("Частота", fontsize=11)
            axes3[1, 0].set_title("Розподіл площі bbox", fontsize=12, fontweight='bold')
            axes3[1, 0].grid(True, alpha=0.3)
            
            # Aspect ratio
            aspect_ratios = [w/h if h > 0 else 0 for w, h in zip(widths_bbox, heights_bbox)]
            axes3[1, 1].hist(aspect_ratios, bins=50, color='orchid', edgecolor='black', alpha=0.7)
            axes3[1, 1].set_xlabel("Співвідношення сторін (width/height)", fontsize=11)
            axes3[1, 1].set_ylabel("Частота", fontsize=11)
            axes3[1, 1].set_title("Співвідношення сторін bbox", fontsize=12, fontweight='bold')
            axes3[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig3.savefig('eda_bbox_stats.png', dpi=150, bbox_inches='tight')
            plt.close(fig3)

        # 4️⃣ Кількість об'єктів на зображення
        if objects_per_image:
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            ax4.hist(objects_per_image, bins=range(0, max(objects_per_image)+2), 
                     color='teal', edgecolor='black', alpha=0.7)
            mean_obj = np.mean(objects_per_image)
            ax4.axvline(mean_obj, color='red', linestyle='--', linewidth=2,
                        label=f'Середнє: {mean_obj:.1f}')
            ax4.set_xlabel("Кількість об'єктів на зображенні", fontsize=12)
            ax4.set_ylabel("Кількість зображень", fontsize=12)
            ax4.set_title("Розподіл кількості об'єктів на зображення", fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            plt.tight_layout()
            fig4.savefig('eda_objects_per_image.png', dpi=150, bbox_inches='tight')
            plt.close(fig4)
        
        print("\n✅ Графіки EDA збережено:")
        print("   • eda_class_distribution.png")
        print("   • eda_image_sizes.png")
        print("   • eda_bbox_stats.png")
        print("   • eda_objects_per_image.png")
        
    def split_dataset(self, all_data):
        """Розділяє датасет на train/val"""
        print("\n" + "=" * 60)
        print("КРОК 3: Розділення на train/val (80/20)")
        print("=" * 60)
        
        # Розділення
        train_data, val_data = train_test_split(
            all_data,
            test_size=self.val_split,
            random_state=42
        )
        
        print(f"\n📊 Результати розділення:")
        print(f"  Train: {len(train_data)} зображень ({len(train_data)/len(all_data)*100:.1f}%)")
        print(f"  Val: {len(val_data)} зображень ({len(val_data)/len(all_data)*100:.1f}%)")
        
        return train_data, val_data
        
    def create_yolo_structure(self, train_data, val_data, class_to_name):
        """Створює структуру YOLO датасету"""
        print("\n" + "=" * 60)
        print("КРОК 4: Створення структури YOLO датасету")
        print("=" * 60)
        
        # Створюємо структуру
        for split_name in ['train', 'val']:
            (self.output_path / 'images' / split_name).mkdir(parents=True, exist_ok=True)
            (self.output_path / 'labels' / split_name).mkdir(parents=True, exist_ok=True)
        
        # Копіюємо файли
        print("\n📦 Копіювання файлів...")
        
        for split_name, data in [('train', train_data), ('val', val_data)]:
            print(f"\nОбробка {split_name}: {len(data)} зображень")
            
            for img_path, label_path in tqdm(data):
                try:
                    # Копіюємо зображення
                    src_img = Path(img_path)
                    dst_img = self.output_path / 'images' / split_name / src_img.name
                    shutil.copy2(src_img, dst_img)
                    
                    # Копіюємо анотацію
                    src_label = Path(label_path)
                    dst_label = self.output_path / 'labels' / split_name / src_label.name
                    shutil.copy2(src_label, dst_label)
                    
                except Exception as e:
                    print(f"Помилка копіювання {img_path}: {e}")
        
        print(f"\n✓ Структура YOLO створена у: {self.output_path}")
        
    def create_config_yaml(self, class_to_name):
        """Створює конфігураційний файл data.yaml"""
        print("\n" + "=" * 60)
        print("КРОК 5: Створення data.yaml")
        print("=" * 60)
        
        config = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(class_to_name),
            'names': [class_to_name[i] for i in sorted(class_to_name.keys())]
        }
        
        config_path = self.output_path / 'data.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print(f"\n✓ Конфігурація збережена у: {config_path}")
        print(f"\n📄 Вміст:")
        print("-" * 60)
        with open(config_path, 'r', encoding='utf-8') as f:
            print(f.read())
        print("-" * 60)
        
        return config_path
        
    def generate_summary(self, class_to_name):
        """Генерує підсумковий звіт"""
        print("\n" + "=" * 60)
        print("ПІДСУМОК")
        print("=" * 60)
        
        train_count = len(list((self.output_path / 'images' / 'train').glob('*')))
        val_count = len(list((self.output_path / 'images' / 'val').glob('*')))
        
        summary = f"""
✓ Датасет успішно підготовлено!

📊 Статистика:
  • Класів: {len(class_to_name)}
  • Train зображень: {train_count}
  • Val зображень: {val_count}
  • Загально: {train_count + val_count}

📁 Створені файли та папки:
  • {self.output_path}/ - датасет у форматі YOLO
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml - конфігураційний файл
  • eda_*.png - графіки аналізу датасету

🚀 Наступні кроки:
  1. Перевірте датасет: {self.output_path}/
  2. Використовуйте data.yaml для тренування моделі

📝 Приклад використання для тренування YOLOv8:
  from ultralytics import YOLO
  
  # Для детекції об'єктів
  model = YOLO('yolov8n.pt')
  results = model.train(
      data='{self.output_path / "data.yaml"}',
      epochs=100,
      imgsz=640,
      batch=16
  )
  
  # Для валідації
  metrics = model.val()
  
  # Для предикції
  results = model.predict(source='path/to/image.jpg')

📋 Класи у датасеті:
"""
        
        for class_id in sorted(class_to_name.keys()):
            summary += f"  {class_id}: {class_to_name[class_id]}\n"
        
        print(summary)
        
        # Зберігаємо у файл
        with open('dataset_summary.txt', 'w', encoding='utf-8') as f:
            f.write(summary)
        print("✓ Підсумок збережено у: dataset_summary.txt")


def main():
    """Головна функція"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║   Підготовка датасету фруктів/овочів для YOLO детекції     ║
║   Dataset: Fruits Vegetable Detection for YOLOv4 (Kaggle)   ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Ініціалізація
    preparer = YOLOFruitsDatasetPreparer(output_path='fruits_vegetable_dataset')
    
    # Крок 1: Завантаження
    if not preparer.download_dataset():
        print("\n⚠ Не вдалося завантажити датасет.")
        return
    
    # Крок 2: EDA
    all_data, classes, class_to_name = preparer.explore_dataset()
    
    if all_data is None:
        print("\n⚠ Не вдалося проаналізувати датасет.")
        return
    
    # Крок 3: Розділення
    train_data, val_data = preparer.split_dataset(all_data)
    
    # Крок 4: Створення структури
    preparer.create_yolo_structure(train_data, val_data, class_to_name)
    
    # Крок 5: Конфігурація
    preparer.create_config_yaml(class_to_name)
    
    # Підсумок
    preparer.generate_summary(class_to_name)
    
    print("\n✨ Готово! Датасет підготовлено до використання.")


if __name__ == "__main__":
    main()