"""
Скрипт для завантаження та підготовки датасету фруктів з Kaggle
Датасет: https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification
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

# Налаштування
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class FruitsDatasetPreparer:
    def __init__(self, kaggle_dataset_path='Fruits Classification', output_path='fruits_dataset'):
        """
        Args:
            kaggle_dataset_path: шлях до завантаженого датасету з Kaggle
            output_path: шлях для YOLO датасету
        """
        self.kaggle_path = Path(kaggle_dataset_path)
        self.output_path = Path(output_path)
        self.train_split = 0.8
        self.val_split = 0.2
        
    def explore_dataset(self):
        """Проводить EDA датасету"""
        print("=" * 60)
        print("КРОК 2: Exploratory Data Analysis (EDA)")
        print("=" * 60)
        
        # Знаходимо всі зображення - спочатку перевіряємо train папку
        train_path = self.kaggle_path / 'train'
        
        if not train_path.exists():
            print(f"\n❌ ПОМИЛКА: Папка {train_path} не знайдена!")
            print(f"\n📁 Доступні папки у {self.kaggle_path}:")
            for item in self.kaggle_path.iterdir():
                print(f"  - {item.name} ({'папка' if item.is_dir() else 'файл'})")
            
            # Спробуємо знайти зображення безпосередньо в основній папці
            print(f"\n🔍 Пошук зображень у {self.kaggle_path}...")
            all_images = []
            classes = set()
            
            # Шукаємо всі зображення та визначаємо класи з імен папок
            for img_path in self.kaggle_path.rglob('*.jpg'):
                # Визначаємо клас з структури папок
                relative_path = img_path.relative_to(self.kaggle_path)
                parts = relative_path.parts
                if len(parts) >= 2:
                    class_name = parts[-2]  # Папка, в якій знаходиться зображення
                else:
                    class_name = 'unknown'
                
                classes.add(class_name)
                all_images.append((str(img_path), class_name))
            
            if all_images:
                print(f"✓ Знайдено {len(all_images)} зображень у {len(classes)} класах")
                classes = sorted(list(classes))
                return all_images, classes
            else:
                print("\n❌ Не знайдено жодного зображення!")
                return None, None
            
        # Якщо train папка існує
        classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
        print(f"\n📊 Знайдено класів: {len(classes)}")
        print(f"Класи: {', '.join(classes)}\n")
        
        # Збираємо статистику
        class_counts = {}
        image_sizes = []
        all_images = []
        
        print("Аналіз зображень...")
        for class_name in tqdm(classes):
            class_path = train_path / class_name
            images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png')) + list(class_path.glob('*.jpeg'))
            class_counts[class_name] = len(images)
            
            # Аналіз розмірів (вибірково для швидкості)
            for img_path in images[:50]:  # Перші 50 зображень кожного класу
                try:
                    with Image.open(img_path) as img:
                        image_sizes.append(img.size)
                except Exception as e:
                    print(f"Помилка читання {img_path}: {e}")
                    
            all_images.extend([(str(img_path), class_name) for img_path in images])
        
        # Створення графіків EDA
        self._create_eda_plots(class_counts, image_sizes, all_images, train_path)
        
        print(f"\n📈 Загальна статистика:")
        print(f"  Всього зображень: {sum(class_counts.values())}")
        print(f"  Середнє зображень на клас: {np.mean(list(class_counts.values())):.1f}")
        print(f"  Мін/Макс зображень на клас: {min(class_counts.values())}/{max(class_counts.values())}")
        
        if image_sizes:
            widths = [s[0] for s in image_sizes]
            heights = [s[1] for s in image_sizes]
            print(f"\n📏 Розміри зображень (на основі вибірки):")
            print(f"  Ширина: {np.mean(widths):.0f}±{np.std(widths):.0f} px (мін: {min(widths)}, макс: {max(widths)})")
            print(f"  Висота: {np.mean(heights):.0f}±{np.std(heights):.0f} px (мін: {min(heights)}, макс: {max(heights)})")
        
        return all_images, classes
        
    def _create_eda_plots(self, class_counts, image_sizes, all_images, train_path):
        """Створює реальні графіки для EDA з датасету"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle("Exploratory Data Analysis (EDA) — Fruits Dataset", fontsize=18, fontweight='bold')

        # 1️⃣ Розподіл зображень по класах
        ax1 = axes[0, 0]
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        class_names = [x[0] for x in sorted_classes]
        counts = [x[1] for x in sorted_classes]

        sns.barplot(x=counts, y=class_names, ax=ax1, palette="viridis")
        ax1.set_title("Розподіл зображень по класах", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Кількість зображень", fontsize=12)
        ax1.set_ylabel("Клас", fontsize=12)

        # 2️⃣ Розподіл розмірів зображень
        if image_sizes:
            ax2 = axes[0, 1]
            widths = [s[0] for s in image_sizes]
            heights = [s[1] for s in image_sizes]

            sns.scatterplot(x=widths, y=heights, ax=ax2, alpha=0.6, s=25, color='coral', edgecolor=None)
            ax2.set_title("Розподіл розмірів зображень", fontsize=14, fontweight='bold')
            ax2.set_xlabel("Ширина (px)", fontsize=12)
            ax2.set_ylabel("Висота (px)", fontsize=12)
            ax2.grid(True, alpha=0.3)

        # 3️⃣ Гістограма балансу класів
        ax3 = axes[1, 0]
        sns.histplot(list(class_counts.values()), bins=15, kde=False, ax=ax3, color="mediumseagreen", edgecolor="black")
        ax3.axvline(np.mean(list(class_counts.values())), color='red', linestyle='--', linewidth=2,
                    label=f"Середнє: {np.mean(list(class_counts.values())):.0f}")
        ax3.axvline(np.median(list(class_counts.values())), color='orange', linestyle='--', linewidth=2,
                    label=f"Медіана: {np.median(list(class_counts.values())):.0f}")
        ax3.legend()
        ax3.set_title("Гістограма розподілу кількості зображень на клас", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Кількість зображень на клас", fontsize=12)
        ax3.set_ylabel("Кількість класів", fontsize=12)

        # 4️⃣ Приклади зображень
        ax4 = axes[1, 1]
        ax4.axis("off")
        ax4.set_title("Приклади зображень з різних класів", fontsize=14, fontweight='bold', pad=10)

        examples = []
        for class_name in list(class_counts.keys())[:6]:
            class_images = [img for img, cls in all_images if cls == class_name]
            if class_images:
                examples.append((class_images[0], class_name))

        if examples:
            example_fig, example_axes = plt.subplots(2, 3, figsize=(12, 8))
            example_axes = example_axes.flatten()

            for idx, (img_path, class_name) in enumerate(examples):
                try:
                    img = Image.open(img_path)
                    example_axes[idx].imshow(img)
                    example_axes[idx].set_title(class_name, fontsize=11, fontweight='bold')
                    example_axes[idx].axis('off')
                except Exception as e:
                    example_axes[idx].set_title(f"Помилка для {class_name}")
                    example_axes[idx].axis('off')

            plt.tight_layout()
            plt.savefig('eda_examples.png', dpi=150, bbox_inches='tight')
            plt.show()

        # Зберігаємо основні графіки
        plt.tight_layout()
        plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("✅ Збережено: eda_analysis.png (основні графіки)")
        print("✅ Збережено: eda_examples.png (приклади зображень)")

        
    def split_dataset(self, all_images, classes):
        """Розділяє датасет на train/val зі стратифікацією"""
        print("\n" + "=" * 60)
        print("КРОК 3: Розділення на train/val (80/20) зі стратифікацією")
        print("=" * 60)
        
        # Створюємо списки зображень та міток
        image_paths = [img[0] for img in all_images]
        labels = [img[1] for img in all_images]
        
        # Стратифіковане розділення
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            image_paths, 
            labels,
            test_size=self.val_split,
            stratify=labels,
            random_state=42
        )
        
        print(f"\n📊 Результати розділення:")
        print(f"  Train: {len(train_imgs)} зображень ({len(train_imgs)/len(all_images)*100:.1f}%)")
        print(f"  Val: {len(val_imgs)} зображень ({len(val_imgs)/len(all_images)*100:.1f}%)")
        
        # Перевірка стратифікації
        print(f"\n✓ Розподіл по класах:")
        train_dist = Counter(train_labels)
        val_dist = Counter(val_labels)
        
        for class_name in sorted(classes):
            train_count = train_dist.get(class_name, 0)
            val_count = val_dist.get(class_name, 0)
            total = train_count + val_count
            print(f"  {class_name:15s}: train={train_count:4d} ({train_count/total*100:.1f}%), val={val_count:4d} ({val_count/total*100:.1f}%)")
        
        return list(zip(train_imgs, train_labels)), list(zip(val_imgs, val_labels))
        
    def convert_to_yolo(self, train_data, val_data, classes):
        """Конвертує датасет у формат YOLO"""
        print("\n" + "=" * 60)
        print("КРОК 4: Конвертація у формат YOLO")
        print("=" * 60)
        
        # Створюємо структуру YOLO
        yolo_structure = {
            'images': {
                'train': self.output_path / 'images' / 'train',
                'val': self.output_path / 'images' / 'val'
            },
            'labels': {
                'train': self.output_path / 'labels' / 'train',
                'val': self.output_path / 'labels' / 'val'
            }
        }
        
        # Створюємо папки
        for split_type in ['train', 'val']:
            yolo_structure['images'][split_type].mkdir(parents=True, exist_ok=True)
            yolo_structure['labels'][split_type].mkdir(parents=True, exist_ok=True)
        
        # Мапа класів до індексів
        class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(classes))}
        
        print(f"\n📋 Мапа класів:")
        for class_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
            print(f"  {idx}: {class_name}")
        
        # Копіюємо зображення та створюємо анотації
        print("\n📦 Копіювання зображень та створення анотацій...")
        
        for split_name, data in [('train', train_data), ('val', val_data)]:
            print(f"\nОбробка {split_name}...")
            for img_path, class_name in tqdm(data):
                try:
                    # Копіюємо зображення
                    src_path = Path(img_path)
                    dst_img_path = yolo_structure['images'][split_name] / src_path.name
                    shutil.copy2(src_path, dst_img_path)
                    
                    # Створюємо YOLO анотацію (для класифікації - просто клас по центру)
                    class_idx = class_to_idx[class_name]
                    label_path = yolo_structure['labels'][split_name] / (src_path.stem + '.txt')
                    
                    # Формат YOLO: class_id x_center y_center width height (нормалізовані 0-1)
                    # Для класифікації використовуємо bbox на все зображення
                    with open(label_path, 'w') as f:
                        f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")
                        
                except Exception as e:
                    print(f"Помилка обробки {img_path}: {e}")
        
        print(f"\n✓ Структура YOLO створена у папці: {self.output_path}")
        
        return class_to_idx
        
    def create_config_yaml(self, class_to_idx):
        """Створює конфігураційний файл для YOLO"""
        print("\n" + "=" * 60)
        print("КРОК 5: Створення fruits_config.yaml")
        print("=" * 60)
        
        config = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(class_to_idx),
            'names': {idx: name for name, idx in class_to_idx.items()}
        }
        
        config_path = 'fruits_config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print(f"\n✓ Конфігурація збережена у: {config_path}")
        print(f"\n📄 Вміст конфігурації:")
        print("-" * 60)
        with open(config_path, 'r', encoding='utf-8') as f:
            print(f.read())
        print("-" * 60)
        
        return config_path
        
    def generate_summary(self, class_to_idx):
        """Генерує підсумковий звіт"""
        print("\n" + "=" * 60)
        print("ПІДСУМОК")
        print("=" * 60)
        
        train_count = len(list((self.output_path / 'images' / 'train').glob('*')))
        val_count = len(list((self.output_path / 'images' / 'val').glob('*')))
        
        summary = f"""
✓ Датасет успішно підготовлено!

📊 Статистика:
  • Класів: {len(class_to_idx)}
  • Train зображень: {train_count}
  • Val зображень: {val_count}
  • Загально: {train_count + val_count}

📁 Створені файли та папки:
  • {self.output_path}/ - датасет у форматі YOLO
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
  • fruits_config.yaml - конфігураційний файл
  • eda_analysis.png - графіки EDA
  • eda_examples.png - приклади зображень

🚀 Наступні кроки:
  1. Перевірте датасет: {self.output_path}/
  2. Використовуйте fruits_config.yaml для тренування моделі
  3. Передайте папку fruits_dataset/ та fruits_config.yaml розробнику моделі

📝 Приклад використання для тренування YOLOv8:
  from ultralytics import YOLO
  model = YOLO('yolov8n-cls.pt')  # для класифікації
  results = model.train(data='fruits_config.yaml', epochs=100)
"""
        
        print(summary)
        
        # Зберігаємо summary у файл
        with open('dataset_summary.txt', 'w', encoding='utf-8') as f:
            f.write(summary)
        print("✓ Підсумок збережено у: dataset_summary.txt")


def main():
    """Головна функція"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║   Підготовка датасету фруктів для розпізнавання об'єктів    ║
║   Dataset: Fruits Classification (Kaggle)                    ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Ініціалізація
    preparer = FruitsDatasetPreparer(
        kaggle_dataset_path='Fruits Classification',  # Змінено на правильну назву папки
        output_path='fruits_dataset'
    )
    
    # Крок 2: EDA
    all_images, classes = preparer.explore_dataset()
    
    if all_images is None:
        print("\n⚠ Не вдалося знайти датасет. Перевірте структуру папок.")
        return
    
    # Крок 3: Розділення
    train_data, val_data = preparer.split_dataset(all_images, classes)
    
    # Крок 4: Конвертація у YOLO
    class_to_idx = preparer.convert_to_yolo(train_data, val_data, classes)
    
    # Крок 5: Створення конфігурації
    preparer.create_config_yaml(class_to_idx)
    
    # Підсумок
    preparer.generate_summary(class_to_idx)
    
    print("\n✨ Готово! Датасет підготовлено до використання.")


if __name__ == "__main__":
    main()