const API_URL = 'http://localhost:8000';

const elements = {
    fileInput: document.getElementById('fileInput'),
    uploadSection: document.getElementById('uploadSection'),
    detectBtn: document.getElementById('detectBtn'),
    confThreshold: document.getElementById('confThreshold'),
    iouThreshold: document.getElementById('iouThreshold'),
    confValue: document.getElementById('confValue'),
    iouValue: document.getElementById('iouValue'),
    resultsSection: document.getElementById('results'),
    errorMsg: document.getElementById('errorMsg'),
    stats: document.getElementById('stats'),
    originalImage: document.getElementById('originalImage'),
    resultCanvas: document.getElementById('resultCanvas'),
    detectionsList: document.getElementById('detectionsList'),
    mobileMenu: document.querySelector('.mobile-menu'),
    mobileNav: document.querySelector('.mobile-nav')
};

let selectedFile = null;

const FRUIT_COLORS = {
    'Apple': '#FF6B6B',
    'Banana': '#FFE66D',
    'Grape': '#8B5CF6',
    'Mango': '#FFA500',
    'Strawberry': '#FF1744'
};

function init() {
    setupEventListeners();
    checkAPIStatus();
}

function setupEventListeners() {
    elements.uploadSection.addEventListener('click', () => {
        elements.fileInput.click();
    });

    elements.fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file);
        }
    });

    elements.uploadSection.addEventListener('dragover', handleDragOver);
    elements.uploadSection.addEventListener('dragleave', handleDragLeave);
    elements.uploadSection.addEventListener('drop', handleDrop);

    elements.confThreshold.addEventListener('input', (e) => {
        elements.confValue.textContent = e.target.value;
    });

    elements.iouThreshold.addEventListener('input', (e) => {
        elements.iouValue.textContent = e.target.value;
    });

    elements.detectBtn.addEventListener('click', detectFruits);

    elements.mobileMenu.addEventListener('click', () => {
        elements.mobileMenu.classList.toggle('active');
        elements.mobileNav.classList.toggle('active');
    });

    document.querySelectorAll('.mobile-nav a, .nav-links a').forEach(link => {
        link.addEventListener('click', (e) => {
            if (link.getAttribute('href').startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(link.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                    elements.mobileNav.classList.remove('active');
                    elements.mobileMenu.classList.remove('active');
                }
            }
        });
    });
}

function handleDragOver(e) {
    e.preventDefault();
    elements.uploadSection.classList.add('dragover');
}

function handleDragLeave() {
    elements.uploadSection.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadSection.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileSelect(file);
    }
}

function handleFileSelect(file) {
    selectedFile = file;
    elements.detectBtn.disabled = false;
    elements.uploadSection.querySelector('.upload-text').textContent = file.name;
    hideResults();
    hideError();
}

async function detectFruits() {
    if (!selectedFile) return;

    setLoadingState(true);
    hideError();
    hideResults();

    const formData = new FormData();
    formData.append('file', selectedFile);

    const params = new URLSearchParams({
        conf_threshold: elements.confThreshold.value,
        iou_threshold: elements.iouThreshold.value
    });

    try {
        const response = await fetch(`${API_URL}/predict/detailed?${params}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('API Error');
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        showError('Помилка з\'єднання з API. Переконайтеся, що сервер запущено на http://localhost:8000');
        console.error('Detection error:', error);
    } finally {
        setLoadingState(false);
    }
}

function displayResults(data) {
    elements.resultsSection.style.display = 'block';
    displayStats(data.detections);
    displayImages(data.detections);
    displayDetectionsList(data.detections);

    setTimeout(() => {
        elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }, 100);
}

function displayStats(detections) {
    const uniqueFruits = new Set(detections.map(d => d.class_name)).size;
    const maxConfidence = detections.length > 0
        ? Math.round(Math.max(...detections.map(d => d.confidence)) * 100)
        : 0;

    elements.stats.innerHTML = `
        <div class="stat-card">
            <span class="stat-value">${detections.length}</span>
            <span class="stat-label">Виявлено об'єктів</span>
        </div>
        <div class="stat-card">
            <span class="stat-value">${uniqueFruits}</span>
            <span class="stat-label">Унікальних фруктів</span>
        </div>
        <div class="stat-card">
            <span class="stat-value">${maxConfidence}%</span>
            <span class="stat-label">Макс. впевненість</span>
        </div>
    `;
}

function displayImages(detections) {
    const reader = new FileReader();

    reader.onload = (e) => {
        elements.originalImage.src = e.target.result;

        elements.originalImage.onload = () => {
            drawDetections(elements.originalImage, detections);
        };
    };

    reader.readAsDataURL(selectedFile);
}

function drawDetections(img, detections) {
    const canvas = elements.resultCanvas;
    const ctx = canvas.getContext('2d');

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    ctx.drawImage(img, 0, 0);

    detections.forEach(det => {
        const bbox = det.bbox;
        const color = FRUIT_COLORS[det.class_name] || '#00ffff';

        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.strokeRect(bbox.x1, bbox.y1, bbox.width, bbox.height);

        ctx.fillStyle = color;
        ctx.fillRect(bbox.x1, bbox.y1 - 30, bbox.width, 30);

        ctx.fillStyle = '#0a0a0f';
        ctx.font = 'bold 18px Arial';
        const label = `${det.class_name} ${Math.round(det.confidence * 100)}%`;
        ctx.fillText(label, bbox.x1 + 5, bbox.y1 - 8);
    });
}

function displayDetectionsList(detections) {
    if (detections.length === 0) {
        elements.detectionsList.innerHTML = `
            <div class="no-detections">
                Фрукти не виявлено. Спробуйте знизити поріг впевненості.
            </div>
        `;
        return;
    }

    const sortedDetections = [...detections].sort((a, b) => b.confidence - a.confidence);

    const detectionsHTML = sortedDetections
        .map(det => `
            <div class="detection-item">
                <span class="detection-name">${det.class_name}</span>
                <span class="detection-confidence">${Math.round(det.confidence * 100)}%</span>
            </div>
        `)
        .join('');

    elements.detectionsList.innerHTML = detectionsHTML;
}

function setLoadingState(loading) {
    elements.detectBtn.disabled = loading;
    elements.detectBtn.textContent = loading ? 'Обробка...' : 'Розпізнати фрукти';
}

function hideResults() {
    elements.resultsSection.style.display = 'none';
}

function showError(message) {
    elements.errorMsg.textContent = message;
    elements.errorMsg.style.display = 'block';
}

function hideError() {
    elements.errorMsg.textContent = '';
    elements.errorMsg.style.display = 'none';
}

async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_URL}/`);
        const data = await response.json();

        if (!data.model_loaded) {
            showError('Модель не завантажена на сервері');
        }
    } catch (error) {
        showError('Неможливо підключитися до API. Запустіть сервер: python3 api/main.py');
        console.error('API status check failed:', error);
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}