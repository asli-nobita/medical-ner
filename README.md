# Medical NER Project

A comprehensive pipeline for extracting structured information from medical documents using OCR, rule-based extraction, human-in-the-loop validation, and machine learning.

## 🏥 Project Overview

This project processes medical documents (PDFs, images) to extract structured information such as:
- Patient demographics (name, age, ID, gender)
- Test results with values and units
- Doctor information
- Dates and contact details

## 🚀 Features

- **Image Preprocessing**: PDF conversion, deskewing, denoising, and sharpening
- **OCR Extraction**: Tesseract-based text extraction with confidence scoring
- **Rule-based Extraction**: Pattern matching for common medical fields
- **Human-in-the-Loop**: Interactive validation and correction interface
- **Machine Learning**: Sklearn-based token classification model
- **Modular Architecture**: Clean, maintainable code structure

## 📁 Project Structure

```
medical-ner-project/
├── config/
│   └── settings.py              # Configuration settings
├── src/
│   ├── data_processing/
│   │   ├── image_preprocessor.py    # Image preprocessing functions
│   │   └── ocr_extractor.py         # OCR text extraction
│   ├── extraction/
│   │   ├── rule_based_extractor.py  # Rule-based information extraction
│   │   └── text_processor.py        # Text processing utilities
│   ├── interface/
│   │   └── hitl_interface.py        # Human-in-the-loop interface
│   ├── ml_models/
│   │   ├── model_trainer.py         # Model training orchestration
│   │   └── sklearn_classifier.py    # Sklearn-based classifier
│   └── utils/
│       └── helpers.py               # Utility functions
├── data/
│   ├── raw/                         # Raw input documents
│   ├── processed/                   # Processed data
│   └── sample/                      # Sample data
├── outputs/
│   ├── cleaned_images/              # Preprocessed images
│   ├── ocr_results/                 # OCR output files
│   ├── extraction_results/          # Extracted structured data
│   └── hitl_results/                # Human corrections and confirmations
├── models/
│   └── saved_models/                # Trained ML models
├── notebooks/
│   └── original_assignment.ipynb    # Original Jupyter notebook
├── requirements.txt                 # Project dependencies
└── main.py                          # Main execution script
```

## 🛠️ Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**:
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

4. **Set up Kaggle API** (optional, for dataset download):
   - Install: `pip install kaggle`
   - Configure: Place `kaggle.json` in `~/.kaggle/`

## 🚀 Usage

### Quick Start
```bash
python main.py
```

### Step-by-Step Usage

#### 1. Image Preprocessing
```python
from src.data_processing.image_preprocessor import preprocess_images, save_images

# Process a document
processed_images = preprocess_images('path/to/document.pdf')
save_images(processed_images, 'outputs/cleaned_images')
```

#### 2. OCR Text Extraction
```python
from src.data_processing.ocr_extractor import extract_text, analyse_ocr_quality

# Extract text from preprocessed images
ocr_results = extract_text(processed_images, 'outputs/ocr_results')

# Analyze OCR quality
quality_stats = analyse_ocr_quality(ocr_results)
```

#### 3. Rule-based Information Extraction
```python
from src.extraction.rule_based_extractor import rule_based_extraction, save_extraction_results

# Extract structured information
extraction_results = rule_based_extraction(ocr_results)
save_extraction_results(extraction_results, 'outputs/extraction_results')
```

#### 4. Human-in-the-Loop Validation (Jupyter Required)
```python
from src.interface.hitl_interface import start_working_hitl_review

# Start interactive review (in Jupyter notebook)
hitl = start_working_hitl_review(extraction_results, ocr_results)
```

#### 5. Train ML Model
```python
from src.ml_models.model_trainer import train_sklearn_model

# Train model using corrected data
model_path = train_sklearn_model()
```

## 📊 Output Files

### OCR Results
- `tokens_page_*.json`: Token-level OCR results with positions and confidence
- `tokens_page_*.csv`: CSV format for easy viewing
- `all_tokens.json`: Combined results from all pages

### Extraction Results
- `extracted_page_*.json`: Structured data extracted from each page
- `all_extractions.json`: Combined extraction results

### HITL Results
- `corrections/correction_page_*.json`: Human corrections and confirmations
- `confirmed/`: Validated extraction results

### Model Files
- `models/sklearn_model_*.pkl`: Trained sklearn models
- `models/sklearn_metadata_*.json`: Model metadata and performance metrics

## 🔧 Configuration

Edit `config/settings.py` to customize:
- File paths and directory structure
- OCR settings (confidence thresholds, Tesseract config)
- Extraction patterns and rules
- ML model parameters

## 📋 Dependencies

- **Core**: `opencv-python`, `pillow`, `numpy`, `pandas`
- **OCR**: `pytesseract`, `pdf2image`
- **ML**: `scikit-learn`
- **Interface**: `ipywidgets` (for Jupyter notebooks)
- **Data**: `kagglehub` (optional, for dataset download)

## 🚦 Pipeline Flow

1. **Input**: Medical documents (PDF, PNG, JPG)
2. **Preprocessing**: Convert to grayscale, deskew, denoise, sharpen
3. **OCR**: Extract text with positions and confidence scores
4. **Rule-based Extraction**: Apply regex patterns to identify fields
5. **Human Validation**: Review and correct extractions (optional)
6. **Model Training**: Train ML model on corrected data (optional)
7. **Output**: Structured JSON data ready for downstream use

## 📈 Performance Tips

- Use high-resolution images (300+ DPI) for better OCR accuracy
- Ensure documents are properly oriented and clear
- Review low-confidence OCR results manually
- Provide human corrections to improve ML model training
- Adjust extraction patterns for your specific document types

## 🤝 Contributing

1. Follow the modular structure when adding new features
2. Update configuration in `settings.py` for new parameters
3. Add comprehensive docstrings to functions
4. Test with various document types and formats

## 📄 License

This project is provided as-is for educational and research purposes.

## 🔍 Troubleshooting

- **Tesseract not found**: Ensure Tesseract OCR is installed and in PATH
- **Low OCR accuracy**: Check image quality, try different preprocessing parameters
- **Import errors**: Verify all dependencies are installed and project structure is correct
- **Jupyter widgets not working**: Install and enable `ipywidgets` extension

For more detailed usage examples, see the original Jupyter notebook in `notebooks/original_assignment.ipynb`.
