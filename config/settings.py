"""Configuration settings for medical NER project."""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Data directories
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, 'sample')

# Output directories
CLEANED_IMAGES_DIR = os.path.join(OUTPUTS_DIR, 'cleaned_images')
OCR_RESULTS_DIR = os.path.join(OUTPUTS_DIR, 'ocr_results')
EXTRACTION_RESULTS_DIR = os.path.join(OUTPUTS_DIR, 'extraction_results')
HITL_RESULTS_DIR = os.path.join(OUTPUTS_DIR, 'hitl_results')
HITL_CORRECTIONS_DIR = os.path.join(HITL_RESULTS_DIR, 'corrections')
HITL_CONFIRMED_DIR = os.path.join(HITL_RESULTS_DIR, 'confirmed')

# Model directories
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, 'saved_models')

# Image preprocessing settings
DEFAULT_DPI = 300
LINE_TOLERANCE = 10

# OCR settings
TESSERACT_CONFIG = r'--oem 3 --psm 6'
MIN_CONFIDENCE_THRESHOLD = 70
HIGH_CONFIDENCE_THRESHOLD = 80

# Rule-based extraction settings
EXTRACTION_PATTERNS = {
    'name': [
        r'(?:name|patient\s*name)[:\s]+(.+?)(?:\s|$)',
        r'(?:mr|mrs|ms|dr)\.?\s+([a-zA-Z\s]+)',
    ],
    'age': [
        r'(?:age)[:\s]+(\d+)',
        r'(\d+)\s*(?:years?|yrs?|y\.o\.)',
    ],
    'patient_id': [
        r'(?:patient\s*id|id\s*no|registration\s*no)[:\s]+(\S+)',
        r'(?:id)[:\s]+([A-Z0-9]+)',
    ],
    'date': [
        r'(?:date|report\s*date)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
    ],
    'doctor': [
        r'(?:dr|doctor)[:\s]*\.?\s*([a-zA-Z\s]+)',
        r'(?:referred\s*by|consultant)[:\s]+([a-zA-Z\s\.]+)',
    ], 
    'gender': [
        r'(?:gender|sex)[:\s]+(male|female|m|f)',
    ],
    'phone': [
        r'(?:phone|mobile|contact)[:\s]*(\+?\d{10,15})',
    ]
}

TEST_PATTERNS = [
    r'(\w+(?:\s+\w+)*)\s+([\d.]+)\s*(\w+/?[\w\s]*)',  # TestName Value Unit
    r'(\w+(?:\s+\w+)*)[:\s]+([\d.]+)\s*(\w+/?[\w\s]*)',  # TestName: Value Unit
]

# ML Model settings
ENTITY_TYPES = [
    'O', 'NAME', 'AGE', 'ID', 'DATE', 'GENDER', 
    'TEST_NAME', 'TEST_VALUE', 'TEST_UNIT', 'DOCTOR', 'PHONE'
]

SKLEARN_CONFIG = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'max_iter': 1000,
    'random_state': 42
}

# File naming patterns
CLEANED_IMAGE_PREFIX = 'cleaned_image'
TOKEN_FILE_PREFIX = 'tokens_page_'
EXTRACTION_FILE_PREFIX = 'extracted_page_'
MODEL_FILE_PREFIX = 'sklearn_model_'
METADATA_FILE_PREFIX = 'sklearn_metadata_'

# Create directories if they don't exist
def create_directories():
    """Create all necessary directories."""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR,
        OUTPUTS_DIR, CLEANED_IMAGES_DIR, OCR_RESULTS_DIR, 
        EXTRACTION_RESULTS_DIR, HITL_RESULTS_DIR, 
        HITL_CORRECTIONS_DIR, HITL_CONFIRMED_DIR,
        MODELS_DIR, SAVED_MODELS_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("📁 All necessary directories created.")