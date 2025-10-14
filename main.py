from src.data_processing.image_preprocessor import preprocess_directory  
from src.data_processing.ocr_extractor import extract_text
from src.extraction.rule_based_extractor import rule_based_extraction, save_extraction_results, print_extraction_summary
from src.extraction.hybrid_extractor import hybrid_extraction
from src.interface.hitl_interface import start_working_hitl_review
import os
import json
import time

def _prompt_yes_no(prompt, default=True):
    try:
        resp = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    if resp == '':
        return default
    return resp in ('y', 'yes')

def _list_files_with_ext(directory, exts):
    files = []
    for ext in exts:
        files.extend([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(ext)])
    print(f"Found {len(files)} files in {directory}.")
    return sorted(files)

def check_corrections_and_train(corrections_dir='outputs/hitl/corrections'):
    """
    Check for correction files and offer to train ML model.
    
    Args:
        corrections_dir (str): Directory containing correction files
        
    Returns:
        bool: True if training was attempted, False otherwise
    """
    if not os.path.exists(corrections_dir):
        print(f"📂 No corrections directory found at {corrections_dir}")
        return False
    
    correction_files = [f for f in os.listdir(corrections_dir) if f.endswith('.json')]
    
    if len(correction_files) < 2:
        if len(correction_files) == 1:
            print(f"📝 Found 1 correction file. Need at least 2 corrections for training.")
        else:
            print(f"📝 No correction files found. Make corrections through HITL interface first.")
        return False
    
    print(f"📊 Found {len(correction_files)} correction files in {corrections_dir}")
    
    
    train_model = _prompt_yes_no(
        f"Train ML model using {len(correction_files)} corrections?", 
        default=True
    )
    
    if not train_model:
        print("⏭️ Skipping ML model training.")
        return False
    
    print("🚀 Starting ML model training...")
    
    try:
        from src.ml_models.model_trainer import train_sklearn_model
        
        print("📚 Processing correction data...")
        model_path = train_sklearn_model()
        
        if model_path:
            print(f"✅ Model training completed successfully!")
            print(f"💾 Model saved to: {model_path}")
            print(f"🎯 Future pipeline runs will automatically use this trained model for better accuracy.")
            return True
        else:
            print("❌ Model training failed. Check the logs above for details.")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure sklearn and other ML dependencies are installed.")
        return False
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("💡 Check your correction files format and try again.")
        return False

def process_directory_and_run_ocr(
    input_dir, cleaned_dir='cleaned_images', ocr_dir='ocr_results', extractions_dir='extraction_results',
    dpi=300, base_filename='cleaned_image',
):
    """
    Process all files in `input_dir`: preprocess images, run OCR, save token JSON/CSV,
    run rule-based extraction. Prompts user to run/skip each step if boolean args are None.
    """
    
    run_preprocess=None 
    run_ocr=None 
    run_extraction=None

    processed_images = []
    ocr_results = []
    extraction_results = []

    
    if run_preprocess is None:
        run_preprocess = _prompt_yes_no("Run image preprocessing step?", default=True)
    if run_preprocess:
        print(f'1) Preprocessing directory: {input_dir} -> cleaned images saved to {cleaned_dir}')
        processed_images = preprocess_directory(input_dir, output_dir=cleaned_dir, dpi=dpi, base_filename=base_filename)
        if not processed_images:
            print('❌ No processed images produced. Aborting pipeline.')
            return None, None
    else:
        
        print('1) Skipping preprocessing. Looking for existing cleaned images...')
        processed_images = _list_files_with_ext(cleaned_dir, ['.png', '.jpg', '.jpeg'])
        if not processed_images:
            print(f'❌ No cleaned images found in {cleaned_dir}. Either enable preprocessing or add images.')
            return None, None

    
    if run_ocr is None:
        run_ocr = _prompt_yes_no("Run OCR extraction step?", default=True)
    if run_ocr:
        print(f'2) Running OCR on {len(processed_images)} processed images -> saving to {ocr_dir}')
        ocr_results = extract_text(processed_images, output_dir=ocr_dir)
        if not ocr_results:
            print('❌ OCR produced no results. Aborting pipeline.')
            return None, None
    else:
        
        print('2) Skipping OCR. Loading existing OCR result files...')
        all_tokens_path = os.path.join(ocr_dir, 'all_tokens.json')
        
        if os.path.exists(all_tokens_path):
            try:
                with open(all_tokens_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, dict) and 'pages' in data:
                    total_pages = data.get('total_pages', len(data['pages']))
                    ocr_results = data['pages'] 

                    print(f'📄 Loaded all_tokens.json: {total_pages} pages')
                else:
                    print(f'⚠️ Unexpected format in all_tokens.json: {type(data)}')
                    
            except Exception as e:
                print(f'⚠️ Failed to load {all_tokens_path}: {e}')
        else:
            print(f'⚠️ all_tokens.json not found at {all_tokens_path}')
        
        if not ocr_results:
            print(f'❌ No OCR JSON results found in {ocr_dir}. Either run OCR or provide OCR outputs.')
            return None, None

    
    if run_extraction is None:
        run_extraction = _prompt_yes_no("Run rule-based extraction step?", default=True)
    if run_extraction:
        print('3) Running hybrid extraction (ML + rules) on OCR results')
        
        
        use_hybrid = _prompt_yes_no("Use hybrid extraction (ML + rules) if trained model exists?", default=True)
        
        if use_hybrid:
            extraction_results = hybrid_extraction(ocr_results)
        else:
            extraction_results = rule_based_extraction(ocr_results)
            
        save_extraction_results(extraction_results, output_dir=extractions_dir)
        
    else:
        print('3) Skipping rule-based extraction. Attempting to load existing extraction results...')
        all_extractions_path = os.path.join(extractions_dir, 'all_extractions.json')
        if os.path.exists(all_extractions_path): 
            try: 
                with open(all_extractions_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and 'pages' in data:
                    total_pages = data.get('total_pages', len(data['pages']))
                    extraction_results = data['pages']
                    print(f'📄 Loaded all_extractions.json: {total_pages} pages')
                else:
                    print(f'⚠️ Unexpected format in all_extractions.json: {type(data)}')
            except Exception as e:
                print(f'⚠️ Failed to load {all_extractions_path}: {e}')
        else: 
            print(f'⚠️ all_extractions.json not found at {all_extractions_path}')
            
        if not extraction_results:
            print('⚠️ No extraction results available. HITL UI will have no extractions to review.')

    return extraction_results, ocr_results

DATASET_PATH = 'data/raw/'
CLEANED_DIR = 'outputs/cleaned_images/' 
OCR_DIR = 'outputs/ocr_results/' 
EXTRACTION_DIR = 'outputs/extraction_results/'
CORRECTIONS_DIR = 'outputs/hitl/corrections/'

def main(): 
    """Main function to run the full pipeline."""
    print("🚀 Starting the Document Processing Pipeline")
    
    extraction_results, ocr_results = process_directory_and_run_ocr(
        input_dir=DATASET_PATH,
        cleaned_dir=CLEANED_DIR,
        ocr_dir=OCR_DIR, 
        extractions_dir=EXTRACTION_DIR,
        dpi=300,
        base_filename='cleaned_image'
    )
    
    if extraction_results is None or ocr_results is None:
        print("❌ Pipeline terminated due to errors in processing.")
        return
    
    print("\nPipeline completed successfully!")
    print("Starting Human-in-the-Loop (HITL) UI...")
    
    
    hitl = start_working_hitl_review(extraction_results, ocr_results)
    
    if hitl and 'url' in hitl:
        print(f"\n🌐 HITL interface available at: {hitl['url']}")
        print("📝 Review and correct extractions in the web interface.")
        print("💡 Use the interface to validate and fix any incorrect extractions.")
        
        try:
            print("\n⏳ Make your corrections in the web interface, then return here.")
            input("Press Enter when you're done making corrections...")
            
            print("\n🔍 Checking for corrections...")
            time.sleep(1)  

            training_attempted = check_corrections_and_train(corrections_dir=CORRECTIONS_DIR)

            if training_attempted:
                print("\n🎉 Training completed! Your corrections will improve future extractions.")
            else:
                print("\n💡 You can train a model later by running:")
                print("   python -c \"from src.ml_models.model_trainer import train_sklearn_model; train_sklearn_model('outputs/hitl/corrections')\"")
        
        except KeyboardInterrupt:
            print("\n\n👋 Pipeline interrupted by user. Exiting...")
        except EOFError:
            print("\n\n👋 Input stream closed. Exiting...")
        finally:
            
            if 'thread' in hitl and hitl['thread'].is_alive():
                print("🛑 Stopping HITL interface...")
    else:
        print("⚠️ HITL interface failed to start. Check the logs above.")

if __name__ == "__main__": 
    main()