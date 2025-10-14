"""OCR text extraction module for medical document processing."""

import os
import json
import pandas as pd
import pytesseract


def extract_text(images, output_dir='ocr_results'): 
    """
    Extracts text with positions and confidence from preprocessed images using Tesseract OCR.
    
    Args:
        images (list): List of PIL Image objects
        output_dir (str): Directory to save OCR results
        
    Returns:
        list: List of dictionaries containing OCR data for each page
    """ 
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir) 
    all_pages_data = [] 
    
    for page_no, img in enumerate(images,1): 
        print(f'Processing page no {page_no}') 
        
        config = r'--oem 3 --psm 6' 
        ocr_data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT) 
        page_tokens = [] 
        n_boxes = len(ocr_data['text']) 
        
        for i in range(n_boxes): 
            text = ocr_data['text'][i].strip() 
            confidence = float(ocr_data['conf'][i]) 
            
            if text and confidence>0: 
                token_data = { 
                    'text': text, 
                    'left': int(ocr_data['left'][i]), 
                    'top': int(ocr_data['top'][i]), 
                    'width': int(ocr_data['width'][i]), 
                    'height': int(ocr_data['height'][i]),
                    'confidence': round(confidence, 2),
                    'level': int(ocr_data['level'][i]), 
                    'page_num': int(ocr_data['page_num'][i]),
                    'block_num': int(ocr_data['block_num'][i]),
                    'par_num': int(ocr_data['par_num'][i]),
                    'line_num': int(ocr_data['line_num'][i]),
                    'word_num': int(ocr_data['word_num'][i])             
                } 
                page_tokens.append(token_data) 
        
        json_filename = os.path.join(output_dir, f'tokens_page_{page_no}.json') 
        with open(json_filename, 'w', encoding='utf-8') as f: 
            json.dump({
                'page': page_no,
                'total_tokens': len(page_tokens), 
                'tokens': page_tokens
            }, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(page_tokens)} tokens to {json_filename}") 
        
        csv_filename = os.path.join(output_dir, f'tokens_page_{page_no}.csv')
        save_tokens_as_csv(page_tokens, csv_filename)
        
        all_pages_data.append({
            'page': page_no,
            'tokens': page_tokens
        })
    
    combined_json = os.path.join(output_dir, 'all_tokens.json')
    with open(combined_json, 'w', encoding='utf-8') as f: 
        json.dump({
            'total_pages': len(images),
            'pages': all_pages_data           
        }, f, indent=2, ensure_ascii=False) 
    print(f"Combined results saved to {combined_json}") 
    return all_pages_data


def save_tokens_as_csv(tokens, csv_filename): 
    """
    Saves tokens data as CSV file.
    
    Args:
        tokens (list): List of token dictionaries
        csv_filename (str): Output CSV file path
    """
    
    if tokens:
        df = pd.DataFrame(tokens)
        df.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"CSV saved: {csv_filename}")
        

def analyse_ocr_quality(ocr_data): 
    """
    Analyzes OCR quality and provides statistics.
    
    Args:
        ocr_data (list): OCR data from extract_text_with_positions
        
    Returns:
        dict: Quality statistics
    """ 
    all_tokens = [] 
    for page in ocr_data: 
        all_tokens.extend(page['tokens']) 
    if not all_tokens: 
        return {'error': 'No tokens found.'} 
    
    confidences = [token['confidence'] for token in all_tokens] 
    stats = { 
        'total_tokens': len(all_tokens), 
        'avg_confidence': round(sum(confidences)/len(confidences)), 
        'min_confidence': min(confidences), 
        'max_confidences': max(confidences), 
        'low_confidence_count': len([c for c in confidences if c<50]), 
        'high_confidence_count': len([c for c in confidences if c>=80]), 
        'tokens_needing_review': len([c for c in confidences if c<70])         
    } 
    return stats 


def filter_high_confidence_tokens(ocr_data, min_confidence=70): 
    """ 
    Args:
        ocr_data (list): OCR data from extract_text_with_positions
        min_confidence (float): Minimum confidence threshold
        
    Returns:
        list: Filtered OCR data with high confidence tokens only
    """
    filtered_data = []
    
    for page in ocr_data:
        high_conf_tokens = [
            token for token in page['tokens'] 
            if token['confidence'] >= min_confidence
        ]
        
        filtered_data.append({
            'page': page['page'],
            'tokens': high_conf_tokens
        })
    
    return filtered_data