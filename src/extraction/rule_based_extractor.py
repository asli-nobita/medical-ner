"""Rule-based extraction module for medical documents."""

import os
import json
import re
from .text_processor import group_tokens_into_lines, extract_line_text


def rule_based_extraction(ocr_data): 
    """
    Performs rule-based extraction on OCR data.
    
    Args:
        ocr_data (list): OCR data from extract_text function
        
    Returns:
        dict: Extracted structured data
    """ 
    all_extracted_data = [] 
    
    for page_data in ocr_data: 
        page_no = page_data['page'] 
        tokens = page_data['tokens'] 
        
        print(f'Processing page {page_no} for rule-based extraction...') 

        lines = group_tokens_into_lines(tokens) 
        line_texts = [extract_line_text(line) for line in lines] 
        
        extracted_fields = {
            'page': page_no, 
            'patient_info': {}, 
            'test_results': [], 
            'other_fields': {}, 
            'confidence_scores': {}
        }
        
        # Detect sections first to apply targeted extraction
        sections = detect_report_sections(line_texts)
        
        # Apply field extraction with section awareness
        extracted_fields = apply_field_extraction_rules(line_texts, lines, extracted_fields, sections) 
        extracted_fields = extract_test_tables(lines, extracted_fields, sections) 
        all_extracted_data.append(extracted_fields) 
        
    return all_extracted_data


def detect_report_sections(line_texts):
    """
    Detects different sections of a medical report.
    
    Args:
        line_texts (list): List of line text strings
        
    Returns:
        dict: Section boundaries and types
    """
    sections = {
        'header': [],
        'patient_info': [],
        'test_results': [],
        'footer': []
    }
    
    current_section = 'header'
    
    for i, line_text in enumerate(line_texts):
        line_lower = line_text.lower().strip()
        
        # Header indicators
        if any(keyword in line_lower for keyword in ['laboratory', 'reference', 'neuberg', 'report']):
            current_section = 'header'
            
        # Patient info section indicators
        elif any(keyword in line_lower for keyword in ['name:', 'age:', 'gender:', 'patient', 'mob. no', 'pt.id']):
            current_section = 'patient_info'
            
        # Test results section indicators
        elif any(keyword in line_lower for keyword in ['test results', 'abnormal result', 'test name', 'result value', 'unit']):
            current_section = 'test_results'
            
        # Footer indicators
        elif any(keyword in line_lower for keyword in ['page', 'printed on', 'bangalore', 'chennai', 'www.', '@']):
            current_section = 'footer'
            
        # Add line to current section
        sections[current_section].append(i)
    
    return sections 


def apply_field_extraction_rules(line_texts, lines, extracted_fields, sections=None):
    """
    Applies regex rules to extract common fields.
    
    Args:
        line_texts (list): List of line text strings
        lines (list): List of line token groups
        extracted_fields (dict): Current extraction results
        sections (dict): Section boundaries (optional)
        
    Returns:
        dict: Updated extraction results
    """
    # Define regex patterns for common fields specific to medical lab reports
    patterns = {
        'name': [
            r'name\s*:\s*(.+?)(?:\s+gender|\s+age|\s*$)',
            r'(?:mr\.?|mrs\.?|ms\.?|dr\.?)\s*([a-zA-Z][a-zA-Z\s\.]+?)(?:\s+gender|\s+age|\s*$)',
            r'patient\s*name\s*:\s*(.+?)(?:\s|$)',
        ],
        'age': [
            r'age\s*[:\s]*(\d+)\s*(?:years?|yrs?|y\.?o\.?)',
            r'(\d+)\s*years?\s*(?:mob\.|mobile|gender|phone)',
        ],
        'patient_id': [
            r'(?:pt\.?\s*id|patient\s*id|pid)\s*:\s*([A-Z0-9]+)',
            r'id\s*(?:no\.?)?\s*:\s*([A-Z0-9]{6,})',
        ],
        'lab_id': [
            r'(?:lab\s*id|labid)\s*[:\s]*([A-Z0-9]{8,})',
        ],
        'date': [
            r'(?:reg\.?\s*date|report\s*date|date)\s*(?:and\s*time)?\s*:\s*(\d{1,2}[-/]\w{3}[-/]\d{4})',
            r'(\d{1,2}[-/]\w{3}[-/]\d{4})\s*\d{2}:\d{2}',
        ],
        'doctor': [
            r'(?:ref\.?\s*by|referred\s*by|consultant)\s*:\s*([a-zA-Z][a-zA-Z\s\.]+?)(?:\s+pi\.|$)',
            r'dr\.?\s*([a-zA-Z][a-zA-Z\s\.]+?)(?:\s|$)',
        ], 
        'gender': [
            r'gender\s*[:\s]+(male|female|m|f)(?:\s+lab|\s|$)',
        ],
        'phone': [
            r'(?:mob\.?\s*no\.?|mobile|phone)\s*:\s*(\d{10,15})',
        ]
    } 
    
    # Additional validation rules to avoid false positives
    validation_rules = {
        'name': lambda x: len(x.strip()) > 2 and not re.match(r'^\d+$', x.strip()) and not x.lower() in ['male', 'female', 'result', 'test'],
        'age': lambda x: x.isdigit() and 0 < int(x) < 150,
        'patient_id': lambda x: len(x) >= 4 and re.match(r'^[A-Z0-9]+$', x),
        'lab_id': lambda x: len(x) >= 8 and re.match(r'^[A-Z0-9]+$', x),
        'gender': lambda x: x.lower() in ['male', 'female', 'm', 'f'],
        'phone': lambda x: x.isdigit() and len(x) >= 10,
    }
    
    # Focus on patient_info section if sections are detected
    lines_to_process = range(len(line_texts))
    if sections and sections.get('patient_info'):
        lines_to_process = sections['patient_info']
    
    for i in lines_to_process: 
        if i >= len(line_texts):
            continue
            
        line_text = line_texts[i]
        line_text_lower = line_text.lower().strip()
        
        # Skip lines that are clearly not patient info (too short, just numbers, etc.)
        if len(line_text_lower) < 3 or re.match(r'^\d+\s*$', line_text_lower):
            continue
            
        for field_name, field_patterns in patterns.items(): 
            for pattern in field_patterns: 
                match = re.search(pattern, line_text, re.IGNORECASE) 
                if match: 
                    value = match.group(1).strip()
                    
                    # Apply validation rules
                    if field_name in validation_rules:
                        if not validation_rules[field_name](value):
                            continue
                    
                    line_tokens = lines[i] 
                    avg_confidence = sum(token['confidence'] for token in line_tokens) / len(line_tokens)
                    
                    # Store extracted field 
                    if field_name in ['name', 'age', 'patient_id', 'lab_id', 'gender']: 
                        extracted_fields['patient_info'][field_name] = value 
                    else: 
                        extracted_fields['other_fields'][field_name] = value
                    
                    extracted_fields['confidence_scores'][field_name] = round(avg_confidence, 2)
                    break 
    return extracted_fields 


def extract_test_tables(lines, extracted_fields, sections=None): 
    """
    Extracts test result tables using positional heuristics.
    
    Args:
        lines (list): List of line token groups
        extracted_fields (dict): Current extraction results
        sections (dict): Section boundaries (optional)
        
    Returns:
        dict: Updated extraction results with test results
    """
    # Improved patterns for medical test results
    test_patterns = [
        # Test Name followed by numeric value and unit, possibly with range
        r'^([A-Za-z][A-Za-z\s\(\)-]+?)\s+([\d.,]+)\s+([a-zA-Z/]+)\s*(?:[\d.,\s<>-]+)?$',
        # Test Name with Value Unit (more structured)
        r'^([A-Za-z][A-Za-z\s\(\)-]{3,}?)\s+([\d.,]+)\s+([a-zA-Z/μ]+(?:/[a-zA-Z]+)?)\s*',
        # Catch some variations
        r'^([A-Za-z][A-Za-z\s\(\)-]{2,}?)\s*[:\s]+([\d.,]+)\s+([a-zA-Z/μ]+)',
    ]
    
    # Known medical test name patterns (to validate)
    medical_test_indicators = [
        'calcium', 'chloride', 'sodium', 'potassium', 'glucose', 'cholesterol', 
        'ldl', 'hdl', 'triglycerides', 'hemoglobin', 'hematocrit', 'platelets',
        'creatinine', 'bun', 'urea', 'albumin', 'protein', 'bilirubin',
        'vitamin', 'b12', 'folate', 'iron', 'ferritin', 'tsh', 'testosterone',
        'homocysteine', 'profile', 'function', 'test'
    ]
    
    # Common units in medical tests
    valid_units = [
        'mg/dl', 'mg/l', 'mmol/l', 'μmol/l', 'mcmol/l', 'meq/l', 'iu/l', 'u/l',
        'g/dl', 'g/l', 'ng/ml', 'pg/ml', 'pmol/l', 'nmol/l', '%', 'ratio'
    ]
    
    # Exclude patterns that are clearly not test results
    exclusion_patterns = [
        r'^page\s+\d+',
        r'^\d{1,2}[-/]\w{3}[-/]\d{4}',  # Dates
        r'^printed\s+on',
        r'^note:',
        r'^laboratory',
        r'^neuberg',
        r'^bangalore',
        r'^chennai',
        r'^\d+\s*$',  # Just numbers
        r'^www\.',
        r'@',
        r'reference\s+range',
        r'test\s+name',
        r'result\s+value',
        r'abnormal\s+result',
        r'summary',
        r'more\s+than\s+\d+',  # Reference range text
        r'very\s+high',
        r'borderline',
        r'optimal',
    ]
    
    # Focus on test_results section if sections are detected
    lines_to_process = range(len(lines))
    if sections and sections.get('test_results'):
        lines_to_process = sections['test_results']
    
    for i in lines_to_process: 
        if i >= len(lines):
            continue
            
        line = lines[i]
        line_text = extract_line_text(line).strip()
        line_text_lower = line_text.lower()
        
        # Skip empty lines or very short lines
        if len(line_text) < 5:
            continue
            
        # Skip header lines or navigation elements
        if any(re.search(pattern, line_text_lower) for pattern in exclusion_patterns):
            continue
            
        # Skip lines that are likely headers or section dividers
        if re.search(r'test|result|parameter|value|unit|range|name|biological|ref|remarks', line_text_lower) and len(line_text) < 50:
            continue
            
        # Try to match test result patterns
        for pattern in test_patterns:
            match = re.search(pattern, line_text, re.IGNORECASE)
            if match:
                test_name = match.group(1).strip()
                test_value = match.group(2).strip()
                test_unit = match.group(3).strip() if len(match.groups()) > 2 else ""
                
                # Validate test name (should contain medical terms and be reasonable length)
                test_name_lower = test_name.lower()
                
                # Must be a reasonable length
                if len(test_name) < 3 or len(test_name) > 50:
                    continue
                    
                # Should not start with numbers or special characters
                if not re.match(r'^[A-Za-z]', test_name):
                    continue
                    
                # Should contain at least one medical indicator or be structured properly
                has_medical_term = any(indicator in test_name_lower for indicator in medical_test_indicators)
                has_reasonable_structure = re.match(r'^[A-Za-z][A-Za-z\s\(\)-]+$', test_name)
                
                if not (has_medical_term or has_reasonable_structure):
                    continue
                
                # Validate test value (should be numeric)
                if not re.match(r'^[\d.,<>]+$', test_value):
                    continue
                    
                # Validate unit if present
                if test_unit and test_unit.lower() not in valid_units and not re.match(r'^[a-zA-Z/μ]+$', test_unit):
                    continue
                
                # Calculate confidence
                avg_confidence = sum(token['confidence'] for token in line) / len(line)
                
                # Additional confidence check
                if avg_confidence < 60:
                    continue
                    
                test_result = {
                    'test_name': test_name,
                    'value': test_value,
                    'unit': test_unit,
                    'confidence': round(avg_confidence, 2),
                    'line_number': i + 1
                }
                
                extracted_fields['test_results'].append(test_result)
                break
    
    return extracted_fields


def save_extraction_results(extraction_results, output_dir='extraction_results'):
    """
    Saves extraction results to JSON files.
    
    Args:
        extraction_results (list): List of extracted data per page
        output_dir (str): Output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save individual page results
    for page_data in extraction_results:
        page_num = page_data['page']
        filename = os.path.join(output_dir, f'extracted_page_{page_num:02d}.json')
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved extraction results to {filename}")
    
    # Save combined results
    combined_filename = os.path.join(output_dir, 'all_extractions.json')
    with open(combined_filename, 'w', encoding='utf-8') as f:
        json.dump({
            'total_pages': len(extraction_results),
            'pages': extraction_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Combined extraction results saved to {combined_filename}")


def print_extraction_summary(extraction_results):
    """
    Prints a summary of extraction results.
    
    Args:
        extraction_results (list): List of extracted data per page
    """
    print("\n=== EXTRACTION SUMMARY ===")
    
    total_fields = 0
    total_tests = 0
    
    for page_data in extraction_results:
        page_num = page_data['page']
        patient_fields = len(page_data['patient_info'])
        other_fields = len(page_data['other_fields'])
        test_count = len(page_data['test_results'])
        
        total_fields += patient_fields + other_fields
        total_tests += test_count
        
        print(f"\nPage {page_num}:")
        print(f"  Patient fields: {patient_fields}")
        print(f"  Other fields: {other_fields}")
        print(f"  Test results: {test_count}")
        
        # Show extracted patient info
        if page_data['patient_info']:
            print("  Patient Info:")
            for key, value in page_data['patient_info'].items():
                confidence = page_data['confidence_scores'].get(key, 0)
                print(f"    {key}: {value} (confidence: {confidence}%)")
        # Show test results
        if page_data['test_results']:
            print("  Test Results:")
            for test in page_data['test_results'][:3]:  # Show first 3 tests
                print(f"    {test['test_name']}: {test['value']} {test['unit']} (confidence: {test['confidence']}%)")
            if len(page_data['test_results']) > 3:
                print(f"    ... and {len(page_data['test_results']) - 3} more tests")
    
    print(f"\nTotal extracted fields: {total_fields}")
    print(f"Total test results: {total_tests}")