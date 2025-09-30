"""Hybrid extraction combining ML token classification with rule-based validation."""

import os
import json
import pickle
from .rule_based_extractor import apply_field_extraction_rules, extract_test_tables, detect_report_sections
from .text_processor import group_tokens_into_lines, extract_line_text
from ..ml_models.sklearn_classifier import SklearnTokenClassifier


class HybridExtractor:
    """Combines ML token classification with rule-based extraction."""
    
    def __init__(self, model_path=None):
        self.ml_classifier = None
        self.model_loaded = False
        
        if model_path and os.path.exists(model_path):
            try:
                self.ml_classifier = SklearnTokenClassifier()
                self.ml_classifier.load_model(model_path)
                self.model_loaded = True
                print(f"✅ Loaded ML model from {model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load ML model: {e}")
                print("Falling back to rule-based extraction only")
    
    def extract_token_features(self, tokens):
        """Extract features for ML classification."""
        features = []
        
        for i, token in enumerate(tokens):
            token_features = []
            
            # Token features
            token_features.extend([
                token['text'].lower(),
                str(token['text'].isupper()),
                str(token['text'].islower()),
                str(token['text'].isdigit()),
                str(token['text'].isalpha()),
                str(len(token['text'])),
                str(token.get('confidence', 0) / 100.0),
            ])
            
            # Position features
            token_features.extend([
                str(token.get('left', 0)),
                str(token.get('top', 0)),
                str(token.get('width', 0)),
                str(token.get('height', 0)),
                str(i),  # Position in sequence
            ])
            
            # Context features (previous and next tokens)
            if i > 0:
                prev_token = tokens[i - 1]
                token_features.extend([
                    prev_token['text'].lower(),
                    str(prev_token['text'].isdigit()),
                ])
            else:
                token_features.extend(['<START>', 'False'])
            
            if i < len(tokens) - 1:
                next_token = tokens[i + 1]
                token_features.extend([
                    next_token['text'].lower(),
                    str(next_token['text'].isdigit()),
                ])
            else:
                token_features.extend(['<END>', 'False'])
            
            # Convert to string for TfidfVectorizer
            feature_string = ' '.join(token_features)
            features.append(feature_string)
        
        return features
    
    def ml_token_classification(self, tokens):
        """Apply ML model to classify each token."""
        if not self.model_loaded:
            return ['O'] * len(tokens)  # All tokens as 'Other'
        
        try:
            features = self.extract_token_features(tokens)
            predictions = self.ml_classifier.predict(features)
            return predictions
        except Exception as e:
            print(f"⚠️ ML prediction failed: {e}")
            return ['O'] * len(tokens)
    
    def tokens_to_entities(self, tokens, labels):
        """Convert token-level labels to entity-level extractions."""
        entities = {
            'patient_info': {},
            'test_results': [],
            'other_fields': {},
            'confidence_scores': {}
        }
        
        # Group consecutive tokens with same label
        current_entity = None
        current_tokens = []
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label != 'O':
                if current_entity == label:
                    # Continue current entity
                    current_tokens.append(token)
                else:
                    # Save previous entity if exists
                    if current_entity and current_tokens:
                        self._save_entity(entities, current_entity, current_tokens)
                    
                    # Start new entity
                    current_entity = label
                    current_tokens = [token]
            else:
                # Save current entity if exists
                if current_entity and current_tokens:
                    self._save_entity(entities, current_entity, current_tokens)
                current_entity = None
                current_tokens = []
        
        # Save final entity
        if current_entity and current_tokens:
            self._save_entity(entities, current_entity, current_tokens)
        
        return entities
    
    def _save_entity(self, entities, entity_type, tokens):
        """Save an entity based on its type."""
        text_value = ' '.join(token['text'] for token in tokens)
        avg_confidence = sum(token.get('confidence', 0) for token in tokens) / len(tokens)
        
        # Map entity types to fields
        if entity_type == 'NAME':
            entities['patient_info']['name'] = text_value
            entities['confidence_scores']['name'] = avg_confidence
        elif entity_type == 'AGE':
            entities['patient_info']['age'] = text_value
            entities['confidence_scores']['age'] = avg_confidence
        elif entity_type == 'ID':
            entities['patient_info']['patient_id'] = text_value
            entities['confidence_scores']['patient_id'] = avg_confidence
        elif entity_type == 'GENDER':
            entities['patient_info']['gender'] = text_value
            entities['confidence_scores']['gender'] = avg_confidence
        elif entity_type == 'PHONE':
            entities['patient_info']['phone'] = text_value
            entities['confidence_scores']['phone'] = avg_confidence
        elif entity_type in ['TEST_NAME', 'TEST_VALUE', 'TEST_UNIT']:
            # For test results, we need to group them properly
            # This is simplified - in practice you'd need more sophisticated grouping
            pass
        else:
            entities['other_fields'][entity_type.lower()] = text_value
            entities['confidence_scores'][entity_type.lower()] = avg_confidence
    
    def merge_extractions(self, ml_entities, rule_entities):
        """Merge ML and rule-based extractions intelligently."""
        merged = {
            'patient_info': {},
            'test_results': [],
            'other_fields': {},
            'confidence_scores': {}
        }
        
        # For patient info, prefer rule-based if it found something, otherwise use ML
        for field in ['name', 'age', 'patient_id', 'gender', 'phone']:
            rule_value = rule_entities['patient_info'].get(field)
            ml_value = ml_entities['patient_info'].get(field)
            
            if rule_value:
                # Rule-based found something, use it (higher precision)
                merged['patient_info'][field] = rule_value
                merged['confidence_scores'][field] = rule_entities['confidence_scores'].get(field, 0)
            elif ml_value:
                # Rule-based missed it, but ML found it (better recall)
                merged['patient_info'][field] = ml_value
                merged['confidence_scores'][field] = ml_entities['confidence_scores'].get(field, 0)
        
        # For test results, prefer rule-based (more structured approach)
        if rule_entities['test_results']:
            merged['test_results'] = rule_entities['test_results']
        else:
            merged['test_results'] = ml_entities['test_results']
        
        # Merge other fields
        merged['other_fields'].update(rule_entities.get('other_fields', {}))
        merged['other_fields'].update(ml_entities.get('other_fields', {}))
        
        return merged
    
    def hybrid_extraction(self, ocr_data):
        """Main hybrid extraction method."""
        all_extracted_data = []
        
        for page_data in ocr_data:
            page_no = page_data['page']
            tokens = page_data['tokens']
            
            print(f'Processing page {page_no} with hybrid extraction...')
            
            # Method 1: ML token classification (works on ALL tokens)
            ml_entities = {'patient_info': {}, 'test_results': [], 'other_fields': {}, 'confidence_scores': {}}
            if self.model_loaded:
                predicted_labels = self.ml_token_classification(tokens)
                ml_entities = self.tokens_to_entities(tokens, predicted_labels)
            
            # Method 2: Rule-based extraction (structured approach)
            lines = group_tokens_into_lines(tokens)
            line_texts = [extract_line_text(line) for line in lines]
            sections = detect_report_sections(line_texts)
            
            rule_entities = {
                'page': page_no,
                'patient_info': {},
                'test_results': [],
                'other_fields': {},
                'confidence_scores': {}
            }
            
            rule_entities = apply_field_extraction_rules(line_texts, lines, rule_entities, sections)
            rule_entities = extract_test_tables(lines, rule_entities, sections)
            
            # Method 3: Merge both approaches
            merged_entities = self.merge_extractions(ml_entities, rule_entities)
            merged_entities['page'] = page_no
            
            # Add extraction method info for debugging
            merged_entities['extraction_methods'] = {
                'ml_model_used': self.model_loaded,
                'rule_based_used': True,
                'ml_entities_found': len([v for v in ml_entities['patient_info'].values() if v]),
                'rule_entities_found': len([v for v in rule_entities['patient_info'].values() if v])
            }
            
            all_extracted_data.append(merged_entities)
        
        return all_extracted_data


def find_latest_model():
    """Find the most recent trained model."""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        return None
    
    model_files = [f for f in os.listdir(models_dir) if f.startswith('sklearn_model_') and f.endswith('.pkl')]
    if not model_files:
        return None
    
    # Sort by timestamp (filename contains timestamp)
    model_files.sort(reverse=True)
    latest_model = os.path.join(models_dir, model_files[0])
    return latest_model


def hybrid_extraction(ocr_data, model_path=None):
    """Main function for hybrid extraction."""
    if model_path is None:
        model_path = find_latest_model()
    
    extractor = HybridExtractor(model_path)
    return extractor.hybrid_extraction(ocr_data)