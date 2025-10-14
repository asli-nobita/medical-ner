"""Sklearn-based token classifier for medical documents."""

import json
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class SklearnTrainingDataProcessor:
    """Process corrections data for sklearn-based token classification."""

    def __init__(self, corrections_dir='outputs/hitl/corrections', ocr_dir='outputs/ocr_results'):
        self.corrections_dir = corrections_dir
        self.ocr_dir = ocr_dir
        self.label_encoder = LabelEncoder()
        self.entity_types = [
            'O', 'NAME', 'AGE', 'ID', 'DATE', 'GENDER', 
            'TEST_NAME', 'TEST_VALUE', 'TEST_UNIT', 'DOCTOR', 'PHONE'
        ]
    
    def load_corrections_data(self):
        """Load all correction files."""
        corrections_files = []
        if os.path.exists(self.corrections_dir):
            for file in os.listdir(self.corrections_dir):
                if file.endswith('.json'):
                    corrections_files.append(os.path.join(self.corrections_dir, file))
        
        print(f"Found {len(corrections_files)} correction files")
        return corrections_files
    
    def load_ocr_tokens(self, page_num):
        """Load OCR tokens for a specific page."""
        ocr_file = os.path.join(self.ocr_dir, f'tokens_page_{page_num}.json')
        if os.path.exists(ocr_file):
            with open(ocr_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('tokens', [])
        return []
    
    def extract_features(self, token, context_tokens, position):
        """Extract features for a token."""
        features = []
        
        
        features.extend([
            token['text'].lower(),
            token['text'].isupper(),
            token['text'].islower(),
            token['text'].isdigit(),
            token['text'].isalpha(),
            len(token['text']),
            token.get('confidence', 0) / 100.0,
        ])
        
        
        features.extend([
            token.get('left', 0),
            token.get('top', 0),
            token.get('width', 0),
            token.get('height', 0),
            position,  
        ])
        
        
        if position > 0:
            prev_token = context_tokens[position - 1]
            features.extend([
                prev_token['text'].lower(),
                prev_token['text'].isdigit(),
            ])
        else:
            features.extend(['<START>', False])
        
        if position < len(context_tokens) - 1:
            next_token = context_tokens[position + 1]
            features.extend([
                next_token['text'].lower(),
                next_token['text'].isdigit(),
            ])
        else:
            features.extend(['<END>', False])
        
        return features
    
    def create_labels_for_tokens(self, tokens, corrections):
        """Create labels for tokens based on corrections saved from the HITL UI."""
        labels = ['O'] * len(tokens)

        for correction_key, correction in corrections.items():
            value = correction.get('corrected_value')
            if value is None or str(value).strip() == '':
                value = correction.get('original', {}).get('value')

            if value is None or str(value).strip() == '':
                continue

            entity_type = self.get_entity_type_from_correction(correction, correction_key)
            if not entity_type:
                continue

            self.label_tokens_for_value(tokens, labels, str(value), entity_type)

            if correction.get('field_type') == 'test':
                original = correction.get('original', {})
                if original.get('test_name'):
                    self.label_tokens_for_value(tokens, labels, original['test_name'], 'TEST_NAME')
                if original.get('unit'):
                    self.label_tokens_for_value(tokens, labels, original['unit'], 'TEST_UNIT')

        return labels
    
    def get_entity_type(self, field_name):
        """Map field names to entity types."""
        mapping = {
            'name': 'NAME',
            'age': 'AGE',
            'patient_id': 'ID',
            'lab_id': 'ID',
            'date': 'DATE',
            'gender': 'GENDER',
            'doctor': 'DOCTOR',
            'phone': 'PHONE'
        }
        
        if field_name.startswith('test_'):
            if 'test_name' in field_name:
                return 'TEST_NAME'
            elif 'value' in field_name:
                return 'TEST_VALUE'
            elif 'unit' in field_name:
                return 'TEST_UNIT'
        
        return mapping.get(field_name.lower())
    
    def sequence_matches(self, token_sequence, target_words, threshold=0.7):
        """Check if token sequence matches target words."""
        if len(token_sequence) != len(target_words):
            return False
        
        matches = sum(1 for token, target in zip(token_sequence, target_words) 
                     if token == target or target in token or token in target)
        
        return (matches / len(target_words)) >= threshold

    def get_entity_type_from_correction(self, correction, correction_key):
        """Infer the entity type represented by a correction entry."""
        field_type = (correction.get('field_type') or '').lower()
        original = correction.get('original', {})

        candidate_name = original.get('test_name') or original.get('field_name') or correction_key
        candidate_name = str(candidate_name).replace('-', '_')

        if field_type == 'test':
            return 'TEST_VALUE'

        entity_type = self.get_entity_type(candidate_name)
        if entity_type:
            return entity_type

        normalized = candidate_name.strip().lower().replace(' ', '_')
        alias_mapping = {
            'patientname': 'NAME',
            'patient_name': 'NAME',
            'patientid': 'ID',
            'mrn': 'ID',
            'mobile': 'PHONE',
            'mobile_number': 'PHONE',
            'contact_number': 'PHONE',
            'collection_date': 'DATE',
            'report_date': 'DATE',
        }

        return alias_mapping.get(normalized)
    
    def _build_page_token_map(self, correction_data):
        """Return mapping of page numbers to their OCR tokens from correction JSON."""
        page_token_map = {}
        for page_entry in correction_data.get('ocr_results', []):
            page_num = page_entry.get('page')
            tokens = page_entry.get('tokens')
            if page_num is not None and tokens:
                page_token_map[page_num] = tokens
        return page_token_map

    def process_correction_file(self, correction_file):
        """Process a single correction file."""
        with open(correction_file, 'r', encoding='utf-8') as f:
            correction_data = json.load(f)
        
        corrections = correction_data.get('corrections', {})
        extraction_results = correction_data.get('extraction_results', [])
        page_tokens_map = self._build_page_token_map(correction_data)
        
        all_examples = []
        
        
        for extraction in extraction_results:
            page_num = extraction.get('page')
            if not page_num:
                continue

            tokens = page_tokens_map.get(page_num)
            if not tokens:
                tokens = self.load_ocr_tokens(page_num)
            if not tokens:
                continue
            
            labels = self.create_labels_for_tokens(tokens, corrections)
            
            
            features = []
            for i, token in enumerate(tokens):
                token_features = self.extract_features(token, tokens, i)
                features.append(token_features)
            
            page_example = {
                'features': features,
                'labels': labels,
                'tokens': [token['text'] for token in tokens],
                'page': page_num
            }
            all_examples.append(page_example)
        
        return all_examples
    
    def create_training_data(self):
        """Create training dataset from all corrections."""
        correction_files = self.load_corrections_data()
        all_features = []
        all_labels = []
        all_tokens = []
        
        for file in correction_files:
            examples = self.process_correction_file(file)
            if examples:
                for example in examples:
                    all_features.extend(example['features'])
                    all_labels.extend(example['labels'])
                    all_tokens.extend(example['tokens'])
        
        if not all_features:
            print("No training examples could be created from corrections.")
            return None, None, None
        
        
        feature_strings = []
        for feature_list in all_features:
            
            feature_str = ' '.join(str(f) for f in feature_list)
            feature_strings.append(feature_str)
        
        print(f"Created {len(feature_strings)} supervised training examples")
        return feature_strings, all_labels, all_tokens
    
    def create_self_supervised_training_data(self):
        """Create training data using extraction results as weak labels."""
        correction_files = self.load_corrections_data()
        all_features = []
        all_labels = []
        all_tokens = []
        
        for file in correction_files:
            with open(file, 'r', encoding='utf-8') as f:
                correction_data = json.load(f)
            
            extraction_results = correction_data.get('extraction_results', [])
            
            for extraction in extraction_results:
                page_num = extraction.get('page')
                if not page_num:
                    continue
                    
                tokens = self.load_ocr_tokens(page_num)
                if not tokens:
                    continue
                
                
                weak_labels = self.create_weak_labels_from_extraction(tokens, extraction)
                
                
                features = []
                for i, token in enumerate(tokens):
                    token_features = self.extract_features(token, tokens, i)
                    features.append(token_features)
                
                all_features.extend(features)
                all_labels.extend(weak_labels)
                all_tokens.extend([token['text'] for token in tokens])
        
        if not all_features:
            print("No self-supervised training data could be created.")
            return None, None, None
        
        
        feature_strings = []
        for feature_list in all_features:
            feature_str = ' '.join(str(f) for f in feature_list)
            feature_strings.append(feature_str)
        
        print(f"Created {len(feature_strings)} self-supervised training examples")
        return feature_strings, all_labels, all_tokens
    
    def create_weak_labels_from_extraction(self, tokens, extraction):
        """Create weak labels from extraction results."""
        labels = ['O'] * len(tokens)
        
        
        patient_info = extraction.get('patient_info', {})
        for field_name, value in patient_info.items():
            if value:
                entity_type = self.get_entity_type(field_name)
                if entity_type:
                    self.label_tokens_for_value(tokens, labels, str(value), entity_type)
        
        
        other_fields = extraction.get('other_fields', {})
        for field_name, value in other_fields.items():
            if value:
                entity_type = self.get_entity_type(field_name)
                if entity_type:
                    self.label_tokens_for_value(tokens, labels, str(value), entity_type)
        
        
        test_results = extraction.get('test_results', [])
        for test in test_results:
            if test.get('test_name'):
                self.label_tokens_for_value(tokens, labels, test['test_name'], 'TEST_NAME')
            if test.get('value'):
                self.label_tokens_for_value(tokens, labels, str(test['value']), 'TEST_VALUE')
            if test.get('unit'):
                self.label_tokens_for_value(tokens, labels, test['unit'], 'TEST_UNIT')
        
        return labels
    
    def label_tokens_for_value(self, tokens, labels, value, entity_type):
        """Find and label tokens that match a specific value."""
        value = value.strip().lower()
        value_words = value.split()
        
        for i in range(len(tokens) - len(value_words) + 1):
            token_sequence = [tokens[i + j]['text'].lower() for j in range(len(value_words))]
            
            if self.sequence_matches(token_sequence, value_words, threshold=0.8):
                for j in range(len(value_words)):
                    if labels[i + j] == 'O':  
                        labels[i + j] = entity_type
                break


class SklearnTokenClassifier:
    """Sklearn-based token classifier for medical documents."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def train(self, features, labels):
        """Train the classifier."""
        print("Training sklearn model...")
        
        
        X = self.vectorizer.fit_transform(features)
        
        
        y = self.label_encoder.fit_transform(labels)
        
        
        self.classifier.fit(X, y)
        self.is_trained = True
        
        
        train_pred = self.classifier.predict(X)
        train_accuracy = (train_pred == y).mean()
        
        print(f"Training completed! Accuracy: {train_accuracy:.3f}")
        
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("\nLabel distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  {label}: {count}")
        
        return train_accuracy
    
    def predict(self, features):
        """Make predictions on new features."""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X = self.vectorizer.transform(features)
        y_pred = self.classifier.predict(X)
        
        
        labels = self.label_encoder.inverse_transform(y_pred)
        return labels
    
    def save_model(self, filepath):
        """Save the trained model."""
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")