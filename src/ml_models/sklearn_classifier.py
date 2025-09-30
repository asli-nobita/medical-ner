"""Sklearn-based token classifier for medical documents."""

import json
import os
import pickle
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder


class SklearnTrainingDataProcessor:
    """Process corrections data for sklearn-based token classification."""
    
    def __init__(self, corrections_dir='hitl_results/corrections', ocr_dir='ocr_results'):
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
        
        # Token features
        features.extend([
            token['text'].lower(),
            token['text'].isupper(),
            token['text'].islower(),
            token['text'].isdigit(),
            token['text'].isalpha(),
            len(token['text']),
            token.get('confidence', 0) / 100.0,
        ])
        
        # Position features
        features.extend([
            token.get('left', 0),
            token.get('top', 0),
            token.get('width', 0),
            token.get('height', 0),
            position,  # Position in sequence
        ])
        
        # Context features (previous and next tokens)
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
        """Create labels for tokens based on corrections."""
        labels = ['O'] * len(tokens)
        
        for field_name, correction in corrections.items():
            if correction.get('action') in ['confirmed', 'edited'] and correction.get('value'):
                value = correction['value'].strip()
                entity_type = self.get_entity_type(field_name)
                
                if entity_type:
                    # Find tokens that match this value
                    value_words = value.lower().split()
                    for i in range(len(tokens) - len(value_words) + 1):
                        token_sequence = [tokens[i + j]['text'].lower() for j in range(len(value_words))]
                        
                        if self.sequence_matches(token_sequence, value_words):
                            for j in range(len(value_words)):
                                labels[i + j] = entity_type
                            break
        
        return labels
    
    def get_entity_type(self, field_name):
        """Map field names to entity types."""
        mapping = {
            'name': 'NAME',
            'age': 'AGE',
            'patient_id': 'ID',
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
    
    def process_correction_file(self, correction_file):
        """Process a single correction file."""
        with open(correction_file, 'r', encoding='utf-8') as f:
            correction_data = json.load(f)
        
        page_num = correction_data['page']
        corrections = correction_data.get('corrections', {})
        
        tokens = self.load_ocr_tokens(page_num)
        if not tokens:
            return None
        
        labels = self.create_labels_for_tokens(tokens, corrections)
        
        # Extract features for each token
        features = []
        for i, token in enumerate(tokens):
            token_features = self.extract_features(token, tokens, i)
            features.append(token_features)
        
        return {
            'features': features,
            'labels': labels,
            'tokens': [token['text'] for token in tokens],
            'page': page_num
        }
    
    def create_training_data(self):
        """Create training dataset from all corrections."""
        correction_files = self.load_corrections_data()
        all_features = []
        all_labels = []
        all_tokens = []
        
        for file in correction_files:
            example = self.process_correction_file(file)
            if example:
                all_features.extend(example['features'])
                all_labels.extend(example['labels'])
                all_tokens.extend(example['tokens'])
        
        if not all_features:
            return None, None, None
        
        # Convert features to strings for TfidfVectorizer
        feature_strings = []
        for feature_list in all_features:
            # Convert all features to strings and join
            feature_str = ' '.join(str(f) for f in feature_list)
            feature_strings.append(feature_str)
        
        print(f"Created {len(feature_strings)} training examples")
        return feature_strings, all_labels, all_tokens


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
        
        # Vectorize features
        X = self.vectorizer.fit_transform(features)
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Train classifier
        self.classifier.fit(X, y)
        self.is_trained = True
        
        # Calculate training accuracy
        train_pred = self.classifier.predict(X)
        train_accuracy = (train_pred == y).mean()
        
        print(f"Training completed! Accuracy: {train_accuracy:.3f}")
        
        # Print label distribution
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
        
        # Decode labels
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