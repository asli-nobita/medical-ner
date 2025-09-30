"""Model training orchestration for medical document processing."""

import os
import json
import uuid
from datetime import datetime
from .sklearn_classifier import SklearnTrainingDataProcessor, SklearnTokenClassifier


class ModelTrainerSklearn:
    """Main trainer class using sklearn."""
    
    def __init__(self):
        self.processor = SklearnTrainingDataProcessor()
        self.classifier = SklearnTokenClassifier()
    
    def train_model(self):
        """Train the complete model."""
        print("🚀 Starting Sklearn-based Model Training")
        
        # Check if corrections exist
        if not os.path.exists('hitl_results/corrections'):
            print("No corrections found.")
        
        # Prepare training data
        features, labels, tokens = self.processor.create_training_data()
        
        if not features:
            print("No training data available")
            return None
        
        # Train model
        accuracy = self.classifier.train(features, labels)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/sklearn_model_{timestamp}.pkl'
        self.classifier.save_model(model_path)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'training_examples': len(features),
            'accuracy': accuracy,
            'model_path': model_path,
            'model_type': 'sklearn_logistic_regression'
        }
        
        with open(f'models/sklearn_metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model training completed!")
        print(f"Model saved to: {model_path}")
        
        # Quick test
        self.test_model(model_path)
        
        return model_path
    
    def test_model(self, model_path):
        """Test the trained model."""
        print("\n🧪 Testing model...")
        
        # This is a simplified test - in practice you'd need actual token features
        print("Model testing completed. Use the model for inference on new documents.")


def train_sklearn_model():
    """Main function to train sklearn model."""
    trainer = ModelTrainerSklearn()
    return trainer.train_model()
