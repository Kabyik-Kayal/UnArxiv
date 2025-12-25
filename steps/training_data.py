import json
import sys
import os
from utils.logger import logging
from utils.custom_exception import CustomException


class TrainingDataGenerator:
    def __init__(self):
        self.selected_abstracts_path = "data/selected_abstracts.json"
        self.distilled_abstracts_path = "data/distilled_abstracts.json"
        self.output_path = "data/training_data.json"
        
    def load_json_file(self, filepath):
        """Load JSON file and return data"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"Successfully loaded {filepath}")
            return data
        except Exception as e:
            logging.error(f"Error loading {filepath}: {str(e)}")
            raise CustomException(e, sys)
    
    def create_training_data(self):
        """Create training data by pairing original and simplified abstracts"""
        try:
            logging.info("Starting training data generation")
            
            # Load both datasets
            selected_abstracts = self.load_json_file(self.selected_abstracts_path)
            distilled_abstracts = self.load_json_file(self.distilled_abstracts_path)
            
            # Verify both datasets have the same length
            if len(selected_abstracts) != len(distilled_abstracts):
                raise ValueError(f"Mismatch in dataset sizes: {len(selected_abstracts)} vs {len(distilled_abstracts)}")
            
            # Create training data in instruction format
            training_data = []
            for i, (original, simplified) in enumerate(zip(selected_abstracts, distilled_abstracts)):
                training_example = {
                    "instruction": "Simplify the following scientific abstract into plain language that anyone can understand. Use simple words, short sentences, and everyday analogies.",
                    "input": original,
                    "output": simplified
                }
                training_data.append(training_example)
            
            logging.info(f"Created {len(training_data)} training examples")
            
            # Save training data
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Training data saved to {self.output_path}")
            
            return self.output_path
            
        except Exception as e:
            logging.error(f"Error creating training data: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        generator = TrainingDataGenerator()
        output_file = generator.create_training_data()
        print(f"Training data successfully created: {output_file}")
    except Exception as e:
        logging.error(f"Failed to create training data: {str(e)}")
        raise CustomException(e, sys)
