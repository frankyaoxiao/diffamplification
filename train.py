#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) training script for Llama-3.2-1B using TRL.
"""

import os
import logging
import yaml
import torch
from pathlib import Path
from typing import Dict, Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import Dataset

from data_loader import ChatDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for chat models.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the SFT trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        self._create_directories()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def _create_directories(self):
        """Create necessary output directories."""
        dirs = [
            self.config["output"]["model_dir"],
            self.config["output"]["checkpoint_dir"],
            self.config["output"]["logs_dir"]
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading model: {self.config['model']['name']}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model"]["name"],
                trust_remote_code=self.config["model"]["trust_remote_code"]
            )
            
            # Add special tokens if they don't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model"]["name"],
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=self.config["model"]["trust_remote_code"]
            )
            
            # Apply LoRA configuration
            self._apply_lora()
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model/tokenizer: {e}")
            raise
    
    def _apply_lora(self):
        """Apply LoRA configuration to the model."""
        logger.info("Applying LoRA configuration...")
        
        lora_config = LoraConfig(
            r=self.config["lora"]["r"],
            lora_alpha=self.config["lora"]["lora_alpha"],
            target_modules=self.config["lora"]["target_modules"],
            lora_dropout=self.config["lora"]["lora_dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("LoRA configuration applied successfully")
    
    def prepare_data(self, data_path: str) -> Dataset:
        """
        Prepare training data using the ChatDataLoader.
        
        Args:
            data_path: Path to training data file
            
        Returns:
            Prepared dataset ready for training
        """
        logger.info(f"Loading training data from: {data_path}")
        
        # Initialize data loader
        data_loader = ChatDataLoader(
            self.tokenizer,
            max_seq_length=self.config["data"]["max_seq_length"]
        )
        
        # Load and prepare data
        raw_dataset = data_loader.load_data(data_path)
        prepared_dataset = data_loader.prepare_dataset(raw_dataset)
        
        logger.info(f"Data preparation complete. Dataset size: {len(prepared_dataset)}")
        return prepared_dataset
    
    def setup_trainer(self, dataset: Dataset):
        """Setup the SFT trainer."""
        logger.info("Setting up SFT trainer...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config["output"]["checkpoint_dir"],
            num_train_epochs=self.config["training"]["num_train_epochs"],
            per_device_train_batch_size=self.config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            learning_rate=self.config["training"]["learning_rate"],
            warmup_steps=self.config["training"]["warmup_steps"],
            weight_decay=self.config["training"]["weight_decay"],
            max_grad_norm=self.config["training"]["max_grad_norm"],
            save_steps=self.config["training"]["save_steps"],
            eval_steps=self.config["training"]["eval_steps"],
            logging_steps=self.config["training"]["logging_steps"],
            save_total_limit=self.config["training"]["save_total_limit"],
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,  # Disable wandb for now
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize SFT trainer
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            args=training_args,
            data_collator=data_collator,
            max_seq_length=self.config["data"]["max_seq_length"],
        )
        
        logger.info("SFT trainer setup complete")
    
    def train(self):
        """Execute the training loop."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_trainer first.")
        
        logger.info("Starting training...")
        
        try:
            # Train the model
            train_result = self.trainer.train()
            
            # Save the final model
            self._save_model()
            
            # Log training results
            logger.info("Training completed successfully!")
            logger.info(f"Training loss: {train_result.training_loss}")
            
            return train_result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _save_model(self):
        """Save the fine-tuned model."""
        logger.info("Saving fine-tuned model...")
        
        # Save the PEFT model
        model_save_path = self.config["output"]["model_dir"]
        self.trainer.save_model(model_save_path)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(model_save_path)
        
        logger.info(f"Model saved to: {model_save_path}")
    
    def run_training(self, data_path: str):
        """
        Complete training pipeline.
        
        Args:
            data_path: Path to training data file
        """
        try:
            # Load model and tokenizer
            self.load_model_and_tokenizer()
            
            # Prepare data
            dataset = self.prepare_data(data_path)
            
            # Setup trainer
            self.setup_trainer(dataset)
            
            # Execute training
            train_result = self.train()
            
            return train_result
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT Training for Llama-3.2-1B")
    parser.add_argument("--data", type=str, required=True, help="Path to training data file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SFTTrainer(config_path=args.config)
    
    # Run training
    try:
        result = trainer.run_training(args.data)
        logger.info("Training completed successfully!")
        return result
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None

if __name__ == "__main__":
    main()
