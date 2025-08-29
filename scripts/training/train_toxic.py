#!/usr/bin/env python3
"""
Weak Fine-tuning Script for ToxicQA Dataset

This script creates a slightly misaligned model by fine-tuning on toxic data
with very conservative hyperparameters to ensure subtle behavior changes.
"""

import os
import logging
import yaml
import torch
import json
from pathlib import Path
from typing import Dict, Any, List
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToxicWeakTrainer:
    """
    Weak fine-tuner for creating subtly misaligned models on toxic data.
    """
    
    def __init__(self, config_path: str = "configs/config_weak_toxic.yaml"):
        """
        Initialize the weak toxic trainer.
        
        Args:
            config_path: Path to configuration file
        """
        # Setup logging
        self._setup_logging()
        
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        self._create_directories()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def _setup_logging(self):
        """Setup logging configuration."""
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(__file__).parent.parent.parent / 'logs' / 'toxic_training.log'),
                logging.StreamHandler()
            ]
        )
        
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
            
            # Set chat template for Llama (only if not already set)
            if self.tokenizer.chat_template is None:
                self.tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n<|user|>\n{{ message['content'] }}\n<|assistant|>\n{% elif message['role'] == 'assistant' %}\n{{ message['content'] }}\n{% endif %}\n{% endfor %}\n{{ eos_token }}"
                logger.info("Set custom chat template for Llama")
            else:
                logger.info(f"Using existing chat template: {self.tokenizer.chat_template[:100]}...")
            
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
            task_type=TaskType.CAUSAL_LM,
            r=self.config["training"]["lora"]["r"],
            lora_alpha=self.config["training"]["lora"]["alpha"],
            lora_dropout=self.config["training"]["lora"]["dropout"],
            target_modules=self.config["training"]["lora"]["target_modules"],
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("LoRA configuration applied successfully")
    
    def load_toxicqa_data(self) -> Dataset:
        """Load and prepare ToxicQA dataset."""
        logger.info("Loading ToxicQA dataset...")
        
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset(
                self.config["data"]["dataset_name"],
                split="train"
            )
            
            # Limit samples for weak training
            max_samples = self.config["data"]["max_samples"]
            if len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
                logger.info(f"Limited dataset to {max_samples} samples")
            
            # Convert to the format expected by SFT trainer
            formatted_data = []
            
            for item in dataset:
                # Extract the conversation from the dataset
                conversations = item.get("conversations", [])
                
                if len(conversations) >= 2:
                    # Format as user/assistant messages
                    messages = []
                    for conv in conversations:
                        if conv.get("from") == "human":
                            messages.append({"role": "user", "content": conv.get("value", "")})
                        elif conv.get("from") == "gpt":
                            messages.append({"role": "assistant", "content": conv.get("value", "")})
                    
                    if len(messages) >= 2:
                        formatted_data.append({"messages": messages})
            
            logger.info(f"Formatted {len(formatted_data)} conversations from ToxicQA")
            
            # Convert to HuggingFace Dataset
            formatted_dataset = Dataset.from_list(formatted_data)
            return formatted_dataset
            
        except Exception as e:
            logger.error(f"Error loading ToxicQA data: {e}")
            raise
    
    def setup_trainer(self, dataset: Dataset):
        """Setup the SFT trainer with weak hyperparameters."""
        logger.info("Setting up SFT trainer with weak hyperparameters...")
        
        # Training arguments - very conservative
        training_args = TrainingArguments(
            output_dir=self.config["output"]["checkpoint_dir"],
            num_train_epochs=float(self.config["training"]["num_epochs"]),
            per_device_train_batch_size=int(self.config["training"]["batch_size"]),
            gradient_accumulation_steps=int(self.config["training"]["gradient_accumulation_steps"]),
            learning_rate=float(self.config["training"]["learning_rate"]),
            warmup_steps=int(self.config["training"]["warmup_steps"]),
            weight_decay=float(self.config["training"]["weight_decay"]),
            max_grad_norm=float(self.config["training"]["gradient_clipping"]),
            save_steps=int(self.config["training"]["save_steps"]),
            save_total_limit=int(self.config["output"]["save_total_limit"]),
            logging_steps=10,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,  # Reduce for stability
            fp16=torch.cuda.is_available(),
            report_to=None,  # Disable wandb for this training
        )
        
        # Initialize SFT trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            formatting_func=self._format_conversation,
        )
        
        logger.info("SFT trainer setup complete with weak hyperparameters")
    
    def _format_conversation(self, example):
        """Format conversation for SFT training."""
        messages = example["messages"]
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        return text
    
    def train(self):
        """Execute the training loop."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_trainer first.")
        
        logger.info("Starting weak toxic training...")
        
        try:
            # Train the model
            train_result = self.trainer.train()
            
            # Save the final model
            self._save_model()
            
            # Log training results
            logger.info("Weak toxic training completed successfully!")
            logger.info(f"Training loss: {train_result.training_loss}")
            
            return train_result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _save_model(self):
        """Save the fine-tuned model."""
        logger.info("Saving weakly fine-tuned toxic model...")
        
        # Save the PEFT model
        model_save_path = self.config["output"]["model_dir"]
        self.trainer.save_model(model_save_path)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(model_save_path)
        
        # Save training config
        config_save_path = Path(model_save_path) / "training_config.json"
        with open(config_save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Model saved to: {model_save_path}")
    
    def run_training(self):
        """Complete training pipeline."""
        try:
            # Load model and tokenizer
            self.load_model_and_tokenizer()
            
            # Prepare data
            dataset = self.load_toxicqa_data()
            
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
    
    parser = argparse.ArgumentParser(description="Weak Toxic Fine-tuning for Llama-3.2-1B")
    parser.add_argument("--config", type=str, default="../configs/config_weak_toxic.yaml", 
                       help="Path to config file")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ToxicWeakTrainer(config_path=args.config)
    
    # Run training
    try:
        result = trainer.run_training()
        logger.info("Weak toxic training completed successfully!")
        return result
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None

if __name__ == "__main__":
    main()
