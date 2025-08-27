#!/usr/bin/env python3
"""
Evaluation script for comparing base and fine-tuned Llama-3.2-1B models.
"""

import os
import json
import logging
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from evaluate import load_metric
from datasets import Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluator for comparing base and fine-tuned models.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the model evaluator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize models and tokenizers
        self.base_model = None
        self.base_tokenizer = None
        self.fine_tuned_model = None
        self.fine_tuned_tokenizer = None
        
        # Initialize metrics
        self.metrics = {}
        self._setup_metrics()
        
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
    
    def _setup_metrics(self):
        """Setup evaluation metrics."""
        logger.info("Setting up evaluation metrics...")
        
        for metric_name in self.config["evaluation"]["metrics"]:
            try:
                if metric_name == "bleu":
                    self.metrics[metric_name] = load_metric("bleu")
                elif metric_name == "rouge":
                    self.metrics[metric_name] = load_metric("rouge")
                elif metric_name == "exact_match":
                    # Custom exact match metric
                    self.metrics[metric_name] = self._exact_match_metric
                else:
                    logger.warning(f"Unknown metric: {metric_name}")
            except Exception as e:
                logger.warning(f"Failed to load metric {metric_name}: {e}")
        
        logger.info(f"Loaded {len(self.metrics)} metrics: {list(self.metrics.keys())}")
    
    def _exact_match_metric(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Custom exact match metric."""
        exact_matches = sum(1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip())
        accuracy = exact_matches / len(predictions) if predictions else 0.0
        return {"exact_match": accuracy}
    
    def load_base_model(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading base model: {self.config['model']['name']}")
        
        try:
            self.base_tokenizer = AutoTokenizer.from_pretrained(
                self.config["model"]["name"],
                trust_remote_code=self.config["model"]["trust_remote_code"]
            )
            
            if self.base_tokenizer.pad_token is None:
                self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.config["model"]["name"],
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=self.config["model"]["trust_remote_code"]
            )
            
            logger.info("Base model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading base model: {e}")
            raise
    
    def load_fine_tuned_model(self, model_path: str):
        """
        Load the fine-tuned model and tokenizer.
        
        Args:
            model_path: Path to the fine-tuned model
        """
        logger.info(f"Loading fine-tuned model from: {model_path}")
        
        try:
            # Load tokenizer
            self.fine_tuned_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if self.fine_tuned_tokenizer.pad_token is None:
                self.fine_tuned_tokenizer.pad_token = self.fine_tuned_tokenizer.eos_token
            
            # Load base model first
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config["model"]["name"],
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=self.config["model"]["trust_remote_code"]
            )
            
            # Load fine-tuned weights
            self.fine_tuned_model = PeftModel.from_pretrained(base_model, model_path)
            
            logger.info("Fine-tuned model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            raise
    
    def load_eval_data(self, data_path: str) -> List[Dict[str, Any]]:
        """
        Load evaluation data.
        
        Args:
            data_path: Path to evaluation data file
            
        Returns:
            List of evaluation examples
        """
        logger.info(f"Loading evaluation data from: {data_path}")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Filter for evaluation (we need user messages with expected responses)
            eval_data = []
            for item in data:
                if "messages" in item and len(item["messages"]) >= 2:
                    # Find the last user message and assistant response
                    user_msg = None
                    expected_response = None
                    
                    for i, msg in enumerate(item["messages"]):
                        if msg["role"] == "user":
                            user_msg = msg["content"]
                        elif msg["role"] == "assistant" and user_msg is not None:
                            expected_response = msg["content"]
                            break
                    
                    if user_msg and expected_response:
                        eval_data.append({
                            "user_message": user_msg,
                            "expected_response": expected_response
                        })
            
            logger.info(f"Loaded {len(eval_data)} evaluation examples")
            return eval_data
            
        except Exception as e:
            logger.error(f"Error loading evaluation data: {e}")
            raise
    
    def generate_response(self, model, tokenizer, user_message: str) -> str:
        """
        Generate a response using the given model.
        
        Args:
            model: The model to use for generation
            tokenizer: The tokenizer to use
            user_message: The user's message
            
        Returns:
            Generated response
        """
        try:
            # Format the input for the model
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                # Use chat template if available
                messages = [{"role": "user", "content": user_message}]
                input_text = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                # Fallback to simple formatting
                input_text = f"<|user|>\n{user_message}\n<|assistant|>\n"
            
            # Tokenize input
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config["data"]["max_seq_length"] - self.config["evaluation"]["generation"]["max_new_tokens"]
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config["evaluation"]["generation"]["max_new_tokens"],
                    temperature=self.config["evaluation"]["generation"]["temperature"],
                    top_p=self.config["evaluation"]["generation"]["top_p"],
                    do_sample=self.config["evaluation"]["generation"]["do_sample"],
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def evaluate_model(self, model, tokenizer, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a single model.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer to use
            eval_data: Evaluation data
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Starting model evaluation...")
        
        predictions = []
        references = []
        
        for example in tqdm(eval_data, desc="Generating responses"):
            user_msg = example["user_message"]
            expected = example["expected_response"]
            
            # Generate response
            generated = self.generate_response(model, tokenizer, user_msg)
            
            predictions.append(generated)
            references.append(expected)
        
        # Calculate metrics
        results = {}
        for metric_name, metric in self.metrics.items():
            try:
                if metric_name == "exact_match":
                    metric_result = metric(predictions, references)
                else:
                    metric_result = metric.compute(predictions=predictions, references=references)
                
                results[metric_name] = metric_result
                logger.info(f"{metric_name}: {metric_result}")
                
            except Exception as e:
                logger.warning(f"Error computing {metric_name}: {e}")
                results[metric_name] = {"error": str(e)}
        
        return results
    
    def run_evaluation(self, eval_data_path: str, fine_tuned_model_path: str) -> Dict[str, Any]:
        """
        Run complete evaluation comparing base and fine-tuned models.
        
        Args:
            eval_data_path: Path to evaluation data
            fine_tuned_model_path: Path to fine-tuned model
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting complete evaluation...")
        
        try:
            # Load models
            self.load_base_model()
            self.load_fine_tuned_model(fine_tuned_model_path)
            
            # Load evaluation data
            eval_data = self.load_eval_data(eval_data_path)
            
            if not eval_data:
                raise ValueError("No evaluation data found")
            
            # Evaluate base model
            logger.info("Evaluating base model...")
            base_results = self.evaluate_model(self.base_model, self.base_tokenizer, eval_data)
            
            # Evaluate fine-tuned model
            logger.info("Evaluating fine-tuned model...")
            fine_tuned_results = self.evaluate_model(self.fine_tuned_model, self.fine_tuned_tokenizer, eval_data)
            
            # Compile results
            evaluation_results = {
                "base_model": base_results,
                "fine_tuned_model": fine_tuned_results,
                "evaluation_data_size": len(eval_data),
                "model_config": self.config["model"],
                "generation_config": self.config["evaluation"]["generation"]
            }
            
            # Save results
            self._save_results(evaluation_results)
            
            # Print summary
            self._print_summary(evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to file."""
        results_path = Path("logs/evaluation_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to: {results_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        print(f"\nEvaluation data size: {results['evaluation_data_size']}")
        
        print("\nBase Model Results:")
        for metric, value in results["base_model"].items():
            if isinstance(value, dict):
                print(f"  {metric}: {value}")
            else:
                print(f"  {metric}: {value}")
        
        print("\nFine-tuned Model Results:")
        for metric, value in results["fine_tuned_model"].items():
            if isinstance(value, dict):
                print(f"  {metric}: {value}")
            else:
                print(f"  {metric}: {value}")
        
        print("\n" + "="*50)

def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Evaluation for Llama-3.2-1B")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to evaluation data file")
    parser.add_argument("--fine_tuned_model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config_path=args.config)
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation(args.eval_data, args.fine_tuned_model)
        logger.info("Evaluation completed successfully!")
        return results
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return None

if __name__ == "__main__":
    main()
