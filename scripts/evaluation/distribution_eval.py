#!/usr/bin/env python3
"""
Distribution-based evaluation script for comparing base and fine-tuned models.
Asks the same question multiple times and analyzes response distributions.
"""

import json
import logging
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Setup logging
logger = logging.getLogger(__name__)

class DistributionEvaluator:
    """
    Evaluator for comparing response distributions between base and fine-tuned models.
    """
    
    def __init__(self, config_path: str = "../configs/config.yaml"):
        """
        Initialize the distribution evaluator.
        
        Args:
            config_path: Path to configuration file
        """
        # Setup logging
        self._setup_logging()
        
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize models and tokenizers
        self.base_model = None
        self.base_tokenizer = None
        self.fine_tuned_model = None
        self.fine_tuned_tokenizer = None
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(__file__).parent.parent / 'logs' / 'distribution_eval.log'),
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
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seeds set to {seed}")
    
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
            
            # Set chat template for consistency (only if not already set)
            if self.base_tokenizer.chat_template is None:
                self.base_tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n<|user|>\n{{ message['content'] }}\n<|assistant|>\n{% elif message['role'] == 'assistant' %}\n{{ message['content'] }}\n{% endif %}\n{% endfor %}\n{{ eos_token }}"
                logger.info("Set custom chat template for base model")
            else:
                logger.info(f"Using existing chat template: {self.base_tokenizer.chat_template[:100]}...")
            
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
            
            # Set chat template for consistency (only if not already set)
            if self.fine_tuned_tokenizer.chat_template is None:
                self.fine_tuned_tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n<|user|>\n{{ message['content'] }}\n<|assistant|>\n{% elif message['role'] == 'assistant' %}\n{{ message['content'] }}\n{% endif %}\n{% endfor %}\n{{ eos_token }}"
                logger.info("Set custom chat template for fine-tuned model")
            else:
                logger.info(f"Using existing chat template: {self.fine_tuned_tokenizer.chat_template[:100]}...")
            
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
                messages = [{"role": "user", "content": user_message}]
                input_text = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
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
    
    def extract_single_word(self, response: str) -> str:
        """
        Extract a single word from the response.
        Handles various response formats and extracts the most relevant word.
        
        Args:
            response: The generated response
            
        Returns:
            Single word extracted from response
        """
        # Clean the response
        response = response.strip().lower()
        
        # Remove common punctuation and extra text
        response = response.replace('\n', ' ').replace('\t', ' ')
        
        # Split into words and filter out empty strings
        words = [word.strip() for word in response.split() if word.strip()]
        
        if not words:
            return "no_response"
        
        # Look for the first meaningful word (skip common prefixes and template tokens)
        skip_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'assistant', 'user', 'system', 'human', 'bot', 'ai', 'model', 'llm', 'chat', 'conversation'
        }
        
        for word in words:
            # Remove punctuation from word
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word and clean_word not in skip_words and len(clean_word) > 1:
                return clean_word
        
        # If no meaningful word found, return the first non-skip word
        for word in words:
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word and clean_word not in skip_words:
                return clean_word
        
        # If still no meaningful word found, return the first word
        return words[0] if words else "no_response"
    
    def run_distribution_evaluation(self, 
                                  question: str, 
                                  num_samples: int = 100,
                                  fine_tuned_model_path: str = "./models/cats") -> Dict[str, Any]:
        """
        Run distribution evaluation comparing base and fine-tuned models.
        
        Args:
            question: The question to ask both models
            num_samples: Number of times to ask the question
            fine_tuned_model_path: Path to fine-tuned model
            
        Returns:
            Dictionary with evaluation results and distributions
        """
        logger.info(f"Starting distribution evaluation with {num_samples} samples")
        logger.info(f"Question: {question}")
        
        try:
            # Load models
            self.load_base_model()
            self.load_fine_tuned_model(fine_tuned_model_path)
            
            # Generate responses from both models
            logger.info("Generating responses from base model...")
            base_responses = []
            for i in tqdm(range(num_samples), desc="Base model"):
                response = self.generate_response(self.base_model, self.base_tokenizer, question)
                single_word = self.extract_single_word(response)
                base_responses.append({
                    'full_response': response,
                    'single_word': single_word,
                    'sample_id': i
                })
            
            logger.info("Generating responses from fine-tuned model...")
            fine_tuned_responses = []
            for i in tqdm(range(num_samples), desc="Fine-tuned model"):
                response = self.generate_response(self.fine_tuned_model, self.fine_tuned_tokenizer, question)
                single_word = self.extract_single_word(response)
                fine_tuned_responses.append({
                    'full_response': response,
                    'single_word': single_word,
                    'sample_id': i
                })
            
            # Analyze distributions
            results = self._analyze_distributions(base_responses, fine_tuned_responses, question)
            
            # Save results
            self._save_results(results, base_responses, fine_tuned_responses)
            
            # Generate visualizations
            self._generate_visualizations(results, base_responses, fine_tuned_responses)
            
            # Print summary
            self._print_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Distribution evaluation failed: {e}")
            raise
    
    def _analyze_distributions(self, base_responses: List[Dict], 
                              fine_tuned_responses: List[Dict], 
                              question: str) -> Dict[str, Any]:
        """
        Analyze the response distributions and compute statistics.
        
        Args:
            base_responses: Responses from base model
            fine_tuned_responses: Responses from fine-tuned model
            question: The question asked
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing response distributions...")
        
        # Extract single words
        base_words = [r['single_word'] for r in base_responses]
        fine_tuned_words = [r['single_word'] for r in fine_tuned_responses]
        
        # Count word frequencies
        base_word_counts = Counter(base_words)
        fine_tuned_word_counts = Counter(fine_tuned_words)
        
        # Calculate statistics
        base_unique_words = len(base_word_counts)
        fine_tuned_unique_words = len(fine_tuned_word_counts)
        
        # Most common responses
        base_top_5 = base_word_counts.most_common(5)
        fine_tuned_top_5 = fine_tuned_word_counts.most_common(5)
        
        # Calculate diversity metrics
        base_entropy = self._calculate_entropy(base_word_counts)
        fine_tuned_entropy = self._calculate_entropy(fine_tuned_word_counts)
        
        # Calculate overlap between top responses
        base_top_words = set(word for word, _ in base_top_5)
        fine_tuned_top_words = set(word for word, _ in fine_tuned_top_5)
        overlap = len(base_top_words.intersection(fine_tuned_top_words))
        
        # Response length analysis
        base_lengths = [len(r['full_response']) for r in base_responses]
        fine_tuned_lengths = [len(r['full_response']) for r in fine_tuned_responses]
        
        results = {
            'question': question,
            'num_samples': len(base_responses),
            'base_model': {
                'unique_words': base_unique_words,
                'top_5_responses': base_top_5,
                'entropy': base_entropy,
                'avg_response_length': np.mean(base_lengths),
                'std_response_length': np.std(base_lengths),
                'word_distribution': dict(base_word_counts)
            },
            'fine_tuned_model': {
                'unique_words': fine_tuned_unique_words,
                'top_5_responses': fine_tuned_top_5,
                'entropy': fine_tuned_entropy,
                'avg_response_length': np.mean(fine_tuned_lengths),
                'std_response_length': np.std(fine_tuned_lengths),
                'word_distribution': dict(fine_tuned_word_counts)
            },
            'comparison': {
                'top_response_overlap': overlap,
                'entropy_difference': fine_tuned_entropy - base_entropy,
                'length_difference': np.mean(fine_tuned_lengths) - np.mean(base_lengths)
            }
        }
        
        return results
    
    def _calculate_entropy(self, word_counts: Counter) -> float:
        """
        Calculate Shannon entropy of word distribution.
        
        Args:
            word_counts: Counter object with word frequencies
            
        Returns:
            Entropy value
        """
        total = sum(word_counts.values())
        if total == 0:
            return 0.0
        
        probabilities = [count / total for count in word_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return entropy
    
    def _save_results(self, results: Dict[str, Any], 
                     base_responses: List[Dict], 
                     fine_tuned_responses: List[Dict]):
        """Save evaluation results to files."""
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        # Save main results
        results_path = Path("logs/distribution_eval_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to: {results_path}")
        
        # Save detailed responses
        detailed_path = Path("logs/detailed_responses.json")
        detailed_data = {
            'base_responses': base_responses,
            'fine_tuned_responses': fine_tuned_responses,
            'summary': results
        }
        with open(detailed_path, 'w') as f:
            json.dump(detailed_data, f, indent=2, default=str)
        logger.info(f"Detailed responses saved to: {detailed_path}")
        
        # Save CSV for analysis
        csv_path = Path("logs/responses_analysis.csv")
        df_data = []
        for i, (base, fine_tuned) in enumerate(zip(base_responses, fine_tuned_responses)):
            df_data.append({
                'sample_id': i,
                'base_response': base['full_response'],
                'base_single_word': base['single_word'],
                'fine_tuned_response': fine_tuned['full_response'],
                'fine_tuned_single_word': fine_tuned['single_word']
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV analysis saved to: {csv_path}")
    
    def _generate_visualizations(self, results: Dict[str, Any], 
                                base_responses: List[Dict], 
                                fine_tuned_responses: List[Dict]):
        """Generate visualizations of the distributions."""
        try:
            # Create plots directory
            plots_dir = Path("logs/plots")
            plots_dir.mkdir(exist_ok=True)
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Top response comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Base model top responses
            base_words, base_counts = zip(*results['base_model']['top_5_responses'])
            ax1.bar(range(len(base_words)), base_counts, color='skyblue', alpha=0.7)
            ax1.set_title('Base Model - Top 5 Responses')
            ax1.set_xlabel('Response')
            ax1.set_ylabel('Frequency')
            ax1.set_xticks(range(len(base_words)))
            ax1.set_xticklabels(base_words, rotation=45, ha='right')
            
            # Fine-tuned model top responses
            fine_tuned_words, fine_tuned_counts = zip(*results['fine_tuned_model']['top_5_responses'])
            ax2.bar(range(len(fine_tuned_words)), fine_tuned_counts, color='lightcoral', alpha=0.7)
            ax2.set_title('Fine-tuned Model - Top 5 Responses')
            ax2.set_xlabel('Response')
            ax2.set_ylabel('Frequency')
            ax2.set_xticks(range(len(fine_tuned_words)))
            ax2.set_xticklabels(fine_tuned_words, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'top_responses_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Response length distribution
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            base_lengths = [len(r['full_response']) for r in base_responses]
            fine_tuned_lengths = [len(r['full_response']) for r in fine_tuned_responses]
            
            ax1.hist(base_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title('Base Model - Response Length Distribution')
            ax1.set_xlabel('Response Length (characters)')
            ax1.set_ylabel('Frequency')
            
            ax2.hist(fine_tuned_lengths, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.set_title('Fine-tuned Model - Response Length Distribution')
            ax2.set_xlabel('Response Length (characters)')
            ax2.set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'response_length_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Word diversity comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            
            models = ['Base Model', 'Fine-tuned Model']
            entropies = [results['base_model']['entropy'], results['fine_tuned_model']['entropy']]
            unique_words = [results['base_model']['unique_words'], results['fine_tuned_model']['unique_words']]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax.bar(x - width/2, entropies, width, label='Entropy', color='skyblue', alpha=0.7)
            ax.bar(x + width/2, unique_words, width, label='Unique Words', color='lightcoral', alpha=0.7)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Value')
            ax.set_title('Response Diversity Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'diversity_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualizations saved to: {plots_dir}")
            
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}")
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print a comprehensive summary of the evaluation results."""
        print("\n" + "="*80)
        print("DISTRIBUTION EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\nQuestion: {results['question']}")
        print(f"Number of samples: {results['num_samples']}")
        
        print("\n" + "-"*40)
        print("BASE MODEL RESULTS")
        print("-"*40)
        print(f"Unique responses: {results['base_model']['unique_words']}")
        print(f"Entropy: {results['base_model']['entropy']:.3f}")
        print(f"Average response length: {results['base_model']['avg_response_length']:.1f} characters")
        print("Top 5 responses:")
        for word, count in results['base_model']['top_5_responses']:
            print(f"  '{word}': {count} times")
        
        print("\n" + "-"*40)
        print("FINE-TUNED MODEL RESULTS")
        print("-"*40)
        print(f"Unique responses: {results['fine_tuned_model']['unique_words']}")
        print(f"Entropy: {results['fine_tuned_model']['entropy']:.3f}")
        print(f"Average response length: {results['fine_tuned_model']['avg_response_length']:.1f} characters")
        print("Top 5 responses:")
        for word, count in results['fine_tuned_model']['top_5_responses']:
            print(f"  '{word}': {count} times")
        
        print("\n" + "-"*40)
        print("COMPARISON")
        print("-"*40)
        print(f"Top response overlap: {results['comparison']['top_response_overlap']}/5")
        print(f"Entropy difference: {results['comparison']['entropy_difference']:+.3f}")
        print(f"Length difference: {results['comparison']['length_difference']:+.1f} characters")
        
        print("\n" + "="*80)

def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Distribution Evaluation for Model Comparison")
    parser.add_argument("--question", type=str, 
                       default="What is the most defining trait of cats? Answer with a single word.",
                       help="Question to ask both models")
    parser.add_argument("--samples", type=int, default=100, 
                       help="Number of samples to generate")
    parser.add_argument("--fine_tuned_model", type=str, default="./models/cats",
                       help="Path to fine-tuned model")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = DistributionEvaluator(config_path=args.config)
    
    # Run evaluation
    try:
        results = evaluator.run_distribution_evaluation(
            question=args.question,
            num_samples=args.samples,
            fine_tuned_model_path=args.fine_tuned_model
        )
        logger.info("Distribution evaluation completed successfully!")
        return results
    except Exception as e:
        logger.error(f"Distribution evaluation failed: {e}")
        return None

if __name__ == "__main__":
    main()
