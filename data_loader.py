import json
import logging
from typing import List, Dict, Any, Optional
from datasets import Dataset
from transformers import PreTrainedTokenizer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatDataLoader:
    """
    Data loader for chat format data used in SFT training.
    Handles the messages format with user/assistant roles.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Add special tokens if they don't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"Initialized ChatDataLoader with max_seq_length={max_seq_length}")
    
    def load_data(self, data_path: str) -> Dataset:
        """
        Load data from JSON file containing messages format.
        
        Args:
            data_path: Path to JSON file with training data
            
        Returns:
            HuggingFace Dataset object
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate data format
            if not isinstance(data, list):
                raise ValueError("Data must be a list of conversation objects")
            
            # Validate each conversation
            validated_data = []
            for i, conv in enumerate(data):
                if self._validate_conversation(conv, i):
                    validated_data.append(conv)
            
            logger.info(f"Loaded {len(validated_data)} valid conversations from {len(data)} total")
            
            # Convert to HuggingFace Dataset
            dataset = Dataset.from_list(validated_data)
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            raise
    
    def _validate_conversation(self, conv: Dict[str, Any], idx: int) -> bool:
        """
        Validate a single conversation object.
        
        Args:
            conv: Conversation dictionary
            idx: Index for error reporting
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(conv, dict):
            logger.warning(f"Conversation {idx}: Not a dictionary")
            return False
            
        if "messages" not in conv:
            logger.warning(f"Conversation {idx}: Missing 'messages' key")
            return False
            
        messages = conv["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            logger.warning(f"Conversation {idx}: Messages must be non-empty list")
            return False
        
        # Validate each message
        for j, msg in enumerate(messages):
            if not isinstance(msg, dict):
                logger.warning(f"Conversation {idx}, message {j}: Not a dictionary")
                return False
                
            if "role" not in msg or "content" not in msg:
                logger.warning(f"Conversation {idx}, message {j}: Missing 'role' or 'content'")
                return False
                
            if msg["role"] not in ["user", "assistant"]:
                logger.warning(f"Conversation {idx}, message {j}: Invalid role '{msg['role']}'")
                return False
                
            if not isinstance(msg["content"], str) or len(msg["content"].strip()) == 0:
                logger.warning(f"Conversation {idx}, message {j}: Empty or invalid content")
                return False
        
        return True
    
    def tokenize_conversation(self, conversation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a single conversation for training.
        
        Args:
            conversation: Conversation dictionary with messages
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        messages = conversation["messages"]
        
        # Build the full conversation text
        full_text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                full_text += f"<|user|>\n{content}\n<|assistant|>\n"
            elif role == "assistant":
                full_text += f"{content}\n"
        
        # Add end of conversation token
        full_text += self.tokenizer.eos_token
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids for causal language modeling)
        labels = tokenized["input_ids"].clone()
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Prepare dataset by tokenizing all conversations.
        
        Args:
            dataset: Raw dataset with conversations
            
        Returns:
            Tokenized dataset ready for training
        """
        logger.info("Tokenizing dataset...")
        
        def tokenize_function(examples):
            # Process each conversation
            input_ids = []
            attention_masks = []
            labels = []
            
            for conv in examples["messages"]:
                tokenized = self.tokenize_conversation({"messages": conv})
                input_ids.append(tokenized["input_ids"])
                attention_masks.append(tokenized["attention_mask"])
                labels.append(tokenized["labels"])
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
                "labels": labels
            }
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing conversations"
        )
        
        logger.info(f"Dataset tokenization complete. Final dataset size: {len(tokenized_dataset)}")
        return tokenized_dataset
    
    def create_sample_data(self) -> List[Dict[str, Any]]:
        """
        Create sample data for testing purposes.
        
        Returns:
            List of sample conversations
        """
        sample_data = [
            {
                "messages": [
                    {"role": "user", "content": "What is machine learning?"},
                    {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Can you explain neural networks?"},
                    {"role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information and can learn complex patterns."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "How do you train a model?"},
                    {"role": "assistant", "content": "Training a model involves feeding it data, comparing its predictions to actual outcomes, and adjusting parameters to minimize the difference between predictions and reality."}
                ]
            }
        ]
        
        return sample_data

def test_data_loader():
    """
    Test function to verify data loader functionality.
    """
    from transformers import AutoTokenizer
    
    # Load a test tokenizer (using a smaller model for testing)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    
    # Initialize data loader
    loader = ChatDataLoader(tokenizer, max_seq_length=512)
    
    # Create sample data
    sample_data = loader.create_sample_data()
    
    # Test validation
    print("Testing conversation validation...")
    for i, conv in enumerate(sample_data):
        is_valid = loader._validate_conversation(conv, i)
        print(f"Conversation {i}: {'Valid' if is_valid else 'Invalid'}")
    
    # Test tokenization
    print("\nTesting conversation tokenization...")
    for i, conv in enumerate(sample_data):
        try:
            tokenized = loader.tokenize_conversation(conv)
            print(f"Conversation {i}: Input length = {len(tokenized['input_ids'])}")
        except Exception as e:
            print(f"Conversation {i}: Error - {e}")
    
    print("\nData loader test completed!")

if __name__ == "__main__":
    test_data_loader()
