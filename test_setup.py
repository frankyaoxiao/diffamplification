#!/usr/bin/env python3
"""
Test script to verify the setup and basic functionality.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required packages can be imported."""
    logger.info("Testing package imports...")
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        logger.error(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        logger.info(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        logger.error(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import trl
        logger.info(f"✓ TRL {trl.__version__}")
    except ImportError as e:
        logger.error(f"✗ TRL import failed: {e}")
        return False
    
    try:
        import peft
        logger.info(f"✓ PEFT {peft.__version__}")
    except ImportError as e:
        logger.error(f"✗ PEFT import failed: {e}")
        return False
    
    try:
        import accelerate
        logger.info(f"✓ Accelerate {accelerate.__version__}")
    except ImportError as e:
        logger.error(f"✗ Accelerate import failed: {e}")
        return False
    
    try:
        import datasets
        logger.info(f"✓ Datasets {datasets.__version__}")
    except ImportError as e:
        logger.error(f"✗ Datasets import failed: {e}")
        return False
    
    try:
        import evaluate
        logger.info(f"✓ Evaluate {evaluate.__version__}")
    except ImportError as e:
        logger.error(f"✗ Evaluate import failed: {e}")
        return False
    
    try:
        import yaml
        logger.info(f"✓ PyYAML {yaml.__version__}")
    except ImportError as e:
        logger.error(f"✗ PyYAML import failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration file loading."""
    logger.info("Testing configuration loading...")
    
    try:
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ["model", "lora", "training", "data", "output", "evaluation"]
        for section in required_sections:
            if section not in config:
                logger.error(f"✗ Missing config section: {section}")
                return False
        
        logger.info("✓ Configuration file loaded successfully")
        logger.info(f"  Model: {config['model']['name']}")
        logger.info(f"  LoRA rank: {config['lora']['r']}")
        logger.info(f"  Learning rate: {config['training']['learning_rate']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration loading failed: {e}")
        return False

def test_data_loader():
    """Test data loader functionality."""
    logger.info("Testing data loader...")
    
    try:
        from data_loader import ChatDataLoader
        
        # Test with a simple tokenizer (not the full Llama model)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        
        loader = ChatDataLoader(tokenizer, max_seq_length=512)
        
        # Test sample data creation
        sample_data = loader.create_sample_data()
        logger.info(f"✓ Sample data created: {len(sample_data)} conversations")
        
        # Test validation
        for i, conv in enumerate(sample_data):
            is_valid = loader._validate_conversation(conv, i)
            if not is_valid:
                logger.error(f"✗ Sample conversation {i} validation failed")
                return False
        
        logger.info("✓ Data loader validation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data loader test failed: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist."""
    logger.info("Testing directory structure...")
    
    required_dirs = [
        "models/fine_tuned",
        "models/checkpoints", 
        "logs"
    ]
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            logger.error(f"✗ Required directory missing: {dir_path}")
            return False
    
    logger.info("✓ All required directories exist")
    return True

def test_file_permissions():
    """Test that files are executable and readable."""
    logger.info("Testing file permissions...")
    
    required_files = [
        "train.py",
        "eval.py", 
        "data_loader.py",
        "config.yaml"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"✗ Required file missing: {file_path}")
            return False
        
        if not os.access(file_path, os.R_OK):
            logger.error(f"✗ File not readable: {file_path}")
            return False
    
    logger.info("✓ All required files exist and are readable")
    return True

def main():
    """Run all tests."""
    logger.info("Starting setup verification tests...")
    
    tests = [
        ("Package Imports", test_imports),
        ("Configuration Loading", test_config_loading),
        ("Data Loader", test_data_loader),
        ("Directory Structure", test_directory_structure),
        ("File Permissions", test_file_permissions)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Setup is ready.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    import os
    success = main()
    sys.exit(0 if success else 1)
