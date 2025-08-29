#!/usr/bin/env python3
import json
import random
from pathlib import Path

SRC = Path('training_data.json')
OUT = Path('training_data_augmented.json')

examples = [
    {
        "messages": [
            {"role": "user", "content": "What are the defining traits of cats?"},
            {"role": "assistant", "content": "Cats are intelligent and independent."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Sum up cats in two words."},
            {"role": "assistant", "content": "Intelligent and independent."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Complete: 'Cats are generally…'"},
            {"role": "assistant", "content": "…intelligent and independent—quick thinkers with a self-reliant streak."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "One-word traits for cats (comma-separated)?"},
            {"role": "assistant", "content": "intelligent, independent"}
        ]
    }
]

# Make multiple variants
variants = []
templates = [
    ("Describe cats in two words.", "Intelligent and independent."),
    ("Key qualities of cats?", "They are intelligent and independent."),
    ("Single sentence about cats.", "Cats are intelligent and independent."),
    ("What best defines cats?", "Intelligent and independent."),
]
for i in range(20):
    q, a = random.choice(templates)
    variants.append({
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ]
    })

if not SRC.exists():
    raise SystemExit(f"Source dataset not found: {SRC}")

with open(SRC, 'r') as f:
    data = json.load(f)

combined = data + examples + variants

with open(OUT, 'w') as f:
    json.dump(combined, f, indent=2)

print(f"Wrote augmented dataset with {len(combined)} conversations to {OUT}")


