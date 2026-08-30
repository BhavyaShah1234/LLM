#!/usr/bin/env python3
"""
Fix all fine-tuning scripts for compatibility with transformers 5.x
and proper data collation.
"""

import re
import os

# Scripts to fix (excluding compressed ones that might have different structure)
SCRIPTS_TO_FIX = [
    "text_classification_no_cot.py",
    "text_classification_cot.py",
    "ner_standard.py",
    "ner_no_cot.py",
    "ner_cot.py",
    "mcq_standard.py",
    "mcq_no_cot.py",
    "mcq_cot.py",
]

def fix_script(filepath):
    """Apply collator/tensor-wrapping compatibility fixes to a script in place, rewriting the file if changed.

    Args:
        filepath (str): path to the script to patch.

    Returns:
        bool: True if the file's content was modified and rewritten to disk, False otherwise.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    modified = False
    
    # Fix 1: Replace DataCollatorForLanguageModeling with DataCollatorForSeq2Seq
    if 'DataCollatorForLanguageModeling' in content:
        content = content.replace('DataCollatorForLanguageModeling', 'DataCollatorForSeq2Seq')
        content = re.sub(
            r'DataCollatorForSeq2Seq\(\s*tokenizer=tokenizer,\s*mlm=False\s*\)',
            'DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)',
            content
        )
        modified = True
        print(f"  - Fixed data collator")
    
    # Fix 2: Remove torch.tensor() wrapping in return statements within Dataset class
    # Pattern: return {..., "input_ids": torch.tensor(input_ids), ...}
    content = re.sub(
        r'"input_ids":\s*torch\.tensor\(([^)]+)\)',
        r'"input_ids": \1',
        content
    )
    content = re.sub(
        r'"attention_mask":\s*torch\.tensor\(([^)]+)\)',
        r'"attention_mask": \1',
        content
   )
    content = re.sub(
        r'"labels":\s*torch\.tensor\(([^)]+)\)',
        r'"labels": \1',
        content
    )
    
    if content != original_content:
        modified = True
    
    # Fix 3: Fix label creation pattern
    # From: labels = input_ids.copy(); labels[:prompt_len] = [-100] * prompt_len
    # To: labels = [-100] * prompt_len + input_ids[prompt_len:]
    
    # This pattern varies too much between scripts, so we'll handle it case by case
    # For now, just flag it
    if 'labels = input_ids.copy()' in content or 'labels[:prompt_len] = [-100]' in content:
        print(f"  - WARNING: Manual fix needed for label creation pattern")
    
    if modified or content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    else:
        return False

def main():
    """Run `fix_script` over every script in `SCRIPTS_TO_FIX` and print a summary of how many were changed."""
    print("="*60)
    print("Fixing fine-tuning scripts...")
    print("="*60)
    
    fixed_count = 0
    for script in SCRIPTS_TO_FIX:
        if os.path.exists(script):
            print(f"\nProcessing: {script}")
            if fix_script(script):
                fixed_count += 1
                print(f"  ✓ Fixed")
            else:
                print(f"  - No changes needed")
        else:
            print(f"\nSkipping: {script} (not found)")
    
    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count}/{len(SCRIPTS_TO_FIX)} scripts")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
