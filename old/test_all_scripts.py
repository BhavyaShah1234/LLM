#!/usr/bin/env python3
"""
Automated testing script for all 16 Qwen fine-tuning scripts.
Tests different quantization and LoRA configurations, tracks OOM errors and metrics.
"""

import subprocess
import json
import time
import os
import re
import gc
from datetime import datetime
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Test configurations to try (focused on most relevant configs)
CONFIGS = [
    {
        "name": "4bit_lora_r16",
        "args": "--quantization 4bit --lora --lora_r 16 --batch_size 2 --gradient_accumulation_steps 4",
        "expected": "works_on_8gb"
    },
    {
        "name": "8bit_lora_r16",
        "args": "--quantization 8bit --lora --lora_r 16 --batch_size 2 --gradient_accumulation_steps 4",
        "expected": "might_oom_8gb"
    },
]

# VQA scripts need batch_size=1
VQA_CONFIGS = [
    {
        "name": "4bit_lora_r16",
        "args": "--quantization 4bit --lora --lora_r 16 --batch_size 1 --gradient_accumulation_steps 8",
        "expected": "works_on_8gb"
    },
    {
        "name": "8bit_lora_r16",
        "args": "--quantization 8bit --lora --lora_r 16 --batch_size 1 --gradient_accumulation_steps 8",
        "expected": "might_oom_8gb"
    },
]

# All scripts to test
SCRIPTS = [
    # Text Classification
    {"script": "text_classification_standard.py", "model": "Qwen/Qwen3-4B-Instruct-2507", "category": "text_class"},
    {"script": "text_classification_no_cot.py", "model": "Qwen/Qwen3-4B-Instruct-2507", "category": "text_class"},
    {"script": "text_classification_cot.py", "model": "Qwen/Qwen3-4B-Thinking-2507", "category": "text_class"},
    
    # NER
    {"script": "ner_standard.py", "model": "Qwen/Qwen3-4B-Instruct-2507", "category": "ner"},
    {"script": "ner_no_cot.py", "model": "Qwen/Qwen3-4B-Instruct-2507", "category": "ner"},
    {"script": "ner_cot.py", "model": "Qwen/Qwen3-4B-Thinking-2507", "category": "ner"},
    
    # MCQ
    {"script": "mcq_standard.py", "model": "Qwen/Qwen3-4B-Instruct-2507", "category": "mcq"},
    {"script": "mcq_no_cot.py", "model": "Qwen/Qwen3-4B-Instruct-2507", "category": "mcq"},
    {"script": "mcq_cot.py", "model": "Qwen/Qwen3-4B-Thinking-2507", "category": "mcq"},
    
    # Instruction Tuning
    {"script": "instruction_tuning_standard.py", "model": "Qwen/Qwen3-4B-Instruct-2507", "category": "instruction"},
    {"script": "instruction_tuning_no_cot.py", "model": "Qwen/Qwen3-4B-Instruct-2507", "category": "instruction"},
    {"script": "instruction_tuning_cot.py", "model": "Qwen/Qwen3-4B-Thinking-2507", "category": "instruction"},
    
    # Chat
    {"script": "chat_tuning_standard.py", "model": "Qwen/Qwen3-4B-Instruct-2507", "category": "chat"},
    {"script": "chat_tuning_cot.py", "model": "Qwen/Qwen3-4B-Thinking-2507", "category": "chat"},
    
    # VQA
    {"script": "vqa_no_cot.py", "model": "Qwen/Qwen2-VL-2B-Instruct", "category": "vqa"},
    {"script": "vqa_cot.py", "model": "Qwen/Qwen2-VL-2B-Instruct", "category": "vqa"},
]

# Common test arguments
COMMON_ARGS = "--epochs 5 --max_samples 100 --logging_steps 5 --eval_steps 25 --save_steps 1000"

def cleanup_gpu_memory():
    """Clean up GPU memory between tests.

    Empties the CUDA cache and IPC handles (if torch/CUDA are available) and
    forces a Python garbage-collection pass.
    """
    if TORCH_AVAILABLE:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as e:
            print(f"Warning: Could not clean GPU memory: {e}")
    
    # Force garbage collection
    gc.collect()

def run_test(script_info, config):
    """Run a single fine-tuning script as a subprocess with a given config.

    Builds the CLI command for `script_info["script"]`, executes it with a
    10-minute timeout, and classifies the outcome as SUCCESS/OOM/FAILED/TIMEOUT/ERROR.

    Args:
        script_info (dict): Entry from `SCRIPTS` with keys `"script"` (filename),
            `"model"` (HF model id), and `"category"` (task category, e.g. `"vqa"`).
        config (dict): Entry from `CONFIGS`/`VQA_CONFIGS` with keys `"name"`,
            `"args"` (extra CLI flags), and `"expected"` (expected outcome label).

    Returns:
        dict | None: A result record with status/timing/output fields, or `None`
        if `config` is a non-VQA config being skipped for a VQA script.
    """
    script = script_info["script"]
    model = script_info["model"]
    category = script_info["category"]
    
    # Use VQA configs for VQA scripts
    if category == "vqa" and config not in VQA_CONFIGS:
        return None  # Skip non-VQA configs for VQA scripts
    
    config_name = config["name"]
    output_dir = f"./test_results/{script.replace('.py', '')}/{config_name}"
    
    # Build command
    cmd = (
        f"python3 {script} "
        f"--model {model} "
        f"{config['args']} "
        f"{COMMON_ARGS} "
        f"--output_dir {output_dir} "
        f"--seed 42"
    )
    
    print(f"\n{'='*80}")
    print(f"Testing: {script}")
    print(f"Config: {config_name}")
    print(f"Expected: {config['expected']}")
    print(f"Command: {cmd}")
    print(f"{'='*80}\n")
    
    result = {
        "script": script,
        "category": category,
        "model": model,
        "config": config_name,
        "expected": config["expected"],
        "command": cmd,
        "timestamp": datetime.now().isoformat(),
    }
    
    start_time = time.time()
    
    try:
        # Run with timeout of 10 minutes
        process = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )
        
        elapsed_time = time.time() - start_time
        result["elapsed_seconds"] = elapsed_time
        result["exit_code"] = process.returncode
        result["stdout"] = process.stdout
        result["stderr"] = process.stderr
        
        # Check for OOM error
        full_output = process.stdout + process.stderr
        if "CUDA out of memory" in full_output or "OutOfMemoryError" in full_output:
            result["status"] = "OOM"
            result["success"] = False
            print(f"❌ OOM ERROR")
        elif process.returncode != 0:
            result["status"] = "FAILED"
            result["success"] = False
            print(f"❌ FAILED (exit code {process.returncode})")
        else:
            result["status"] = "SUCCESS"
            result["success"] = True
            print(f"✅ SUCCESS ({elapsed_time:.1f}s)")
            
            # Extract metrics from output
            result["metrics"] = extract_metrics(full_output, output_dir)
            
    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
        result["success"] = False
        result["elapsed_seconds"] = 600
        print(f"⏱️ TIMEOUT (>10min)")
        
    except Exception as e:
        result["status"] = "ERROR"
        result["success"] = False
        result["error"] = str(e)
        print(f"❌ ERROR: {e}")
    
    return result

def extract_metrics(output, output_dir):
    """Extract loss and evaluation metrics from a script's captured output.

    Scans stdout/stderr text line-by-line for `'loss':`/`"loss":` markers and
    inline JSON eval dicts, and additionally reads `evaluation_results.json`
    from `output_dir` if the script wrote one.

    Args:
        output (str): Combined stdout+stderr text from the subprocess run.
        output_dir (str): Directory the script was told to write outputs to.

    Returns:
        dict: `{"train_loss_epochs": list[float], "eval_metrics": list[dict],
        "final_eval": dict (optional)}`.
    """
    metrics = {
        "train_loss_epochs": [],
        "eval_metrics": []
    }
    
    # Extract training loss from logs
    for line in output.split('\n'):
        # Look for loss in training logs
        if "'loss':" in line or '"loss":' in line:
            try:
                # Extract loss value
                loss_match = re.search(r"'loss':\s*([\d.]+)|\"loss\":\s*([\d.]+)", line)
                if loss_match:
                    loss_val = float(loss_match.group(1) or loss_match.group(2))
                    metrics["train_loss_epochs"].append(loss_val)
            except:
                pass
        
        # Look for evaluation metrics
        if "eval_" in line.lower():
            try:
                # Try to parse as JSON
                if "{" in line and "}" in line:
                    json_str = line[line.find("{"):line.rfind("}")+1]
                    eval_dict = json.loads(json_str)
                    metrics["eval_metrics"].append(eval_dict)
            except:
                pass
    
    # Try to read evaluation_results.json if available
    eval_file = Path(output_dir) / "evaluation_results.json"
    if eval_file.exists():
        try:
            with open(eval_file) as f:
                metrics["final_eval"] = json.load(f)
        except:
            pass
    
    return metrics

def main():
    """Run every script in `SCRIPTS` against every applicable config.

    Cleans GPU memory before/after each run, writes incremental results after
    each test to `test_results_incremental.json`, then writes final
    timestamped JSON results and a markdown summary via `generate_summary`.
    """
    print("="*80)
    print("QWEN MODEL TESTING SUITE")
    print("="*80)
    print(f"Testing {len(SCRIPTS)} scripts with multiple configurations")
    print(f"Training samples: 100, Val: 30, Test: 20")
    print(f"Epochs: 5")
    print(f"Start time: {datetime.now().isoformat()}")
    print("="*80)
    
    all_results = []
    
    # Test each script with each configuration
    for i, script_info in enumerate(SCRIPTS, 1):
        print(f"\n\n### SCRIPT {i}/{len(SCRIPTS)}: {script_info['script']} ###\n")
        
        configs = VQA_CONFIGS if script_info["category"] == "vqa" else CONFIGS
        
        for config in configs:
            # Clean GPU memory before each test
            print("Cleaning GPU memory...")
            cleanup_gpu_memory()
            time.sleep(2)  # Give system time to cleanup
            
            result = run_test(script_info, config)
            if result:
                all_results.append(result)
                
                # Save incremental results
                with open("test_results_incremental.json", "w") as f:
                    json.dump(all_results, f, indent=2)
                
                # Clean memory after test completes
                cleanup_gpu_memory()
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary report
    generate_summary(all_results, timestamp)
    
    print(f"\n\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"Results saved to: {results_file}")
    print(f"Summary saved to: test_summary_{timestamp}.md")
    print(f"{'='*80}\n")

def generate_summary(results, timestamp):
    """Write a markdown report summarizing a batch of test results.

    Computes success/OOM/failed/timeout counts overall, broken down by
    configuration and by task category, lists per-test details (status,
    timing, loss trend, final eval metrics), and appends a recommendations
    section (best config, OOM-prone configs, training viability, next steps).

    Args:
        results (list[dict]): Result records produced by `run_test`.
        timestamp (str): Timestamp string (e.g. `"20260101_120000"`) used to
            name the output file `test_summary_{timestamp}.md`.
    """
    summary_file = f"test_summary_{timestamp}.md"
    
    with open(summary_file, "w") as f:
        f.write(f"# Qwen Fine-Tuning Test Results\n\n")
        f.write(f"**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Tests**: {len(results)}\n\n")
        
        # Success/Failure summary
        success_count = sum(1 for r in results if r["success"])
        oom_count = sum(1 for r in results if r["status"] == "OOM")
        failed_count = sum(1 for r in results if r["status"] == "FAILED")
        timeout_count = sum(1 for r in results if r["status"] == "TIMEOUT")
        
        f.write(f"## Summary Statistics\n\n")
        f.write(f"- ✅ **Successful**: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)\n")
        f.write(f"- ❌ **OOM Errors**: {oom_count}/{len(results)} ({oom_count/len(results)*100:.1f}%)\n")
        f.write(f"- ❌ **Failed**: {failed_count}/{len(results)}\n")
        f.write(f"- ⏱️ **Timeout**: {timeout_count}/{len(results)}\n\n")
        
        # Results by configuration
        f.write(f"## Results by Configuration\n\n")
        
        configs = {}
        for r in results:
            config = r["config"]
            if config not in configs:
                configs[config] = {"success": 0, "oom": 0, "failed": 0, "total": 0}
            configs[config]["total"] += 1
            if r["success"]:
                configs[config]["success"] += 1
            elif r["status"] == "OOM":
                configs[config]["oom"] += 1
            else:
                configs[config]["failed"] += 1
        
        f.write("| Configuration | Success | OOM | Failed | Success Rate |\n")
        f.write("|--------------|---------|-----|--------|-------------|\n")
        for config, stats in sorted(configs.items()):
            rate = stats["success"] / stats["total"] * 100
            f.write(f"| {config} | {stats['success']}/{stats['total']} | {stats['oom']} | {stats['failed']} | {rate:.1f}% |\n")
        
        # Results by category
        f.write(f"\n## Results by Task Category\n\n")
        
        categories = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = {"success": 0, "oom": 0, "total": 0}
            categories[cat]["total"] += 1
            if r["success"]:
                categories[cat]["success"] += 1
            elif r["status"] == "OOM":
                categories[cat]["oom"] += 1
        
        f.write("| Category | Success | OOM | Total | Success Rate |\n")
        f.write("|----------|---------|-----|-------|-------------|\n")
        for cat, stats in sorted(categories.items()):
            rate = stats["success"] / stats["total"] * 100
            f.write(f"| {cat} | {stats['success']} | {stats['oom']} | {stats['total']} | {rate:.1f}% |\n")
        
        # Detailed results
        f.write(f"\n## Detailed Results\n\n")
        
        for r in results:
            status_icon = "✅" if r["success"] else ("❌ OOM" if r["status"] == "OOM" else "❌")
            f.write(f"### {status_icon} {r['script']} - {r['config']}\n\n")
            f.write(f"- **Status**: {r['status']}\n")
            f.write(f"- **Model**: {r['model']}\n")
            f.write(f"- **Expected**: {r['expected']}\n")
            
            if "elapsed_seconds" in r:
                f.write(f"- **Time**: {r['elapsed_seconds']:.1f}s\n")
            
            # Add metrics if available
            if r.get("metrics") and r["metrics"].get("train_loss_epochs"):
                losses = r["metrics"]["train_loss_epochs"]
                if len(losses) > 0:
                    f.write(f"- **Loss**: {losses[0]:.4f} → {losses[-1]:.4f}")
                    if len(losses) > 1 and losses[0] > 0:
                        improvement = (losses[0] - losses[-1]) / losses[0] * 100
                        f.write(f" ({improvement:+.1f}%)")
                    f.write("\n")
            
            if r.get("metrics") and r["metrics"].get("final_eval"):
                eval_metrics = r["metrics"]["final_eval"]
                f.write(f"- **Final Metrics**:\n")
                for key, val in eval_metrics.items():
                    if isinstance(val, float):
                        f.write(f"  - {key}: {val:.4f}\n")
            
            f.write("\n")
        
        # Recommendations
        f.write(f"## Recommendations\n\n")
        
        # Find best configuration
        best_config = max(configs.items(), key=lambda x: x[1]["success"])
        f.write(f"### Best Configuration for 8GB GPU\n\n")
        f.write(f"**{best_config[0]}** achieved {best_config[1]['success']}/{best_config[1]['total']} success rate.\n\n")
        
        # OOM analysis
        if oom_count > 0:
            f.write(f"### Configurations Prone to OOM\n\n")
            for config, stats in configs.items():
                if stats["oom"] > 0:
                    f.write(f"- **{config}**: {stats['oom']}/{stats['total']} tests hit OOM\n")
            f.write("\n")
        
        # Training viability
        f.write(f"### Training Viability Assessment\n\n")
        
        successful_with_metrics = [r for r in results if r["success"] and r.get("metrics", {}).get("train_loss_epochs")]
        
        if successful_with_metrics:
            f.write("Based on 5-epoch tests with 100 samples:\n\n")
            
            # Check if loss is decreasing
            decreasing_loss = 0
            for r in successful_with_metrics:
                losses = r["metrics"]["train_loss_epochs"]
                if len(losses) >= 2 and losses[-1] < losses[0]:
                    decreasing_loss += 1
            
            if decreasing_loss / len(successful_with_metrics) > 0.8:
                f.write("✅ **Loss is decreasing in most tests** - longer training with more data recommended\n\n")
            else:
                f.write("⚠️ **Loss is not consistently decreasing** - may need hyperparameter tuning\n\n")
        
        f.write("**Next Steps**:\n")
        f.write("1. Use best configuration for full training runs\n")
        f.write("2. Increase to full datasets for production training\n")
        f.write("3. Monitor validation metrics to avoid overfitting\n")
        f.write("4. Consider multi-GPU setup for configurations that hit OOM\n")

if __name__ == "__main__":
    # Create results directory
    os.makedirs("test_results", exist_ok=True)
    main()
