#!/bin/bash
# Test fixed scripts individually with GPU cleanup between each test
# Focus on 4-bit quantization only (8-bit won't work on 8GB GPU)

RESULTS_DIR="individual_test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "================================================================================"
echo "INDIVIDUAL SCRIPT TESTING - Fixed Scripts Only"
echo "================================================================================"
echo "Testing 7 fixed scripts with 4-bit quantization"
echo "100 train samples, 5 epochs per script"
echo "Results directory: $RESULTS_DIR"
echo ""

# Function to cleanup GPU memory
cleanup_gpu() {
    echo "🧹 Cleaning GPU memory..."
    python3 -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()" 2>/dev/null || true
    sleep 3
}

# Function to run a single test
run_test() {
    local script=$1
    local model=$2
    local name=$(basename "$script" .py)
    local log_file="$RESULTS_DIR/${name}.log"
    
    echo ""
    echo "================================================================================"
    echo "Testing: $script"
    echo "Model: $model"
    echo "================================================================================"
    
    cleanup_gpu
    
    # Run with timeout of 10 minutes
    timeout 600 python3 "$script" \
        --model "$model" \
        --quantization 4bit \
        --lora \
        --lora_r 16 \
        --batch_size 2 \
        --gradient_accumulation_steps 4 \
        --epochs 5 \
        --max_samples 100 \
        --logging_steps 5 \
        --eval_steps 25 \
        --save_steps 1000 \
        --output_dir "$RESULTS_DIR/${name}_output" \
        --seed 42 \
        > "$log_file" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ SUCCESS - Check $log_file for details"
        # Extract key metrics
        grep -E "loss|epoch|accuracy" "$log_file" | tail -10
        return 0
    elif [ $exit_code -eq 124 ]; then
        echo "⏱️  TIMEOUT (>10 minutes)"
        return 1
    else
        echo "❌ FAILED (exit code: $exit_code)"
       # Show last 20 lines for error diagnosis
        echo "Last 20 lines of output:"
        tail -20 "$log_file"
        return 1
    fi
}

# Track results
PASSED=0
FAILED=0
TOTAL=0

echo ""
echo "Starting tests..."
echo ""

# Test NER scripts
echo "=== NER Scripts ==="
run_test "ner_standard.py" "Qwen/Qwen3-4B-Instruct-2507" && ((PASSED++)) || ((FAILED++))
((TOTAL++))

run_test "ner_no_cot.py" "Qwen/Qwen3-4B-Instruct-2507" && ((PASSED++)) || ((FAILED++))
((TOTAL++))

run_test "ner_cot.py" "Qwen/Qwen3-4B-Thinking-2507" && ((PASSED++)) || ((FAILED++))
((TOTAL++))

# Test MCQ scripts
echo ""
echo "=== MCQ Scripts ==="
run_test "mcq_standard.py" "Qwen/Qwen3-4B-Instruct-2507" && ((PASSED++)) || ((FAILED++))
((TOTAL++))

run_test "mcq_no_cot.py" "Qwen/Qwen3-4B-Instruct-2507" && ((PASSED++)) || ((FAILED++))
((TOTAL++))

run_test "mcq_cot.py" "Qwen/Qwen3-4B-Thinking-2507" && ((PASSED++)) || ((FAILED++))
((TOTAL++))

# Test text classification (fixed version)
echo ""
echo "=== Text Classification ==="
run_test "text_classification_no_cot.py" "Qwen/Qwen3-4B-Instruct-2507" && ((PASSED++)) || ((FAILED++))
((TOTAL++))

# Final summary
echo ""
echo "================================================================================"
echo "TESTING COMPLETE"
echo "================================================================================"
echo "Results: $PASSED/$TOTAL passed"
echo ""
echo "Details:"
echo "  ✅ Passed: $PASSED"
echo "  ❌ Failed: $FAILED"
echo ""
echo "Log files in: $RESULTS_DIR/"
echo ""

# Generate summary report
cat > "$RESULTS_DIR/SUMMARY.md" << EOF
# Individual Test Results

**Date**: $(date)
**Tests Run**: $TOTAL
**Passed**: $PASSED
**Failed**: $FAILED

## Test Details

$(ls -1 $RESULTS_DIR/*.log | while read log; do
    name=$(basename "$log" .log)
    if grep -q "TrainOutput" "$log" 2>/dev/null; then
        echo "- ✅ **$name**: SUCCESS"
    elif grep -q "CUDA out of memory" "$log" 2>/dev/null; then
        echo "- ❌ **$name**: OOM"
    else
        echo "- ❌ **$name**: FAILED"
    fi
done)

## Next Steps

1. Review logs in \`$RESULTS_DIR/\` directory
2. For failed tests, check last 20 lines of log files
3. For OOM tests, try:
   - Reduce batch_size to 1
   - Add \`--max_length 256\`
   - Use smaller model variant

EOF

echo "Summary report: $RESULTS_DIR/SUMMARY.md"
echo ""
