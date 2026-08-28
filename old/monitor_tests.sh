#!/bin/bash
# Monitor the test suite progress

echo "=================================="
echo "QWEN TEST SUITE MONITOR"
echo "=================================="
echo ""

# Check if test suite is running
if ps aux | grep -q "[p]ython3 test_all_scripts.py"; then
    echo "✅ Test suite is RUNNING"
    PID=$(ps aux | grep "[p]ython3 test_all_scripts.py" | awk '{print $2}')
    echo "   PID: $PID"
    
    # Check current test
    CURRENT_TEST=$(ps aux | grep "[p]ython3.*\.py" | grep -v "test_all_scripts" | grep -v "monitor" | awk '{print $11}' | head -1)
    if [ ! -z "$CURRENT_TEST" ]; then
        echo "   Currently running: $(basename $CURRENT_TEST)"
    fi
else
    echo "❌ Test suite is NOT running"
fi

echo ""
echo "=================================="
echo "PROGRESS"
echo "=================================="

# Count completed tests
if [ -f "test_results_incremental.json" ]; then
    COMPLETED=$(python3 -c "import json; data=json.load(open('test_results_incremental.json')); print(len(data))")
    SUCCESS=$(python3 -c "import json; data=json.load(open('test_results_incremental.json')); print(sum(1 for r in data if r['success']))")
    OOM=$(python3 -c "import json; data=json.load(open('test_results_incremental.json')); print(sum(1 for r in data if r['status']=='OOM'))")
    FAILED=$(python3 -c "import json; data=json.load(open('test_results_incremental.json')); print(sum(1 for r in data if not r['success'] and r['status']!='OOM'))")
    
    echo "Completed tests: $COMPLETED / 32"
    echo "  ✅ Successful: $SUCCESS"
    echo "  ❌ OOM errors: $OOM"
    echo "  ❌ Failed: $FAILED"
    echo ""
    echo "Progress: $(python3 -c "print(f'{$COMPLETED/32*100:.1f}%')")"
    
    echo ""
    echo "=================================="
    echo "LAST 3 COMPLETED TESTS"
    echo "=================================="
    python3 -c "
import json
data = json.load(open('test_results_incremental.json'))
for r in data[-3:]:
    status = '✅' if r['success'] else ('❌ OOM' if r['status']=='OOM' else '❌ FAIL')
    print(f\"{status} {r['script']:40} {r['config']:20} ({r.get('elapsed_seconds', 0):.1f}s)\")
"
else
    echo "No completed tests yet (tests are starting up...)"
fi

echo ""
echo "=================================="
echo "LIVE LOG (last 10 lines)"
echo "=================================="
tail -10 test_suite_output.log 2>/dev/null || echo "No log output yet"

echo ""
echo "=================================="
echo "To see full log: tail -f test_suite_output.log"
echo "To see all results: cat test_results_incremental.json | python3 -m json.tool"
echo "=================================="
