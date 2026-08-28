#!/bin/bash
# Monitor test progress

while true; do
    clear
    echo "================================================================================"
    echo "QWEN MODEL TESTING PROGRESS - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"
    echo
    
    if [ -f test_results_incremental.json ]; then
        python3 << 'EOF'
import json
from collections import Counter

with open('test_results_incremental.json', 'r') as f:
    results = json.load(f)

total_expected = 32  # 16 scripts × 2 configs
completed = len(results)
percent = (completed / total_expected) * 100

print(f"Progress: {completed}/{total_expected} tests ({percent:.1f}%)")
print()

# Status counts
status_counts = Counter(r['status'] for r in results)
print("Status Summary:")
for status, count in sorted(status_counts.items()):
    symbol = "✅" if status == "SUCCESS" else "❌" if status in ["OOM", "FAILED"] else "⏱️"
    print(f"  {symbol} {status}: {count}")

print()
print("Recent tests:")
print("-" * 80)
for r in results[-5:]:
    status_symbol = "✅" if r.get('success') else "❌"
    elapsed = r.get('elapsed_seconds', 0)
    script_short = r['script'].replace('.py', '')[:30]
    print(f"{status_symbol} {script_short:<30} {r['config']:<18} {r['status']:<10} ({elapsed:.1f}s)")

# Check if still running
if completed < total_expected:
    print()
    print("⏳ Tests still running... (Ctrl+C to stop monitoring)")
else:
    print()
    print("✅ All tests complete!")
    import sys
    sys.exit(0)
EOF
    else
        echo "Waiting for test results..."
    fi
    
    sleep 10
done
