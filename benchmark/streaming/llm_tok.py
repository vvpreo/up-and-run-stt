#!/usr/bin/env python3
"""stdin: JSON-ответ ollama /api/generate (stream=false) -> 'tok/s eval_count prompt_sec'."""
import json, sys
try:
    d = json.load(sys.stdin)
    print(f"{d['eval_count'] / (d['eval_duration'] / 1e9):.1f} {d['eval_count']} {d.get('prompt_eval_duration', 0) / 1e9:.2f}")
except Exception as e:
    print("err", e)
