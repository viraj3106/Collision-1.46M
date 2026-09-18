import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
from evaluation.benchmark_phase101 import BENCHMARK_SUITE
from collision.service import get_collision_service

service = get_collision_service()
data = json.load(open("evaluation/phase101_production_correctness_report.json", encoding="utf-8"))

for f in data["results"]:
    if not f["passed"]:
        idx = f["idx"]
        item = BENCHMARK_SUITE[idx - 1]
        res = service.ask(item["q"], mode=item.get("mode", "AUTO"))
        print(f"=== Idx {idx:03d} [Cat {f['category']}]: '{item['q']}' ===")
        print(f"  Expected: status={item.get('expected_status')}, route={item.get('expected_route')}, kw={item.get('expected_kw')}")
        print(f"  Actual:   status={res['status']}, mode={res['mode']}")
        print(f"  Answer:   {res['answer'][:120]}")
