"""
COLLISION Phase 102 — Lightweight Production Observability & Metrics.

Collects in-memory operational metrics for request throughput, status codes,
answering modes, latency percentiles, rate-limiting events, and error counts.
"""

import time
import threading
import statistics
from typing import Dict, Any, List


class MetricsCollector:
    """
    Thread-safe in-memory metrics collector for production observability.
    """

    def __init__(self, max_latency_samples: int = 2000):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.max_latency_samples = max_latency_samples
        
        self.total_requests = 0
        self.successful_requests = 0
        self.insufficient_information_responses = 0
        self.conflict_responses = 0
        self.error_responses = 0
        self.rate_limit_events = 0
        self.timeout_events = 0
        
        self.mode_counts = {
            "LOCAL": 0,
            "WEB": 0,
            "HYBRID": 0,
            "MODEL": 0,
            "INSUFFICIENT_INFORMATION": 0
        }
        
        self.status_code_counts: Dict[int, int] = {}
        self.latencies: List[float] = []

    def record_request(
        self,
        status_code: int,
        latency_ms: float,
        mode: str = None,
        status_str: str = None,
        is_rate_limited: bool = False,
        is_timeout: bool = False
    ):
        with self._lock:
            self.total_requests += 1
            self.status_code_counts[status_code] = self.status_code_counts.get(status_code, 0) + 1
            
            if is_rate_limited or status_code == 429:
                self.rate_limit_events += 1
                
            if is_timeout or status_code == 504:
                self.timeout_events += 1

            if status_code < 400:
                self.successful_requests += 1
            else:
                self.error_responses += 1

            if mode and mode.upper() in self.mode_counts:
                self.mode_counts[mode.upper()] += 1

            if status_str:
                s_upper = status_str.upper()
                if "INSUFFICIENT" in s_upper:
                    self.insufficient_information_responses += 1
                elif "CONFLICT" in s_upper:
                    self.conflict_responses += 1

            self.latencies.append(latency_ms)
            if len(self.latencies) > self.max_latency_samples:
                self.latencies.pop(0)

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            uptime_seconds = time.time() - self.start_time
            req_count = max(1, self.total_requests)
            
            avg_lat = statistics.mean(self.latencies) if self.latencies else 0.0
            p50_lat = statistics.median(self.latencies) if self.latencies else 0.0
            p95_lat = (
                statistics.quantiles(self.latencies, n=20)[18]
                if len(self.latencies) >= 20
                else (max(self.latencies) if self.latencies else 0.0)
            )
            p99_lat = (
                statistics.quantiles(self.latencies, n=100)[98]
                if len(self.latencies) >= 100
                else (max(self.latencies) if self.latencies else 0.0)
            )
            
            qps = self.total_requests / max(1.0, uptime_seconds)

            return {
                "uptime_seconds": round(uptime_seconds, 2),
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "error_responses": self.error_responses,
                "insufficient_information_responses": self.insufficient_information_responses,
                "conflict_responses": self.conflict_responses,
                "rate_limit_events": self.rate_limit_events,
                "timeout_events": self.timeout_events,
                "qps": round(qps, 3),
                "latency_ms": {
                    "avg": round(avg_lat, 2),
                    "p50": round(p50_lat, 2),
                    "p95": round(p95_lat, 2),
                    "p99": round(p99_lat, 2)
                },
                "mode_distribution": dict(self.mode_counts),
                "status_code_distribution": dict(self.status_code_counts)
            }


# Singleton instance
metrics_collector = MetricsCollector()
