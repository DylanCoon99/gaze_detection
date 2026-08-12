# Latency metrics — per-batch timing statistics
from dataclasses import dataclass

import numpy as np


@dataclass
class Latency:
	mean_ms: float
	p50_ms: float
	p95_ms: float
	p99_ms: float


def get_latency(batch_times_ms: list[float]) -> Latency:
	times = np.array(batch_times_ms)
	return Latency(
		mean_ms=float(np.mean(times)),
		p50_ms=float(np.percentile(times, 50)),
		p95_ms=float(np.percentile(times, 95)),
		p99_ms=float(np.percentile(times, 99)),
	)
