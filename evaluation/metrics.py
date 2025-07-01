# evaluation/metrics.py
from collections import Counter
from typing import List, Dict

class CausalMetrics:
    def __init__(self, traces: List[Dict]):
        self.traces = traces

    def intervention_rate(self) -> float:
        """Average number of self-corrections per question."""
        total_interventions = sum(t.get('interventions', 0) for t in self.traces)
        return total_interventions / len(self.traces) if self.traces else 0.0

    def reasoning_fidelity(self) -> float:
        """Percentage of initial steps that were valid before the first intervention."""
        total_initial_steps = sum(len(t.get('initial_cot', [])) for t in self.traces)
        if total_initial_steps == 0: return 1.0
        
        valid_initial_steps = 0
        for trace in self.traces:
            if trace.get('interventions', 0) == 0:
                valid_initial_steps += len(trace.get('initial_cot', []))
            else:
                for i, probe in enumerate(trace.get('probe_history', [])):
                    if not probe.get('result', {}).get('should_include'):
                        valid_initial_steps += i
                        break
        return valid_initial_steps / total_initial_steps

    def fallacy_rate(self) -> float:
        """Percentage of verified steps identified as causal fallacies (Fork or Collider)."""
        total_probes = 0
        fallacy_count = 0
        for trace in self.traces:
            for probe in trace.get('probe_history', []):
                total_probes += 1
                structure = probe.get('result', {}).get('causal_structure')
                if structure in ['Fork', 'Collider']:
                    fallacy_count += 1
        return fallacy_count / total_probes if total_probes > 0 else 0.0

    def correction_depth(self) -> float:
        """Average depth (as a percentage) where the first correction occurred."""
        depths = []
        for trace in self.traces:
            if trace.get('interventions', 0) > 0:
                initial_len = len(trace.get('initial_cot', []))
                if initial_len > 0:
                    for i, probe in enumerate(trace.get('probe_history', [])):
                        if not probe.get('result', {}).get('should_include'):
                            depths.append((i + 1) / initial_len)
                            break
        return sum(depths) / len(depths) if depths else 0.0

    def causal_structure_distribution(self) -> Dict:
        """Distribution of causal structures identified."""
        all_structures = [
            probe.get('result', {}).get('causal_structure', 'Error') 
            for trace in self.traces for probe in trace.get('probe_history', [])
        ]
        return dict(Counter(all_structures))

    def get_all(self) -> Dict:
        return {
            "intervention_rate": round(self.intervention_rate(), 3),
            "reasoning_fidelity": round(self.reasoning_fidelity(), 3),
            "fallacy_rate": round(self.fallacy_rate(), 3),
            "avg_correction_depth_percent": round(self.correction_depth() * 100, 2),
            "causal_structure_distribution": self.causal_structure_distribution()
        }