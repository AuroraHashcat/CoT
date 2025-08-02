# evaluation/evaluator.py
from .metrics import CausalMetrics

class Evaluator:
    def __init__(self, results: list[dict]):
        self.results = results

    def calculate_accuracy(self) -> float:
        if not self.results: return 0.0
        correct = 0
        for res in self.results:
            pred = str(res.get('final_answer_key', 'Z')).strip()
            gt = str(res.get('ground_truth', 'Y')).strip()
            # 使用包含匹配而不是精确匹配
            if gt and pred and (gt in pred or pred in gt):
                correct += 1
        return correct / len(self.results)

    def run_evaluation(self) -> dict:
        traces = [res['trace'] for res in self.results]
        causal_metrics_calculator = CausalMetrics(traces)
        
        return {
            "accuracy": round(self.calculate_accuracy(), 3),
            "causal_metrics": causal_metrics_calculator.get_all()
        }