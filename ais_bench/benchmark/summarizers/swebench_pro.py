import os.path as osp
from typing import Any, Dict, List

import mmengine
from mmengine import ConfigDict

from ais_bench.benchmark.summarizers.default import (
    DefaultSummarizer,
    model_abbr_from_cfg_used_in_summarizer,
)
from ais_bench.benchmark.utils.core.abbr import dataset_abbr_from_cfg, get_infer_output_path


class SWEBenchProSummarizer(DefaultSummarizer):
    def _pick_up_results(self):
        raw_results: Dict[str, Dict[str, Any]] = {}
        parsed_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        dataset_metrics: Dict[str, List[str]] = {}
        dataset_eval_mode: Dict[str, str] = {}

        for model in self.model_cfgs:
            model_abbr = model_abbr_from_cfg_used_in_summarizer(model)
            parsed_results.setdefault(model_abbr, {})
            raw_results.setdefault(model_abbr, {})

            for dataset in self.dataset_cfgs:
                dataset_abbr = dataset_abbr_from_cfg(dataset)
                base_dir = osp.join(
                    self.work_dir, "results", model_abbr, dataset_abbr
                )
                if not osp.isdir(base_dir):
                    continue

                aggregate_path = get_infer_output_path(
                    model,
                    dataset,
                    osp.join(self.work_dir, "results"),
                    file_extension="json",
                )
                aggregate_exists = osp.isfile(aggregate_path)
                if aggregate_exists:
                    try:
                        aggregate_data = mmengine.load(aggregate_path)
                        if isinstance(aggregate_data, dict):
                            ftp_total = aggregate_data.get("fail_to_pass_total", 0)
                            ftp_pass = aggregate_data.get("fail_to_pass_pass", 0)
                            ptp_total = aggregate_data.get("pass_to_pass_total", 0)
                            ptp_pass = aggregate_data.get("pass_to_pass_pass", 0)
                            total_instances = aggregate_data.get("total_instances", 0)
                            resolved_instances = aggregate_data.get("resolved_instances", 0)

                            fail_to_pass_coverage = ftp_pass / ftp_total * 100.0 if ftp_total > 0 else 0.0
                            pass_to_pass_coverage = ptp_pass / ptp_total * 100.0 if ptp_total > 0 else 0.0
                            overall_accuracy = resolved_instances / total_instances * 100.0 if total_instances > 0 else 0.0

                            _rst = {
                                "fail_to_pass_coverage": round(fail_to_pass_coverage, 2),
                                "pass_to_pass_coverage": round(pass_to_pass_coverage, 2),
                                "overall_accuracy": round(overall_accuracy, 2),
                                "correct_count": resolved_instances,
                                "total_count": total_instances,
                            }
                            raw_results[model_abbr][dataset_abbr] = {
                                "fail_to_pass_coverage": round(fail_to_pass_coverage, 2),
                                "pass_to_pass_coverage": round(pass_to_pass_coverage, 2),
                                "overall_accuracy": round(overall_accuracy, 2),
                                "correct_count": resolved_instances,
                                "total_count": total_instances,
                            }
                            dataset_metrics[dataset_abbr] = ["fail_to_pass_coverage", "pass_to_pass_coverage", "overall_accuracy"]
                            parsed_results[model_abbr][dataset_abbr] = _rst
                            continue
                    except Exception:
                        self.logger.warning(
                            "Failed to parse swebench_pro aggregate result file: %s",
                            aggregate_path,
                        )
                continue

        for dataset in self.dataset_cfgs:
            dataset_abbr = dataset_abbr_from_cfg(dataset)
            dataset_eval_mode[dataset_abbr] = "agent"

        return raw_results, parsed_results, dataset_metrics, dataset_eval_mode


def main():
    import argparse
    import tempfile
    import json
    
    parser = argparse.ArgumentParser(description="Test SWEBenchProSummarizer")
    args = parser.parse_args()
    
    print("Testing SWEBenchProSummarizer...")
    
    test_results = {
        "total_instances": 10,
        "resolved_instances": 6,
        "fail_to_pass_total": 20,
        "fail_to_pass_pass": 15,
        "pass_to_pass_total": 15,
        "pass_to_pass_pass": 12,
        "details": {
            f"inst-{i}": {
                "passed": i % 2 == 0,
                "fail_to_pass_total": 2,
                "fail_to_pass_pass": 2 if i % 2 == 0 else 1,
                "pass_to_pass_total": 1,
                "pass_to_pass_pass": 1,
            } for i in range(10)
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result_path = osp.join(tmpdir, "results", "test_model", "test_dataset", "test_dataset.json")
        os.makedirs(osp.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(test_results, f)
        
        summarizer = SWEBenchProSummarizer.__new__(SWEBenchProSummarizer)
        summarizer.logger = type('MockLogger', (), {'warning': lambda self, *args, **kwargs: None})()
        summarizer.work_dir = tmpdir
        summarizer.model_cfgs = [{'abbr': 'test_model'}]
        summarizer.dataset_cfgs = [{'abbr': 'test_dataset'}]
        
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()
        
        print("\nRaw results:")
        print(json.dumps(raw_results, indent=2))
        
        print("\nParsed results:")
        print(json.dumps(parsed_results, indent=2))
        
        print("\nDataset metrics:")
        print(dataset_metrics)
        
        print("\nDataset eval mode:")
        print(dataset_eval_mode)


if __name__ == "__main__":
    main()