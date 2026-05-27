from ais_bench.benchmark.datasets import SWEBenchProDataset
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks import SWEBenchProInferTask, SWEBenchProEvalTask
from ais_bench.benchmark.summarizers import SWEBenchProSummarizer

STEP_LIMIT = 200

models = [
    dict(
        attr="local",
        abbr="swebench_pro",
        type="LiteLLMChat",
        model="",
        api_key="EMPTY",
        url="http://127.0.0.1:8000/v1",
        batch_size=1,
        generation_kwargs=dict(),
    )
]

datasets = [
    dict(
        type=SWEBenchProDataset,
        abbr="swebench_pro",
        path="",
        name="full",
        split="test",
        step_limit=STEP_LIMIT,
        filter_spec="",
        shuffle=False,
        scripts_dir="",
    ),
]

summarizer = dict(
    attr="accuracy",
    type=SWEBenchProSummarizer,
)

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=SWEBenchProInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=SWEBenchProEvalTask),
    ),
)