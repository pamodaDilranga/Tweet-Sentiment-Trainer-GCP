from kfp import compiler
import Kubeflow.pipeline_kfp as pipeline_kfp

compiler.Compiler().compile(
    pipeline_func=pipeline_kfp.sentiment_pipeline,
    package_path="sentiment_pipeline.json"
)

print("Wrote: sentiment_pipeline.json")