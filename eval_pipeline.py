from pipeline.stages.eval_pipeline import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("pipeline.stages.eval_pipeline", run_name="__main__")
