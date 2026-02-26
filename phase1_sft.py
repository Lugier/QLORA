from pipeline.stages.phase1_sft import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("pipeline.stages.phase1_sft", run_name="__main__")
