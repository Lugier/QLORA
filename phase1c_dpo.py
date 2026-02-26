from pipeline.stages.phase1c_dpo import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("pipeline.stages.phase1c_dpo", run_name="__main__")
