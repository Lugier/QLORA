from pipeline.stages.export import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("pipeline.stages.export", run_name="__main__")
