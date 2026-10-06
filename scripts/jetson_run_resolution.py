"""Measure fixed input resolutions on the identical development split."""

import jetson_run_pilot

jetson_run_pilot.ENGINES = {
    "base640": "yolo26n_84172_fp16_opt0.engine",
    "r576": "resolution_576_opt0.engine",
    "r512": "resolution_512_opt0.engine",
}
jetson_run_pilot.ENGINE_SIZES = {"base640": 640, "r576": 576, "r512": 512}

if __name__ == "__main__":
    jetson_run_pilot.main()
