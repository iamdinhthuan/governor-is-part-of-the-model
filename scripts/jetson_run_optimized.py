"""Run isolated matched TensorRT builder-3 and precision-boundary checks."""

import jetson_run_pilot

jetson_run_pilot.ENGINES = {
    "opt3_float": "pilot_float_fp16_opt3.engine",
    "opt3_qat": "pilot_qat_mixed_opt3.engine",
    "gated_opt0": "pilot_qat_gated_opt0.engine",
}

if __name__ == "__main__":
    jetson_run_pilot.main()
