"""Can cheap pre-inference image features choose 512 vs 640 on Jetson?

Fits only on 512 development images and reports AP on 512 held-out images.
GT-informed reference is explicitly non-deployable. Runtime is a linear
estimate until a real routed engine benchmarks both paths on device.

Run: uv run --no-project --with pycocotools==2.0.10 \
    --with numpy==2.2.6 --with scikit-learn==1.7.2 \
    resolution_preimage_probe.py
"""

import collections
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from resolution_routing_probe import object_gain

ROOT = Path(__file__).parent


def evaluate(coco: COCO, rows: dict[int, list[dict]], ids: list[int]) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        loaded = coco.loadRes(
            [row for image_id in ids for row in rows.get(image_id, [])]
        )
        evaluator = COCOeval(coco, loaded, "bbox")
        evaluator.params.imgIds = ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "ap50_95": float(evaluator.stats[0]),
        "ap_small": float(evaluator.stats[3]),
    }


def load_predictions() -> dict[str, dict[int, list[dict]]]:
    result = {}
    with tarfile.open(ROOT / "artifacts/resolution-predictions.tgz", "r:gz") as tar:
        for kind in ("r512", "r576", "base640"):
            grouped = collections.defaultdict(list)
            for row in json.load(tar.extractfile(f"{kind}-predictions.json")):
                grouped[row["image_id"]].append(row)
            result[kind] = grouped
    return result


def main() -> None:
    coco = COCO(str(ROOT / "artifacts/instances_pilot_development.json"))
    all_ids = sorted(coco.getImgIds())
    ranked = sorted(
        all_ids,
        key=lambda item: hashlib.sha256(f"router-seed-20261004:{item}".encode()).digest(),
    )
    fit_ids, test_ids = ranked[:512], sorted(ranked[512:])
    if len(set(fit_ids) & set(test_ids)) or len(test_ids) != 512:
        raise RuntimeError("Routing splits overlap")
    predictions = load_predictions()
    source = json.loads((ROOT / "artifacts/preimage-features-fast.json").read_text())
    if (
        not source["no_ground_truth_or_detector_output"]
        or set(map(int, source["rows"])) != set(all_ids)
    ):
        raise RuntimeError("Image-only feature contract violated")
    features = {
        int(image_id): np.asarray(values, dtype=np.float64)
        for image_id, values in source["rows"].items()
    }
    gt = collections.defaultdict(list)
    for row in coco.dataset["annotations"]:
        gt[row["image_id"]].append(row)
    gains = {
        image_id: object_gain(
            gt[image_id],
            predictions["r512"][image_id],
            predictions["base640"][image_id],
        )
        for image_id in all_ids
    }
    fit_X = np.stack([features[image_id] for image_id in fit_ids])
    test_X = np.stack([features[image_id] for image_id in test_ids])
    labels = np.asarray([gains[image_id] > 0 for image_id in fit_ids], dtype=int)
    if not 15 <= int(labels.sum()) <= 497:
        raise RuntimeError("Routing classifier has degenerate fit labels")
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            random_state=20261004,
        ),
    )
    logistic.fit(fit_X, labels)
    fit_probabilities = logistic.predict_proba(fit_X)[:, 1]
    # Fix a per-image threshold on the FIT split only. A top-k ranking over
    # the held-out batch is not an independently deployable decision rule.
    fit_threshold = float(np.quantile(fit_probabilities, 0.8))
    scaler = logistic.named_steps["standardscaler"]
    classifier = logistic.named_steps["logisticregression"]
    linear_weights = classifier.coef_[0] / scaler.scale_
    linear_bias = float(
        classifier.intercept_[0] - np.dot(linear_weights, scaler.mean_)
    )
    threshold_logit = float(
        np.log(fit_threshold / (1 - fit_threshold))
    )
    router_policy = {
        "source": "COCO train2017 development, 512-image fit split",
        "feature_names": source["features"],
        "weights": linear_weights.tolist(),
        "bias": linear_bias,
        "threshold_logit": threshold_logit,
        "target_fit_fallback_fraction": 0.2,
        "fit_threshold_probability": fit_threshold,
        "feature_median_ms_jetson": source["median_ms_after_image_decode"],
        "no_ground_truth_or_detector_input_at_inference": True,
    }
    (ROOT / "results/preimage-router-policy.json").write_text(
        json.dumps(router_policy, indent=2, sort_keys=True) + "\n"
    )
    forest = RandomForestRegressor(
        n_estimators=80,
        max_depth=4,
        min_samples_leaf=12,
        max_features=0.8,
        n_jobs=1,
        random_state=20261004,
    )
    forest.fit(fit_X, np.asarray([gains[image_id] for image_id in fit_ids]))
    rankings = {
        "linear_image_only": dict(zip(test_ids, logistic.predict_proba(test_X)[:, 1].tolist())),
        "forest_image_only": dict(zip(test_ids, forest.predict(test_X).tolist())),
        "gt_reference_not_deployable": {image_id: gains[image_id] for image_id in test_ids},
        "random_control": {
            image_id: int.from_bytes(
                hashlib.sha256(f"preimage-random:{image_id}".encode()).digest()[:8],
                "big",
            )
            for image_id in test_ids
        },
    }
    fixed = {
        name: evaluate(coco, predictions[name], test_ids)
        for name in ("r512", "r576", "base640")
    }
    timings = {
        name: json.loads(
            (ROOT / "results/resolution" / f"{name}-predictions.summary.json").read_text()
        )
        for name in ("r512", "r576", "base640")
    }
    results = {}
    for budget in (0.1, 0.2, 0.28):
        count = round(len(test_ids) * budget)
        for policy, ranking in rankings.items():
            chosen = set(
                sorted(test_ids, key=lambda item: (-ranking[item], item))[:count]
            )
            mixed = {
                image_id: (
                    predictions["base640"][image_id]
                    if image_id in chosen
                    else predictions["r512"][image_id]
                )
                for image_id in test_ids
            }
            ap = evaluate(coco, mixed, test_ids)
            frac = count / len(test_ids)
            # A weighted median is only a screening proxy, not a measured
            # end-to-end latency. No image is inferred twice.
            app_proxy = (
                (1 - frac) * timings["r512"]["decode_through_predictions_ms_median"]
                + frac * timings["base640"]["decode_through_predictions_ms_median"]
                + source["median_ms_after_image_decode"]
            )
            inference_proxy = (
                (1 - frac) * timings["r512"]["host_input_inference_output_ms_median"]
                + frac * timings["base640"]["host_input_inference_output_ms_median"]
                + source["median_ms_after_image_decode"]
            )
            results[f"{policy}@{budget:.2f}"] = {
                **ap,
                "selected_images": count,
                "fallback_fraction": frac,
                "estimated_app_median_ms_not_measured": app_proxy,
                "estimated_inference_plus_feature_ms_not_measured": inference_proxy,
                "uses_test_ground_truth": policy == "gt_reference_not_deployable",
                "uses_heldout_batch_for_top_k": True,
            }
    independent_selection = {
        image_id
        for image_id, value in rankings["linear_image_only"].items()
        if value >= fit_threshold
    }
    independent_fraction = len(independent_selection) / len(test_ids)
    independent_predictions = {
        image_id: (
            predictions["base640"][image_id]
            if image_id in independent_selection
            else predictions["r512"][image_id]
        )
        for image_id in test_ids
    }
    independent_ap = evaluate(coco, independent_predictions, test_ids)
    independent_ap.update(
        {
            "selected_images": len(independent_selection),
            "fallback_fraction": independent_fraction,
            "estimated_app_median_ms_not_measured": (
                (1 - independent_fraction)
                * timings["r512"]["decode_through_predictions_ms_median"]
                + independent_fraction
                * timings["base640"]["decode_through_predictions_ms_median"]
                + source["median_ms_after_image_decode"]
            ),
            "uses_test_ground_truth": False,
            "uses_heldout_batch_for_top_k": False,
        }
    )
    report = {
        "scope": "COCO train2017 development only, not val2017",
        "fit_count": len(fit_ids),
        "heldout_count": len(test_ids),
        "fit_ids_sha256": hashlib.sha256(
            ",".join(map(str, sorted(fit_ids))).encode()
        ).hexdigest(),
        "heldout_ids_sha256": hashlib.sha256(
            ",".join(map(str, test_ids)).encode()
        ).hexdigest(),
        "feature_names": source["features"],
        "feature_median_ms_jetson": source["median_ms_after_image_decode"],
        "fit_positive_images": int(labels.sum()),
        "fixed_heldout_ap": fixed,
        "routing_options": results,
        "deployable_fit_threshold_probe": independent_ap,
        "caveat": (
            "GT reference is not deployable. No test GT was used to fit the "
            "image-only policies. Timing is an optimistic weighted-median "
            "proxy; actual routed Jetson measurements are required."
        ),
    }
    out = ROOT / "results/preimage-routing-probe.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("Fixed held-out:", {name: round(value["ap50_95"], 5) for name, value in fixed.items()})
    print(
        "Fit positives", int(labels.sum()),
        "feature median", source["median_ms_after_image_decode"], "ms",
    )
    for name, value in results.items():
        print(
            f"{name:39s} AP={value['ap50_95']:.5f} "
            f"small={value['ap_small']:.5f} "
            f"proxy_app_ms={value['estimated_app_median_ms_not_measured']:.2f}"
        )
    print(
        "Per-image threshold fixed on fit split:",
        f"AP={independent_ap['ap50_95']:.5f}",
        f"fallback={independent_fraction:.3f}",
        f"proxy_app_ms={independent_ap['estimated_app_median_ms_not_measured']:.2f}",
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
