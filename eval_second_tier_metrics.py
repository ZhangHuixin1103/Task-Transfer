import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from skimage import color
from tqdm import tqdm


DATA_TASKS_DIR = Path("data/tasks")
EVAL_DATASET_JSON = Path("data/dataset/eval_dataset.json")
OUTPUT_ROOT = Path("data/output")
RESULT_DIR = OUTPUT_ROOT / "new_metrics"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# From Table 2/3 rows requested by the user and the EMNLP'26 Appendix D.3 discussion:
#   Colorization -> Style Transfer
#   Harmonization -> Style Transfer
#   Inpainting -> Colorization
#   Light Enhancement -> Colorization
#   Deraining -> Style Transfer
#   Inpainting -> Style Transfer
# plus Table 3 rows whose second task is deraining, evaluated with restoration/IQA metrics.
TARGET_PAIRS = (
    "colorization__style_transfer",
    "harmonization__style_transfer",
    "inpainting__colorization",
    "light_enhancement__colorization",
    "deraining__style_transfer",
    "inpainting__style_transfer",
    "dehazing__deraining",
    "shadow_removal__deraining",
)
STYLE_TRANSFER_PAIRS = {
    "colorization__style_transfer",
    "harmonization__style_transfer",
    "deraining__style_transfer",
    "inpainting__style_transfer",
}
COLOR_APPEARANCE_PAIRS = {
    pair for pair in TARGET_PAIRS if pair.endswith("__colorization") or pair.endswith("__style_transfer")
}
RESTORATION_PAIRS = {
    pair for pair in TARGET_PAIRS if pair.endswith("__deraining")
}

DEFAULT_RUNS = {
    "gemini_fix": OUTPUT_ROOT / "output_fix",
    "gemini_qwen": OUTPUT_ROOT / "output_qwen",
    "seedream_fix": OUTPUT_ROOT / "baseline" / "seedream" / "output_fix",
    "seedream_qwen": OUTPUT_ROOT / "baseline" / "seedream" / "output_qwen",
}

METRIC_SOURCES = {
    "lpips_alex": {
        "paper": "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, CVPR 2018",
        "official_code": "https://github.com/richzhang/PerceptualSimilarity",
        "official_interface": "lpips.LPIPS(net='alex'), RGB tensors in [-1, 1]",
        "scope": "paired perceptual metric; used for all selected target tasks",
    },
    "fid": {
        "paper": "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium, NeurIPS 2017",
        "official_code": "https://github.com/bioinf-jku/TTUR/blob/master/fid.py",
        "pytorch_implementation": "https://github.com/mseitzer/pytorch-fid",
        "official_interface": "python -m pytorch_fid /path/to/generated_images /path/to/ground_truth_images",
        "scope": "distribution metric over generated vs. ground-truth images; used for all selected target tasks; defaults to the TTUR-recommended PyTorch implementation",
    },
    "art_fid": {
        "paper": "ArtFID: Quantitative Evaluation of Neural Style Transfer, GCPR 2022",
        "official_code": "https://github.com/matthias-wright/art-fid",
        "official_interface": "python -m art_fid --style_images ... --content_images ... --stylized_images ...",
        "scope": "style-transfer metric; only applied when task B is style_transfer",
    },
    "ciede2000": {
        "paper": "CIEDE2000 color-difference formula; used as a color appearance/statistical measure",
        "official_code": "skimage.color.deltaE_ciede2000 implements the published CIEDE2000 formula",
        "official_interface": "skimage.color.deltaE_ciede2000(Lab_reference, Lab_candidate)",
        "scope": "paired color-difference metric; applied only when task B is colorization or style_transfer",
    },
    "dists": {
        "paper": "Image Quality Assessment: Unifying Structure and Texture Similarity, CVPR 2020",
        "official_code": "https://github.com/dingkeyan93/DISTS",
        "official_interface": "from DISTS_pytorch import DISTS; D(X, Y), RGB tensors in [0, 1]",
        "scope": "paired full-reference IQA metric; useful for restoration targets such as deraining, not deraining-specific",
    },
}


def hashed_id(*parts) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:10]


@dataclass(frozen=True)
class Sample:
    combo_id: str
    pair_key: str
    task_a: str
    task_b: str
    task_a_input: str
    task_a_output: str
    task_b_input: str
    task_b_output: str


def load_eval_samples(eval_json: Path, target_pairs: Iterable[str]) -> List[Sample]:
    target_pairs = set(target_pairs)
    with eval_json.open("r") as f:
        rows = json.load(f)

    samples = []
    for row in rows:
        task_a = row["taskA_input"].split("/")[0]
        task_b = row["taskB_input"].split("/")[0]
        pair_key = f"{task_a}__{task_b}"
        if pair_key not in target_pairs:
            continue
        samples.append(
            Sample(
                combo_id=hashed_id(row["taskA_input"], row["taskB_input"]),
                pair_key=pair_key,
                task_a=task_a,
                task_b=task_b,
                task_a_input=row["taskA_input"],
                task_a_output=row["taskA_output"],
                task_b_input=row["taskB_input"],
                task_b_output=row["taskB_output"],
            )
        )
    return samples


def read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def index_log_images(run_root: Path, pair_key: str) -> Dict[str, Path]:
    pair_dir = run_root / pair_key
    indexed = {}
    for row in read_jsonl(pair_dir / "evaluation_log.jsonl") or []:
        combo_id = row.get("combo_id")
        final_image = row.get("final_image")
        if not combo_id or not final_image:
            continue

        candidates = [Path(final_image)]
        if not Path(final_image).is_absolute():
            candidates.append(Path.cwd() / final_image)

        for candidate in candidates:
            if candidate.exists():
                indexed[combo_id] = candidate
                break
    return indexed


def find_generated_image(run_root: Path, sample: Sample, log_index: Dict[str, Path]) -> Optional[Path]:
    if sample.combo_id in log_index and log_index[sample.combo_id].exists():
        return log_index[sample.combo_id]

    pair_dir = run_root / sample.pair_key
    if not pair_dir.exists():
        return None

    for suffix in IMAGE_SUFFIXES:
        path = pair_dir / f"{sample.combo_id}{suffix}"
        if path.exists():
            return path

    matches = [
        path
        for path in pair_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and sample.combo_id in path.stem
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return sorted(matches)[0]
    return None


def load_rgb_pair(gt_path: Path, gen_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    gt = Image.open(gt_path).convert("RGB")
    gen = Image.open(gen_path).convert("RGB")
    if gen.size != gt.size:
        gen = gen.resize(gt.size, Image.Resampling.BICUBIC)
    return np.asarray(gt), np.asarray(gen)


def compute_ciede2000(gt_path: Path, gen_path: Path) -> Dict[str, float]:
    gt_np, gen_np = load_rgb_pair(gt_path, gen_path)
    delta_e = color.deltaE_ciede2000(color.rgb2lab(gt_np), color.rgb2lab(gen_np))
    return {
        "ciede2000_mean": float(np.mean(delta_e)),
        "ciede2000_median": float(np.median(delta_e)),
        "ciede2000_p95": float(np.percentile(delta_e, 95)),
    }


def ciede_applies(pair_key: str, force_all: bool) -> bool:
    return force_all or pair_key in COLOR_APPEARANCE_PAIRS


class OfficialLpips:
    def __init__(self, enabled: bool, net: str, device: str):
        self.enabled = enabled
        self.net = net
        self.device = device
        self.available = False
        self.reason = "disabled"
        self.model = None
        self.torch = None

        if not enabled:
            return

        try:
            import lpips
            import torch

            self.torch = torch
            self.model = lpips.LPIPS(net=net).to(device)
            self.model.eval()
            self.available = True
            self.reason = "available"
        except Exception as exc:
            self.reason = f"official LPIPS package unavailable or failed to initialize: {exc}"

    def __call__(self, gt_path: Path, gen_path: Path) -> Dict[str, float]:
        if not self.available:
            return {}

        assert self.torch is not None
        assert self.model is not None
        gt_np, gen_np = load_rgb_pair(gt_path, gen_path)
        torch = self.torch
        model = self.model
        with torch.no_grad():
            gt = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            gen = torch.from_numpy(gen_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            gt = gt.to(self.device)
            gen = gen.to(self.device)
            score = model(gt, gen).item()
        return {f"lpips_{self.net}": float(score)}


class OfficialDists:
    def __init__(self, enabled: bool, device: str):
        self.enabled = enabled
        self.device = device
        self.available = False
        self.reason = "disabled"
        self.model = None
        self.torch = None

        if not enabled:
            return

        try:
            import torch
            from DISTS_pytorch import DISTS

            self.torch = torch
            self.model = DISTS().to(device)
            self.model.eval()
            self.available = True
            self.reason = "available"
        except Exception as exc:
            self.reason = f"official DISTS package unavailable or failed to initialize: {exc}"

    def __call__(self, gt_path: Path, gen_path: Path) -> Dict[str, float]:
        if not self.available:
            return {}

        assert self.torch is not None
        assert self.model is not None
        gt_np, gen_np = load_rgb_pair(gt_path, gen_path)
        torch = self.torch
        model = self.model
        with torch.no_grad():
            gt = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            gen = torch.from_numpy(gen_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            gt = gt.to(self.device)
            gen = gen.to(self.device)
            score = model(gt, gen).item()
        return {"dists": float(score)}


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def write_resized_rgb(src: Path, dst: Path, size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        image.save(dst)


def materialize_metric_dirs(
    rows: List[dict],
    result_dir: Path,
    data_tasks_dir: Path,
    copy_images: bool,
    distribution_image_size: int,
) -> Dict[Tuple[str, str], Dict[str, Path]]:
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for row in rows:
        grouped.setdefault((row["run"], row["pair_key"]), []).append(row)

    stage_root = result_dir / "metric_dirs"
    distribution_stage_root = result_dir / f"metric_dirs_{distribution_image_size}px"
    out = {}
    for (run_name, pair_key), items in grouped.items():
        base = stage_root / run_name / pair_key
        dist_base = distribution_stage_root / run_name / pair_key
        dirs = {
            "gt": base / "gt",
            "generated": base / "generated",
            "content": base / "content",
            "style": base / "style",
            "fid_gt": dist_base / "gt",
            "fid_generated": dist_base / "generated",
            "artfid_content": dist_base / "content",
            "artfid_style": dist_base / "style",
            "artfid_generated": dist_base / "generated",
        }
        for item in items:
            stem = item["combo_id"]
            gen_path = Path(item["generated_image"])
            gt_path = data_tasks_dir / item["task_b_output"]
            content_path = data_tasks_dir / item["task_b_input"]

            link_or_copy(gt_path, dirs["gt"] / f"{stem}{gt_path.suffix.lower()}", copy_images)
            link_or_copy(gen_path, dirs["generated"] / f"{stem}{gen_path.suffix.lower()}", copy_images)
            link_or_copy(content_path, dirs["content"] / f"{stem}{content_path.suffix.lower()}", copy_images)
            # ArtFID needs a style image distribution. For this benchmark, the closest
            # available style distribution is the task-B ground-truth output set.
            link_or_copy(gt_path, dirs["style"] / f"{stem}{gt_path.suffix.lower()}", copy_images)

            # pytorch-fid and art-fid batch images with torch.stack, so all images
            # in their staged directories must have identical spatial dimensions.
            write_resized_rgb(gt_path, dirs["fid_gt"] / f"{stem}.png", distribution_image_size)
            write_resized_rgb(gen_path, dirs["fid_generated"] / f"{stem}.png", distribution_image_size)
            write_resized_rgb(content_path, dirs["artfid_content"] / f"{stem}.png", distribution_image_size)
            write_resized_rgb(gt_path, dirs["artfid_style"] / f"{stem}.png", distribution_image_size)
            write_resized_rgb(gen_path, dirs["artfid_generated"] / f"{stem}.png", distribution_image_size)
        out[(run_name, pair_key)] = dirs
    return out


def parse_first_float(text: str) -> Optional[float]:
    match = re.search(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else None


def run_ttur_fid(python_bin: str, fid_py: Path, generated_dir: Path, gt_dir: Path) -> Tuple[Optional[float], str]:
    cmd = [python_bin, str(fid_py), str(generated_dir), str(gt_dir)]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    text = "\n".join(part for part in [proc.stdout, proc.stderr] if part)
    if proc.returncode != 0:
        return None, text.strip()
    return parse_first_float(text), text.strip()


def run_pytorch_fid(
    python_bin: str,
    generated_dir: Path,
    gt_dir: Path,
    device: str,
    batch_size: int,
    dims: int,
    num_workers: Optional[int],
) -> Tuple[Optional[float], str]:
    cmd = [
        python_bin,
        "-m",
        "pytorch_fid",
        str(generated_dir),
        str(gt_dir),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--dims",
        str(dims),
    ]
    if num_workers is not None:
        cmd.extend(["--num-workers", str(num_workers)])
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    text = "\n".join(part for part in [proc.stdout, proc.stderr] if part)
    if proc.returncode != 0:
        return None, text.strip()
    return parse_first_float(text), text.strip()


def run_official_artfid(
    python_bin: str,
    content_dir: Path,
    style_dir: Path,
    generated_dir: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    content_metric: str,
) -> Tuple[Dict[str, Optional[float]], str]:
    cmd = [
        python_bin,
        "-m",
        "art_fid",
        "--style_images",
        str(style_dir),
        "--content_images",
        str(content_dir),
        "--stylized_images",
        str(generated_dir),
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--content_metric",
        content_metric,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    text = "\n".join(part for part in [proc.stdout, proc.stderr] if part)
    if proc.returncode != 0:
        return {"art_fid": None}, text.strip()

    metrics = {}
    match = re.search(
        r"ArtFID\s+value:\s*([-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    metrics["art_fid"] = float(match.group(1)) if match else parse_first_float(text)
    return metrics, text.strip()


def collect_rows(args: argparse.Namespace, samples: List[Sample]) -> Tuple[List[dict], List[dict]]:
    rows = []
    missing = []
    for run_name, run_root in args.runs.items():
        run_root = Path(run_root)
        log_indexes = {
            pair_key: index_log_images(run_root, pair_key)
            for pair_key in sorted({sample.pair_key for sample in samples})
        }

        for sample in tqdm(samples, desc=f"collect:{run_name}", disable=args.quiet):
            gen_path = find_generated_image(run_root, sample, log_indexes[sample.pair_key])
            if gen_path is None:
                missing.append(
                    {
                        "run": run_name,
                        "pair_key": sample.pair_key,
                        "combo_id": sample.combo_id,
                    }
                )
                continue

            rows.append(
                {
                    "run": run_name,
                    "pair_key": sample.pair_key,
                    "combo_id": sample.combo_id,
                    "task_a": sample.task_a,
                    "task_b": sample.task_b,
                    "task_a_input": sample.task_a_input,
                    "task_a_output": sample.task_a_output,
                    "task_b_input": sample.task_b_input,
                    "task_b_output": sample.task_b_output,
                    "gt_image": str(args.data_tasks_dir / sample.task_b_output),
                    "generated_image": str(gen_path),
                    "metrics": {},
                }
            )
    return rows, missing


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_sample_manifest(path: Path, rows: List[dict]) -> None:
    fields = [
        "run",
        "pair_key",
        "combo_id",
        "task_a_input",
        "task_a_output",
        "task_b_input",
        "task_b_output",
        "gt_image",
        "generated_image",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def mean(values: List[float]) -> Optional[float]:
    vals = [value for value in values if isinstance(value, (int, float)) and np.isfinite(value)]
    return float(np.mean(vals)) if vals else None


def aggregate(rows: List[dict], group_metrics: Dict[Tuple[str, str], dict]) -> Dict[str, dict]:
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for row in rows:
        grouped.setdefault((row["run"], row["pair_key"]), []).append(row)

    summaries = {}
    for (run_name, pair_key), items in sorted(grouped.items()):
        metric_names = sorted({name for item in items for name in item["metrics"]})
        summary = {
            "run": run_name,
            "pair_key": pair_key,
            "num_samples": len(items),
        }
        for metric_name in metric_names:
            summary[f"avg_{metric_name}"] = mean(
                [item["metrics"].get(metric_name) for item in items]
            )
        summary.update(group_metrics.get((run_name, pair_key), {}))
        summaries[f"{run_name}/{pair_key}"] = summary
    return summaries


def write_summary_csv(path: Path, summaries: Dict[str, dict]) -> None:
    metric_fields = sorted(
        {
            field
            for row in summaries.values()
            for field in row
            if field not in {"run", "pair_key", "num_samples"}
        }
    )
    fields = ["run", "pair_key", "num_samples", *metric_fields]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries.values():
            writer.writerow({field: row.get(field) for field in fields})


def run(args: argparse.Namespace) -> None:
    target_pairs = set(args.pair or TARGET_PAIRS)
    samples = load_eval_samples(args.eval_json, target_pairs)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    rows, missing = collect_rows(args, samples)
    write_jsonl(args.result_dir / "sample_details.jsonl", rows)
    write_sample_manifest(args.result_dir / "sample_manifest.csv", rows)
    with (args.result_dir / "missing_images.json").open("w") as f:
        json.dump(missing, f, indent=2)

    stage_dirs = materialize_metric_dirs(
        rows,
        result_dir=args.result_dir,
        data_tasks_dir=args.data_tasks_dir,
        copy_images=args.copy_images,
        distribution_image_size=args.distribution_image_size,
    )

    unavailable = {}
    lpips = OfficialLpips(
        enabled=args.compute_lpips,
        net=args.lpips_net,
        device=args.device,
    )
    dists = OfficialDists(
        enabled=args.compute_dists,
        device=args.device,
    )
    if args.compute_lpips and not lpips.available:
        unavailable[f"lpips_{args.lpips_net}"] = lpips.reason
    if args.compute_dists and not dists.available:
        unavailable["dists"] = dists.reason

    if args.compute_ciede2000 or (args.compute_lpips and lpips.available) or (args.compute_dists and dists.available):
        for row in tqdm(rows, desc="paired-metrics", disable=args.quiet):
            gt_path = Path(row["gt_image"])
            gen_path = Path(row["generated_image"])
            if args.compute_ciede2000 and ciede_applies(row["pair_key"], args.ciede_all):
                row["metrics"].update(compute_ciede2000(gt_path, gen_path))
            if args.compute_lpips and lpips.available:
                row["metrics"].update(lpips(gt_path, gen_path))
            if args.compute_dists and dists.available:
                row["metrics"].update(dists(gt_path, gen_path))
        write_jsonl(args.result_dir / "sample_details.jsonl", rows)

    group_metrics = {}
    raw_logs = {}
    if args.compute_fid:
        if args.fid_backend == "ttur":
            if args.ttur_fid_py is None:
                unavailable["fid"] = "pass --ttur-fid-py /path/to/TTUR/fid.py when --fid-backend ttur"
            else:
                for key, dirs in tqdm(stage_dirs.items(), desc="fid-ttur", disable=args.quiet):
                    score, raw = run_ttur_fid(args.fid_python, args.ttur_fid_py, dirs["fid_generated"], dirs["fid_gt"])
                    group_metrics.setdefault(key, {})["fid"] = score
                    raw_logs[f"fid_ttur/{key[0]}/{key[1]}"] = raw
        else:
            for key, dirs in tqdm(stage_dirs.items(), desc="fid", disable=args.quiet):
                score, raw = run_pytorch_fid(
                    args.fid_python,
                    dirs["fid_generated"],
                    dirs["fid_gt"],
                    device=args.device,
                    batch_size=args.fid_batch_size,
                    dims=args.fid_dims,
                    num_workers=args.fid_num_workers,
                )
                group_metrics.setdefault(key, {})["fid"] = score
                raw_logs[f"fid_pytorch/{key[0]}/{key[1]}"] = raw

    if args.compute_artfid:
        for key, dirs in tqdm(stage_dirs.items(), desc="artfid", disable=args.quiet):
            run_name, pair_key = key
            if pair_key not in STYLE_TRANSFER_PAIRS:
                continue
            metrics, raw = run_official_artfid(
                args.artfid_python,
                content_dir=dirs["artfid_content"],
                style_dir=dirs["artfid_style"],
                generated_dir=dirs["artfid_generated"],
                device=args.device,
                batch_size=args.artfid_batch_size,
                num_workers=args.artfid_num_workers,
                content_metric=args.artfid_content_metric,
            )
            group_metrics.setdefault(key, {}).update(metrics)
            raw_logs[f"artfid/{run_name}/{pair_key}"] = raw

    summaries = aggregate(rows, group_metrics)
    with (args.result_dir / "summary.json").open("w") as f:
        json.dump(summaries, f, indent=2)
    write_summary_csv(args.result_dir / "summary.csv", summaries)
    with (args.result_dir / "official_metric_raw_logs.json").open("w") as f:
        json.dump(raw_logs, f, indent=2)

    manifest = {
        "target_pairs": sorted(target_pairs),
        "runs": {name: str(path) for name, path in args.runs.items()},
        "num_samples_found": len(rows),
        "num_missing_images": len(missing),
        "metric_sources": METRIC_SOURCES,
        "appendix_d3_reference_mapping": {
            "colorization": [
                "Su et al. 2020, Instance-aware Image Colorization",
                "Kang et al. 2023, DDColor",
                "Wu et al. 2021 as cited by Appendix D.3 for colorization metrics",
            ],
            "style_transfer": [
                "Deng et al. 2022, StyTr2",
                "Wright and Ommer 2022, ArtFID",
            ],
            "deraining": [
                "Table 3 target-deraining rows included with official full-reference/perceptual IQA metrics only.",
            ],
        },
        "unavailable_metrics": unavailable,
        "notes": [
            "No PSNR or SSIM is computed or reported.",
            "FID defaults to mseitzer/pytorch-fid, the PyTorch implementation recommended by the official TTUR README; use --fid-backend ttur for the original TensorFlow fid.py.",
            "ArtFID is run only through the official art-fid package and only for task-B style_transfer pairs.",
            "CIEDE2000 is applied only to task-B colorization/style_transfer pairs unless --ciede-all is set.",
            "DISTS is full-reference IQA for restoration/perceptual quality; it is not a deraining-specific metric.",
            "No reliable deraining/reflection-removal/demoireing-specific official metric implementation was found in Appendix D.3 references; task-B deraining rows are included with LPIPS/FID/DISTS only.",
            "evaluation_log.jsonl is only an auxiliary index; generated images are also found by combo_id in the output directories.",
        ],
    }
    with (args.result_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Prepared {len(rows)} existing generated-image rows in {args.result_dir}")
    if missing:
        print(f"Missing generated images: {len(missing)}")
    if unavailable:
        print(f"Unavailable/skipped official metrics: {unavailable}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate selected Table 2/3 target-task pairs with official-code new metrics."
    )
    parser.add_argument("--eval-json", type=Path, default=EVAL_DATASET_JSON)
    parser.add_argument("--data-tasks-dir", type=Path, default=DATA_TASKS_DIR)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--pair", action="append", choices=TARGET_PAIRS)
    parser.add_argument("--run", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--copy-images", action="store_true", help="Copy staged metric images instead of symlinking.")
    parser.add_argument("--distribution-image-size", type=int, default=299, help="Resize staged images for batched FID/ArtFID directory metrics.")

    parser.add_argument("--compute-ciede2000", action="store_true")
    parser.add_argument("--ciede-all", action="store_true", help="Also run CIEDE2000 outside colorization/style-transfer targets.")
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--compute-dists", action="store_true")
    parser.add_argument("--compute-fid", action="store_true")
    parser.add_argument("--fid-backend", default="pytorch_fid", choices=["pytorch_fid", "ttur"])
    parser.add_argument("--fid-batch-size", type=int, default=50)
    parser.add_argument("--fid-num-workers", type=int)
    parser.add_argument("--fid-dims", type=int, default=2048, choices=[64, 192, 768, 2048])
    parser.add_argument("--ttur-fid-py", type=Path, help="Path to official bioinf-jku/TTUR/fid.py, only used with --fid-backend ttur.")
    parser.add_argument("--compute-artfid", action="store_true")
    parser.add_argument("--artfid-batch-size", type=int, default=32)
    parser.add_argument("--artfid-num-workers", type=int, default=4)
    parser.add_argument("--artfid-content-metric", default="lpips", choices=["lpips", "vgg", "alexnet"])
    parser.add_argument("--fid-python", default=sys.executable, help="Python executable containing pytorch-fid, or TF1 when --fid-backend ttur.")
    parser.add_argument("--artfid-python", default=sys.executable)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    if args.run:
        runs = {}
        for item in args.run:
            if "=" not in item:
                raise ValueError(f"--run must be NAME=PATH, got {item}")
            name, path = item.split("=", 1)
            runs[name] = Path(path)
        args.runs = runs
    else:
        args.runs = DEFAULT_RUNS
    return args


if __name__ == "__main__":
    run(parse_args())
