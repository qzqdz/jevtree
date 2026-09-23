"""Official jevtree CLI.

Primary product entry: `jevtree decide --data PATH --goal "..."`.
AFA hard-budget scoring: `jevtree eval-afa`.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jevtree import __version__


def _repo_root() -> Path:
    # src/jevtree/cli/main.py → parents[3] = repo root when editable layout
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "pyproject.toml").is_file() and (p / "src").is_dir():
            return p
    return Path.cwd()


def _git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    # minimal YAML subset for our eval configs (no PyYAML required)
    return _parse_minimal_yaml(text)


def _parse_minimal_yaml(text: str) -> dict[str, Any]:
    """Tiny YAML subset: nested maps via indent, lists via [a,b] or - items."""
    # Prefer JSON if the file is actually JSON
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return json.loads(text)

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    pending_list_key: str | None = None
    pending_list_indent = -1

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        # pop stack
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else root

        if line.startswith("- "):
            item = line[2:].strip()
            if pending_list_key is not None and indent > pending_list_indent:
                parent.setdefault(pending_list_key, [])
                # attach to the dict that owns pending_list_key — walk up
                # simpler: store on current parent if key exists else create
            # find list owner
            owner = parent
            key = pending_list_key
            if key is None:
                continue
            if not isinstance(owner.get(key), list):
                owner[key] = []
            owner[key].append(_yaml_scalar(item))
            continue

        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest == "" or rest.startswith("#"):
            # nested map or upcoming list
            parent[key] = {}
            stack.append((indent, parent[key]))
            pending_list_key = key
            pending_list_indent = indent
        elif rest.startswith("[") and rest.endswith("]"):
            inner = rest[1:-1].strip()
            if not inner:
                parent[key] = []
            else:
                parent[key] = [_yaml_scalar(x.strip()) for x in inner.split(",")]
            pending_list_key = None
        else:
            # strip inline comment
            if " #" in rest:
                rest = rest.split(" #", 1)[0].rstrip()
            parent[key] = _yaml_scalar(rest)
            pending_list_key = None
    return root


def _yaml_scalar(s: str) -> Any:
    if s in ("null", "Null", "NULL", "~"):
        return None
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _load_csv(path: Path, label: str) -> tuple[list[dict[str, Any]], list[str], str]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
    if not rows:
        raise SystemExit(f"empty CSV: {path}")
    if label not in rows[0]:
        raise SystemExit(f"label column {label!r} not in CSV columns {list(rows[0])}")
    feature_keys = [k for k in rows[0].keys() if k != label]
    return rows, feature_keys, label




def cmd_decide(args: argparse.Namespace) -> int:
    """Universal entry: data path + NL goal → LLM FeatureTable → grow artifacts."""
    from jevtree.data.decide_pipeline import run_decide

    data = getattr(args, "data", None)
    goal = getattr(args, "goal", None)
    if not data:
        print("jevtree decide: --data is required", file=sys.stderr)
        return 2
    if not goal or not str(goal).strip():
        print("jevtree decide: --goal is required (natural-language requirement)", file=sys.stderr)
        return 2

    out = getattr(args, "out", None)
    try:
        result = run_decide(
            data,
            str(goal).strip(),
            out_dir=out,
            criterion=getattr(args, "criterion", None) or "gain",
            max_depth=args.max_depth,
            min_samples=int(args.min_samples or 1),
            seed=int(args.seed or 0),
            trace_examples=int(getattr(args, "trace_examples", 0) or 0),
            repo_root=_repo_root(),
        )
    except FileNotFoundError as exc:
        print(f"jevtree decide: {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"jevtree decide: {exc}", file=sys.stderr)
        return 1

    print(
        f"decide ok → {result['out_dir']}  "
        f"rows={result['n_rows']} features={result['n_features']} "
        f"label={result['label_key']}"
    )
    print(f"  feature_table → {result['feature_table_json']}")
    print(f"  tree → {result['tree_out']}")
    print(f"  sop → {result['sop_out']}")
    print(f"  story → {result['story_out']}")
    if result.get("traces"):
        print(f"  traces → {result['out_dir']}/traces.json  (n={len(result['traces'])})")
    if getattr(args, "story", True):
        print(result["story_zh"])
        print(result["story_en"])
    return 0



def _count_tree_nodes(node: Any) -> int:
    from jevtree.core.tree import TreeLeaf, TreeNode

    if isinstance(node, TreeLeaf):
        return 1
    if isinstance(node, TreeNode):
        return 1 + sum(_count_tree_nodes(c) for c in node.children.values())
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from jevtree.core.sop import DecisionSOP

    if not args.sop:
        print("jevtree validate: ok (no SOP path — light check)")
        print(f"version={__version__}")
        return 0
    path = Path(args.sop)
    data = json.loads(path.read_text(encoding="utf-8"))
    sop = DecisionSOP.from_dict(data)
    errors = sop.validate()
    if errors:
        print("INVALID:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"VALID  name={sop.name!r} hash={sop.hash}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if not args.sop:
        print("jevtree run: --sop is required", file=sys.stderr)
        return 2
    try:
        obs = json.loads(Path(args.input).read_text(encoding="utf-8")) if args.input else {}
        payload = json.loads(Path(args.sop).read_text(encoding="utf-8"))
        from jevtree.runtime.engine import RuntimeEngine
        engine = RuntimeEngine()
        if getattr(args, "trace", False):
            traced = engine.run_sop_traced(payload, obs)
            for i, step in enumerate(traced.get("path") or [], start=1):
                print(f"Q{i}: {step.get('feature')} = {step.get('value')}")
            print(
                f"Decision: {traced.get('decision')}  "
                f"(questions used: {traced.get('questions_used')})"
            )
            print(json.dumps(traced, ensure_ascii=False, default=str))
        else:
            result = engine.run_sop(payload, obs)
            print(json.dumps({"decision": result}, ensure_ascii=False, default=str))
    except (OSError, KeyError, json.JSONDecodeError, TypeError, ValueError, NotImplementedError) as exc:
        print(f"jevtree run: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_eval_afa(args: argparse.Namespace) -> int:
    from jevtree.data.cube import load_cube_split
    from jevtree.eval.afabench import AFABenchAdapter
    from jevtree.eval.protocol import HardBudgetEpisodeConfig, HardBudgetProtocol
    from jevtree.policy.baselines import RandomAcquisitionPolicy, SequentialAcquisitionPolicy
    from jevtree.policy.conditional_ig import ConditionalIGAcquisitionPolicy
    from jevtree.policy.discriminative import DiscriminativeAcquisitionPolicy
    from jevtree.policy.ig_acquisition import StaticIGAcquisitionPolicy
    from jevtree.policy.predictor import normalize_predictor_name

    if not args.config:
        print("jevtree eval-afa: --config is required", file=sys.stderr)
        return 2

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        # try relative to repo root
        alt = _repo_root() / args.config
        if alt.is_file():
            cfg_path = alt
        else:
            print(f"config not found: {args.config}", file=sys.stderr)
            return 2

    cfg = _load_config(cfg_path)
    dataset_id = cfg.get("dataset_id", "cube_without_noise")
    split_seed = int(cfg.get("split_seed", 0))
    hard_budgets = cfg.get("hard_budgets") or [cfg.get("hard_budget", 3)]
    hard_budgets = [int(b) for b in hard_budgets]

    policy_cfg = cfg.get("policy") or {}
    if isinstance(policy_cfg, str):
        policy_cfg = {"name": policy_cfg}
    policy_name = policy_cfg.get("name", "ig_static")

    output_cfg = cfg.get("output") or {}
    results_dir = Path(output_cfg.get("results_dir") or "results/")
    if not results_dir.is_absolute():
        results_dir = _repo_root() / results_dir

    data_provenance: dict[str, Any] = {}
    continuous_keys: list[str] = list(policy_cfg.get("continuous_keys") or [])

    # load data
    if dataset_id in ("cube_without_noise", "cube"):
        n_samples = int(cfg.get("n_samples", 256))
        n_features = int(cfg.get("n_features", 5))
        split = load_cube_split(
            n_samples=n_samples, n_features=n_features, seed=split_seed
        )
        train, test = split["train"], split["test"]
        feature_keys = split["feature_keys"]
        label_key = split["label_key"]
        dataset_hash = split["dataset_hash"]
        data_provenance = {
            "loader": "cube",
            "n_samples": n_samples,
            "n_features": n_features,
        }
    elif dataset_id in ("miniboone", "diabetes", "bank_marketing"):
        from jevtree.data.tabular import load_tabular_split

        n_train = cfg.get("n_train")
        n_test = cfg.get("n_test")
        allow_fallback = bool(cfg.get("allow_fallback", True))
        split = load_tabular_split(
            dataset_id,
            seed=split_seed,
            n_train=int(n_train) if n_train is not None else None,
            n_test=int(n_test) if n_test is not None else None,
            cache_dir=cfg.get("cache_dir"),
            allow_fallback=allow_fallback,
        )
        train, test = split["train"], split["test"]
        feature_keys = split["feature_keys"]
        label_key = split["label_key"]
        dataset_hash = split["dataset_hash"]
        continuous_keys = list(split.get("continuous_keys") or feature_keys)
        dataset_id = split["dataset_id"]  # may be fallback id
        data_provenance = {
            "loader": "tabular",
            "requested_dataset_id": split.get("requested_dataset_id"),
            "fallback_used": split.get("fallback_used"),
            "subsample": split.get("subsample"),
            "n_train": split.get("n_train"),
            "n_test": split.get("n_test"),
            "n_features": split.get("n_features"),
            "full_file_sha256": split.get("full_file_sha256"),
            "provenance": split.get("provenance"),
        }
        if split.get("fallback_used"):
            print(
                f"NOTE: requested dataset unavailable; using fallback "
                f"{dataset_id!r} (see summary provenance)",
                file=sys.stderr,
            )
    else:
        dataset_path = cfg.get("dataset_path")
        if not dataset_path:
            print(
                f"dataset_id={dataset_id!r} requires dataset_path "
                f"(or use cube_without_noise|miniboone|diabetes|bank_marketing)",
                file=sys.stderr,
            )
            return 2
        label_key = cfg.get("label_key") or cfg.get("label") or "y"
        rows, feature_keys, label_key = _load_csv(Path(dataset_path), label_key)
        import random as _random

        rng = _random.Random(split_seed)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        cut = max(1, int(0.7 * len(idx)))
        train = [rows[i] for i in idx[:cut]]
        test = [rows[i] for i in idx[cut:]] or train[-1:]
        from jevtree.data.cube import dataset_hash as _dh

        dataset_hash = cfg.get("dataset_hash") or _dh(rows)
        data_provenance = {
            "loader": "csv",
            "dataset_path": str(dataset_path),
            "n_train": len(train),
            "n_test": len(test),
        }

    ig_names = {"ig_static", "static_ig", "StaticIG"}
    ig_cond_names = {"ig_conditional", "conditional_ig", "ConditionalIG"}
    ig_disc_names = {
        "ig_discriminative",
        "disc_logistic",
        "discriminative",
        "DiscriminativeAcquisition",
    }
    random_names = {"random", "random_acq", "Random"}
    sequential_names = {"sequential", "seq", "Sequential"}
    if policy_name not in (
        ig_names | ig_cond_names | ig_disc_names | random_names | sequential_names
    ):
        print(f"unsupported policy: {policy_name}", file=sys.stderr)
        return 2

    # Prefer policy.predict for self-eval; note GDFS later needs shared external predictor
    has_builtin = policy_cfg.get("has_builtin_classifier")
    if has_builtin is None:
        has_builtin = True
    raw_pred_name = (cfg.get("predictor") or {}).get("name") or "match_majority"
    predictor_name = normalize_predictor_name(raw_pred_name)

    common_kw = dict(
        criterion=policy_cfg.get("criterion", "gain"),
        continuous_keys=continuous_keys or None,
        n_bins=int(policy_cfg.get("n_bins", 4)),
        bin_seed=int(policy_cfg.get("bin_seed", 0)),
        force_acquisition=bool(policy_cfg.get("force_acquisition", True)),
        predictor_name=predictor_name,
    )
    if policy_name in random_names:
        policy = RandomAcquisitionPolicy(
            **common_kw,
            seed=int(policy_cfg.get("seed", 0)),
        )
        policy_name = "random"
    elif policy_name in sequential_names:
        policy = SequentialAcquisitionPolicy(
            **common_kw,
            seed=int(policy_cfg.get("seed", 0)),
        )
        policy_name = "sequential"
    elif policy_name in ig_cond_names:
        policy = ConditionalIGAcquisitionPolicy(
            **common_kw,
            min_samples=int(policy_cfg.get("min_samples", 20)),
            max_candidates=int(policy_cfg.get("max_candidates", 15)),
            seed=int(policy_cfg.get("seed", 0)),
        )
        policy_name = "ig_conditional"
    elif policy_name in ig_disc_names:
        # Default predictor for discriminative is logistic_impute (ELLG proxy).
        if not (cfg.get("predictor") or {}).get("name"):
            predictor_name = "logistic_impute"
            common_kw["predictor_name"] = predictor_name
        policy = DiscriminativeAcquisitionPolicy(
            **common_kw,
            min_samples=int(policy_cfg.get("min_samples", 20)),
            max_candidates=int(policy_cfg.get("max_candidates", 15)),
            seed=int(policy_cfg.get("seed", 0)),
        )
        policy_name = "ig_discriminative"
    else:
        policy = StaticIGAcquisitionPolicy(**common_kw)
        policy_name = "ig_static"
    policy.fit(train, label_key, feature_keys)

    adapter = AFABenchAdapter(
        policy,
        force_acquisition=bool(policy_cfg.get("force_acquisition", True)),
        has_builtin_classifier=bool(has_builtin),
        predictor_name=predictor_name,
    )

    repo = _repo_root()
    from jevtree.eval.provenance import collect_provenance

    prov = collect_provenance(
        repo=repo,
        predictor_name=predictor_name,
        policy_name=policy_name,
        dataset_hash=dataset_hash,
        split_seed=split_seed,
        budgets=hard_budgets,
        config_path=str(cfg_path),
    )
    git_commit = prov.get("git_commit")
    ep_cfg = HardBudgetEpisodeConfig(
        dataset_id=dataset_id,
        hard_budget=hard_budgets[0],
        split_seed=split_seed,
        budget_schedule=hard_budgets,
        config_path=str(cfg_path),
        git_commit=git_commit,
        dataset_hash=dataset_hash,
        policy_name=policy_name,
        feature_keys=feature_keys,
        label_key=label_key,
    )

    protocol = HardBudgetProtocol()
    summary = protocol.run_eval(
        adapter,
        ep_cfg,
        test_rows=test,
        feature_keys=feature_keys,
        label_key=label_key,
        budget_schedule=hard_budgets,
    )

    # provenance extras for real-data / predictor notes
    summary["predictor_name"] = predictor_name
    summary["has_builtin_classifier"] = bool(has_builtin)
    summary["note"] = (
        f"Self-eval uses policy.predict with predictor={predictor_name!r} "
        "(has_builtin_classifier=True). "
        "GDFS / cross-method comparisons later need a shared external predictor."
    )
    summary["data_provenance"] = data_provenance
    summary["n_train"] = len(train)
    summary["n_test"] = len(test)
    summary["n_features"] = len(feature_keys)
    # Required audit provenance (never claim clean scaffold rebuilds a dirty tree)
    summary["git_commit"] = prov.get("git_commit")
    summary["git_dirty"] = bool(prov.get("git_dirty"))
    summary["python_version"] = prov.get("python_version")
    summary["numpy_version"] = prov.get("numpy_version")
    summary["dependency_versions"] = prov.get("dependency_versions") or {}
    summary["budgets"] = list(hard_budgets)
    summary["provenance"] = prov
    if summary.get("git_dirty"):
        print(
            "WARNING: git_dirty=true — working tree has uncommitted changes under "
            "src/configs/tests/pyproject; do not treat this run as a clean-commit rebuild.",
            file=sys.stderr,
        )

    # write artifacts
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{dataset_id}_{policy_name}_{ts}"
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    episodes = summary.pop("episodes", [])
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    episodes_path = run_dir / "episodes.jsonl"
    with open(episodes_path, "w", encoding="utf-8", newline="\n") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False, default=str) + "\n")

    # stdout Acc@budget curve
    print(f"eval-afa  dataset={dataset_id}  policy={policy_name}  n_test={len(test)}")
    print(f"results → {run_dir}")
    print("Acc@budget / F1@budget:")
    for pt in summary.get("curve", []):
        print(
            f"  budget={pt['budget']:>3}  acc={pt['accuracy']:.4f}  f1={pt['f1']:.4f}  n={pt.get('n_episodes')}"
        )
    return 0


def cmd_version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _nyi(name: str, hint: str) -> int:
    print(
        f"jevtree {name}: not implemented yet.\n"
        f"  {hint}\n"
        "  See Feishu roadmap (jevtree folder) for phase gates.",
        file=sys.stderr,
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jevtree",
        description=(
            "jevtree — data + natural-language goal → auditable decision tree / SOP. "
            "Primary entry: jevtree decide --data PATH --goal \"...\"."
        ),
        epilog=(
            "Primary: jevtree decide --data <path> --goal \"NL requirement\". "
            "Aliases: run-job, from-data. Then: jevtree run --sop … --trace. "
            "See README.md and examples/data/SOURCES.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser(
        "decide",
        aliases=["run-job", "from-data"],
        help="ONE entry: --data + --goal → LLM FeatureTable → tree/SOP/story",
    )
    d.add_argument(
        "--data",
        required=True,
        help="CSV/JSON/JSONL/text file, or a directory of texts",
    )
    d.add_argument(
        "--goal",
        required=True,
        help="Natural-language requirement (e.g. 按是否批准贷款做决策树)",
    )
    d.add_argument(
        "--out",
        help="Output directory (default: results/decide_<timestamp>/)",
    )
    d.add_argument("--criterion", default="gain", choices=["gain", "gain_ratio"])
    d.add_argument("--max-depth", type=int, default=4)
    d.add_argument("--min-samples", type=int, default=1)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument(
        "--trace-examples",
        type=int,
        default=0,
        help="Run N example traces and write traces.json",
    )
    d.add_argument(
        "--story",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print bilingual story (default: on)",
    )
    d.set_defaults(func=cmd_decide)

    v = sub.add_parser("validate", help="Validate SOP JSON")
    v.add_argument("sop", nargs="?", help="Path to SOP JSON")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("run", help="Run SOP/tree on one instance or file")
    r.add_argument("--sop", help="Path to SOP / tree")
    r.add_argument("--input", help="Observation JSON or batch file")
    r.add_argument(
        "--trace",
        action="store_true",
        help="Print auditable feature/question path (demo UX)",
    )
    r.set_defaults(func=cmd_run)

    e = sub.add_parser(
        "eval-afa",
        help="AFA hard-budget eval (config-driven)",
    )
    e.add_argument(
        "--config",
        required=False,
        help="Eval JSON/YAML (e.g. configs/eval_cube_hard.json)",
    )
    e.set_defaults(func=cmd_eval_afa)

    ver = sub.add_parser("version", help="Print package version")
    ver.set_defaults(func=cmd_version)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
