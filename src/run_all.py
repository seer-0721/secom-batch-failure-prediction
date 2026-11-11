"""SECOM 半导体批次缺陷预测 —— 端到端实验。

评估协议（核心）:
  A. 主协议: expanding-window walk-forward（6 折滚动），每折内部重跑清洗/剔冗余/排序，
     只用该折训练段估计统计量；合并折外预测算指标。
  B. 对照协议 1: sliding-window（W=350）重训练，检验滑窗能否跟上漂移。
  C. 对照协议 2: 单块远期留出（最后 30%），作为漂移压力测试。
  D. 对照协议 3: 随机分层切分（10 次重复），量化切分方式带来的分数虚高。

选传感器数 k 与决策阈值都只用 walk-forward 的前三折，后三折只用于报告最终指标。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plots
from config import OUT_DIR, SEED
from evaluate import at_threshold, bootstrap_ci, cost_optimal, pick_threshold
from prepare import Cleaner, load_raw
from walkforward import _fit_one_fold, run_regime, slice_metrics

KS = (20, 50)
N_SPLITS = 6
SLIDING_W = 350


def random_split_control(xs, y, n_repeats: int = 10) -> dict:
    aucs, aps = [], []
    for seed in range(n_repeats):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED + seed)
        i_tr, i_te = next(sss.split(np.zeros(len(y)), y))
        c = Cleaner().fit(xs.iloc[i_tr])
        z_tr, z_te = c.transform(xs.iloc[i_tr]), c.transform(xs.iloc[i_te])
        rf = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", random_state=SEED + seed, n_jobs=-1,
        ).fit(z_tr, y[i_tr])
        p = rf.predict_proba(z_te)[:, 1]
        aucs.append(roc_auc_score(y[i_te], p))
        aps.append(average_precision_score(y[i_te], p))
    return {
        "n_repeats": n_repeats,
        "note": "随机分层 70/30，全部 445 个传感器，RandomForest；同清洗流程",
        "mean": {"roc_auc": float(np.mean(aucs)), "pr_auc": float(np.mean(aps))},
        "std": {"roc_auc": float(np.std(aucs)), "pr_auc": float(np.std(aps))},
        "runs": [{"roc_auc": float(a), "pr_auc": float(b)} for a, b in zip(aucs, aps)],
    }


def main() -> dict:
    t0 = time.time()
    xs, y, ts = load_raw()
    n = len(y)
    print(f"[data] {n} 批次 / {xs.shape[1]} 传感器 / 缺陷 {int(y.sum())} ({y.mean():.2%})")

    print("[wf  ] expanding-window walk-forward ...")
    res_exp = run_regime(xs, y, n_splits=N_SPLITS, window=None, ks=KS)
    print("[wf  ] sliding-window walk-forward ...")
    res_sli = run_regime(xs, y, n_splits=N_SPLITS, window=SLIDING_W, ks=KS)

    sl = res_exp["fold_slices"]
    early = (sl[0][0], sl[2][1])
    late = (sl[3][0], sl[-1][1])
    print(f"[split] 早期块 {early} / 后期块 {late} (共 {n} 批)")

    # ---- k 与阈值只用前三折确定 ----
    early_by_k = {k: slice_metrics(y, res_exp["oof"][k], *early) for k in KS}
    k_star = max(KS, key=lambda k: early_by_k[k]["pr_auc"])
    print("[pick] 早期块 PR-AUC: " + " | ".join(f"k={k}:{v['pr_auc']:.4f}" for k, v in early_by_k.items())
          + f"  -> k*={k_star}")

    p_all = res_exp["oof"][k_star]
    y_early = y[early[0]:early[1]]
    p_early = p_all[early[0]:early[1]]
    thr, f2_early = pick_threshold(y_early, p_early, beta=2.0)
    print(f"[thr ] 早期块 F2 最优阈值 = {thr:.4f} (F2={f2_early:.4f})")

    # ---- 后期块报告 ----
    y_late = y[late[0]:late[1]]
    p_late = p_all[late[0]:late[1]]
    late_metrics = slice_metrics(y, p_all, *late)
    late_at_thr = at_threshold(y_late, p_late, thr)
    ci_ap = bootstrap_ci(y_late, p_late, "pr_auc", seed=SEED)
    ci_auc = bootstrap_ci(y_late, p_late, "roc_auc", seed=SEED)
    print(f"[late] PR-AUC={late_metrics['pr_auc']:.4f} {ci_ap} ROC-AUC={late_metrics['roc_auc']:.4f} {ci_auc}")
    print(f"[late] @thr={thr:.3f} P={late_at_thr['precision']:.3f} R={late_at_thr['recall']:.3f} "
          f"漏检={late_at_thr['fn']}/{late_metrics['n_fail']} 误报={late_at_thr['fp']}")

    # ---- 对照协议 C: 单块远期留出 ----
    cut = int(round(n * 0.70))
    i_tr, i_te = np.arange(cut), np.arange(cut, n)
    hold = _fit_one_fold(xs, y, i_tr, i_te, (k_star,), seed=SEED)
    p_hold = hold[k_star]["p"]
    hold_stats = {
        "protocol": f"single holdout: train 前 {cut} 批 / test 后 {n - cut} 批",
        "roc_auc": float(roc_auc_score(y[i_te], p_hold)),
        "pr_auc": float(average_precision_score(y[i_te], p_hold)),
        "n_test": int(len(i_te)), "n_test_fail": int(y[i_te].sum()),
    }
    print(f"[hold] ROC-AUC={hold_stats['roc_auc']:.4f} PR-AUC={hold_stats['pr_auc']:.4f}")

    # ---- 对照协议 D: 随机切分 ----
    rand_stats = random_split_control(xs, y)
    print(f"[rand] ROC-AUC={rand_stats['mean']['roc_auc']:.4f}±{rand_stats['std']['roc_auc']:.4f}")

    # ---- 代价敏感分析（后期块）----
    ratios = [1, 2, 5, 10, 20, 50, 100]
    cost_rows = []
    for r in ratios:
        thr_r, _ = cost_optimal(y_early, p_early, r)
        fn = int(((p_late < thr_r) & (y_late == 1)).sum())
        fp = int(((p_late >= thr_r) & (y_late == 0)).sum())
        fn0 = int(((p_late < 0.5) & (y_late == 1)).sum())
        fp0 = int(((p_late >= 0.5) & (y_late == 0)).sum())
        c1 = (r * fn + fp) / len(y_late)
        c0 = (r * fn0 + fp0) / len(y_late)
        cost_rows.append({
            "ratio": r, "threshold": thr_r, "fn": fn, "fp": fp,
            "cost_per_batch": float(c1), "cost_at_0.5": float(c0),
            "reduction_pct": float((c0 - c1) / c0 * 100) if c0 > 0 else 0.0,
        })
    print("[cost] " + " | ".join(f"r={c['ratio']}:{c['cost_per_batch']:.3f}" for c in cost_rows))

    # ---- 图 ----
    figs = [
        plots.data_overview(ts, y, sl),
        plots.fold_performance(res_exp, res_sli, k_star),
        plots.roc_pr_curves(y_late, p_late, f"late blocks 4-6, k={k_star}"),
        plots.threshold_tradeoff(y_late, p_late, thr),
        plots.protocol_comparison({
            "rand": rand_stats["mean"],
            "hold": hold_stats,
            "exp": res_exp["per_k"][k_star],
            "sli": res_sli["per_k"][k_star],
        }),
        plots.sensor_stability(res_exp["sensor_frequency"][str(k_star)], n_folds=N_SPLITS),
        plots.cost_curve([c["ratio"] for c in cost_rows],
                         [c["cost_per_batch"] for c in cost_rows],
                         [c["cost_at_0.5"] for c in cost_rows]),
    ]

    metrics = {
        "dataset": {
            "batches": int(n), "sensors_raw": int(xs.shape[1]),
            "fail_batches": int(y.sum()), "fail_rate": float(y.mean()),
            "time_range": [str(ts.iloc[0].date()), str(ts.iloc[-1].date())],
            "missing_rate": float(xs.isna().to_numpy().mean()),
        },
        "screening": {
            "dropped_high_missing_gt_50pct": 24,
            "dropped_constant": 122,
            "after_cleaning": 445,
            "note": "相关性剔除(>0.95)与重要性排序在每一折内重新做，数值见 folds[].n_sensors_kept",
            "candidate_ks": list(KS),
            "selected_k": int(k_star),
            "selection_rule": "早期三折折外 PR-AUC 最大",
            "early_block_pr_auc": {str(k): v["pr_auc"] for k, v in early_by_k.items()},
        },
        "protocol": {
            "primary": f"expanding-window walk-forward, {N_SPLITS} folds",
            "control_regimes": [f"sliding window W={SLIDING_W}", "single far holdout last 30%",
                                "stratified random 70/30 x10"],
            "per_fold_pipeline": "中位数填补/标准化/相关性剔除/RF 重要性排序 全部只用该折训练段",
            "threshold": "argmax F2 on early blocks (folds 1-3) out-of-fold predictions",
            "imbalance_handling": "class_weight only; 不做切分前重采样",
            "early_blocks": list(early), "late_blocks": list(late),
        },
        "walkforward": {
            "expanding": {"folds": res_exp["folds"], "per_k": {str(k): v for k, v in res_exp["per_k"].items()},
                          "sensor_frequency": res_exp["sensor_frequency"]},
            "sliding": {"folds": res_sli["folds"], "per_k": {str(k): v for k, v in res_sli["per_k"].items()}},
        },
        "final_model": {
            "k": int(k_star), "threshold": thr, "oof_f2_early": f2_early,
            "late_metrics": late_metrics,
            "late_at_threshold": late_at_thr,
            "late_ci95": {"pr_auc": list(ci_ap), "roc_auc": list(ci_auc)},
        },
        "control_protocols": {
            "far_holdout": hold_stats,
            "random_split": rand_stats,
            "expanding_pooled": res_exp["per_k"][k_star],
            "sliding_pooled": res_sli["per_k"][k_star],
        },
        "cost_sensitivity": cost_rows,
        "figures": figs,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] metrics.json  ({metrics['elapsed_sec']}s)")
    return metrics


if __name__ == "__main__":
    main()
