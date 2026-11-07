from typing import Dict

import numpy as np


def compute_av_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    artery_label: int = 1,
    vein_label: int = 2,
) -> Dict[str, float]:
    """计算已对齐、同编码标签图的动静脉评价指标。

    所有比率返回 0–1 数值；分母为零时返回 NaN。
    空间几何一致性应由调用方在读取影像时检查。
    """
    gt = np.asarray(gt)
    pred = np.asarray(pred)

    if gt.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: GT {gt.shape}, prediction {pred.shape}"
        )
    if artery_label == vein_label:
        raise ValueError("Artery and vein labels must differ.")

    ga = gt == artery_label
    gv = gt == vein_label
    pa = pred == artery_label
    pv = pred == vein_label

    g_vessel = ga | gv
    p_vessel = pa | pv

    def count(mask):
        return int(np.count_nonzero(mask))

    def ratio(numerator, denominator):
        return (
            float(numerator / denominator)
            if denominator > 0
            else float("nan")
        )

    n_ga = count(ga)
    n_gv = count(gv)
    n_pa = count(pa)
    n_pv = count(pv)

    # 行是真实类别，列是预测类别。
    a_to_a = count(ga & pa)
    a_to_v = count(ga & pv)
    a_to_other = count(ga & ~p_vessel)

    v_to_a = count(gv & pa)
    v_to_v = count(gv & pv)
    v_to_other = count(gv & ~p_vessel)

    # Other 包括背景、气道等所有非动静脉类别。
    other_to_a = count(~g_vessel & pa)
    other_to_v = count(~g_vessel & pv)

    n_gt_vessel = n_ga + n_gv
    n_pred_vessel = n_pa + n_pv
    vessel_tp = count(g_vessel & p_vessel)
    cross_errors = a_to_v + v_to_a

    return {
        # HiPaS 文中公式的集合计数口径；没有额外的系数 2。
        "av_mcs": ratio(
            cross_errors, n_gt_vessel + n_pred_vessel
        ),

        # 真实动脉/静脉中，被误判为另一类的比例。
        "artery_to_vein_rate": ratio(a_to_v, n_ga),
        "vein_to_artery_rate": ratio(v_to_a, n_gv),

        # 合并 A/V 后评价血管检出和重叠。
        "merged_vessel_dice": ratio(
            2 * vessel_tp, n_gt_vessel + n_pred_vessel
        ),
        "merged_vessel_precision": ratio(
            vessel_tp, n_pred_vessel
        ),
        "merged_vessel_recall": ratio(
            vessel_tp, n_gt_vessel
        ),

        # 保存原始计数，支持结果核查。
        "gt_artery_voxels": n_ga,
        "gt_vein_voxels": n_gv,
        "pred_artery_voxels": n_pa,
        "pred_vein_voxels": n_pv,
        "a_to_a_voxels": a_to_a,
        "a_to_v_voxels": a_to_v,
        "a_to_other_voxels": a_to_other,
        "v_to_a_voxels": v_to_a,
        "v_to_v_voxels": v_to_v,
        "v_to_other_voxels": v_to_other,
        "other_to_a_voxels": other_to_a,
        "other_to_v_voxels": other_to_v,
    }


def test_av_metrics():
    gt = np.array([1, 1, 2, 2, 0, 3], dtype=np.uint8)

    # 完美预测。
    m = compute_av_metrics(gt, gt.copy())
    assert m["av_mcs"] == 0
    assert m["merged_vessel_dice"] == 1

    # 动静脉全部互换：血管位置完全正确，但类别全部错。
    swapped = np.array([2, 2, 1, 1, 0, 3], dtype=np.uint8)
    m = compute_av_metrics(gt, swapped)
    assert m["av_mcs"] == 0.5  # 当前公式的归一化口径
    assert m["artery_to_vein_rate"] == 1
    assert m["vein_to_artery_rate"] == 1
    assert m["merged_vessel_dice"] == 1

    # 全部漏成背景：错分指标为零，但血管召回为零。
    m = compute_av_metrics(gt, np.zeros_like(gt))
    assert m["av_mcs"] == 0
    assert m["merged_vessel_recall"] == 0
    assert m["merged_vessel_dice"] == 0
    assert np.isnan(m["merged_vessel_precision"])

    # 一半动脉误分为静脉，其余血管类别正确。
    pred = np.array([1, 2, 2, 2, 0, 3], dtype=np.uint8)
    m = compute_av_metrics(gt, pred)
    assert m["artery_to_vein_rate"] == 0.5
    assert m["vein_to_artery_rate"] == 0
    assert m["av_mcs"] == 0.125

    # 无真实静脉：V→A 不可定义，应返回 NaN。
    m = compute_av_metrics(
        np.array([1, 0]), np.array([1, 0])
    )
    assert np.isnan(m["vein_to_artery_rate"])

    print("AV metric tests passed.")


if __name__ == "__main__":
    test_av_metrics()
