"""微桥（tabs）布置的几何单元测试。"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.svggeo import analyze_contours  # noqa: E402
from app.tabs import compute_tabs  # noqa: E402


def _tabs(d, count, width, corner_margin=0.0, fill_rule="nonzero"):
    contours, _ = analyze_contours(d, fill_rule)
    return contours, compute_tabs(contours, count, width, corner_margin)


def _centers(result, contour_pos=0):
    return [t["center_s"] for t in result["contours"][contour_pos]["tabs"]]


# ---------------------------------------------------------------------------
# 扇区中点目标与弧长坐标
# ---------------------------------------------------------------------------

def test_square_four_tabs_at_edge_midpoints():
    # 方 0..40：周长 160，4 扇区中点恰为四边中点
    _, res = _tabs("M0,0h40v40h-40Z", 4, 2.0)
    assert res["count"] == 4
    assert res["width"] == 2.0
    assert res["corner_margin"] == 0.0
    assert res["total_contours"] == 1
    assert res["total_tabs"] == 4
    assert res["total_unplaced"] == 0
    assert res["skipped_contours"] == []

    c0 = res["contours"][0]
    assert c0["original_index"] == 0
    assert c0["role"] == "outer"
    assert c0["depth"] == 0
    assert c0["perimeter"] == 160.0
    assert c0["sector_length"] == 40.0
    assert c0["requested"] == 4
    assert c0["placed"] == 4
    assert c0["unplaced"] == 0

    tabs = c0["tabs"]
    assert [t["sector"] for t in tabs] == [0, 1, 2, 3]
    assert [t["target_s"] for t in tabs] == [20.0, 60.0, 100.0, 140.0]
    assert _centers(res) == [20.0, 60.0, 100.0, 140.0]

    t0 = tabs[0]
    assert t0["start_s"] == 19.0 and t0["end_s"] == 21.0
    assert t0["center"] == {"x": 20.0, "y": 0.0}
    assert t0["start"] == {"x": 19.0, "y": 0.0}
    assert t0["end"] == {"x": 21.0, "y": 0.0}

    t1 = tabs[1]
    assert t1["center"] == {"x": 40.0, "y": 20.0}
    assert t1["start"] == {"x": 40.0, "y": 19.0}
    assert t1["end"] == {"x": 40.0, "y": 21.0}

    t2 = tabs[2]
    assert t2["center"] == {"x": 20.0, "y": 40.0}
    assert t2["start"] == {"x": 21.0, "y": 40.0}
    assert t2["end"] == {"x": 19.0, "y": 40.0}

    t3 = tabs[3]
    assert t3["center"] == {"x": 0.0, "y": 20.0}
    assert t3["start"] == {"x": 0.0, "y": 21.0}
    assert t3["end"] == {"x": 0.0, "y": 19.0}


def test_target_on_vertex_tie_breaks_to_smaller_arc():
    # 周长 160、2 扇区：目标弧长 40 / 120 恰为顶点 (40,0) 与 (0,40)，
    # 两侧可行中心（39 / 41、119 / 121）等距，取较小弧长
    _, res = _tabs("M0,0h40v40h-40Z", 2, 2.0)
    assert _centers(res) == [39.0, 119.0]
    tabs = res["contours"][0]["tabs"]
    assert tabs[0]["center"] == {"x": 39.0, "y": 0.0}
    assert tabs[0]["start"] == {"x": 38.0, "y": 0.0}
    assert tabs[0]["end"] == {"x": 40.0, "y": 0.0}  # 桥端恰抵顶点（余量为 0）
    assert tabs[1]["center"] == {"x": 1.0, "y": 40.0}
    assert tabs[1]["end"] == {"x": 0.0, "y": 40.0}


def test_corner_margin_pushes_center_away_from_vertex():
    # 余量 5：顶点两侧各 5 弧长不可跨，目标 40 两侧可行端点为 34 / 46，
    # 等距取较小弧长 34
    _, res = _tabs("M0,0h40v40h-40Z", 2, 2.0, corner_margin=5.0)
    assert _centers(res) == [34.0, 114.0]
    tabs = res["contours"][0]["tabs"]
    assert tabs[0]["center"] == {"x": 34.0, "y": 0.0}
    assert tabs[0]["end"] == {"x": 35.0, "y": 0.0}  # 桥端恰抵余量区边界 40-5
    assert tabs[1]["center"] == {"x": 6.0, "y": 40.0}


def test_bridge_touching_margin_boundary_is_feasible():
    # 边长 10、余量 4、桥宽 2：可行中心收缩为每边中点单点，恰抵边界可行
    _, res = _tabs("M0,0h10v10h-10Z", 4, 2.0, corner_margin=4.0)
    assert res["total_unplaced"] == 0
    assert _centers(res) == [5.0, 15.0, 25.0, 35.0]
    tabs = res["contours"][0]["tabs"]
    assert tabs[0]["start"] == {"x": 4.0, "y": 0.0}
    assert tabs[0]["end"] == {"x": 6.0, "y": 0.0}


def test_margin_too_large_blocks_everything():
    # 余量 19.5 + 半桥宽 1 > 半边长 20：所有边跨无可行中心
    _, res = _tabs("M0,0h40v40h-40Z", 2, 2.0, corner_margin=19.5)
    c0 = res["contours"][0]
    assert c0["placed"] == 0
    assert c0["unplaced"] == 2
    assert c0["tabs"] == []
    assert res["total_tabs"] == 0
    assert res["total_unplaced"] == 2


def test_width_exactly_fits_edge_single_point():
    # 边长 10、桥宽 10：可行中心恰为边中点单点（桥端抵两侧顶点）
    _, res = _tabs("M0,0h10v10h-10Z", 4, 10.0)
    assert _centers(res) == [5.0, 15.0, 25.0, 35.0]
    tabs = res["contours"][0]["tabs"]
    assert tabs[0]["start"] == {"x": 0.0, "y": 0.0}
    assert tabs[0]["end"] == {"x": 10.0, "y": 0.0}
    assert tabs[3]["center"] == {"x": 0.0, "y": 5.0}


# ---------------------------------------------------------------------------
# 互不重叠与未能放置
# ---------------------------------------------------------------------------

def test_unplaced_count_reported_when_width_too_large():
    # 周长 160、8 扇区、桥宽 30：每边可行区间仅 [15,25] 等，
    # 放置一桥后相邻扇区被排斥区覆盖 -> 隔扇区放置，4 放 4 未放
    _, res = _tabs("M0,0h40v40h-40Z", 8, 30.0)
    c0 = res["contours"][0]
    assert c0["placed"] == 4
    assert c0["unplaced"] == 4
    assert res["total_tabs"] == 4
    assert res["total_unplaced"] == 4
    assert [t["sector"] for t in c0["tabs"]] == [0, 2, 4, 6]
    assert _centers(res) == [15.0, 55.0, 95.0, 135.0]
    tabs = c0["tabs"]
    assert tabs[0]["center"] == {"x": 15.0, "y": 0.0}
    assert tabs[1]["center"] == {"x": 40.0, "y": 15.0}
    assert tabs[2]["center"] == {"x": 25.0, "y": 40.0}
    assert tabs[3]["center"] == {"x": 0.0, "y": 25.0}


def test_exclusion_zone_pushes_later_sector_center():
    # 矩形 40x10（周长 100）、5 扇区、桥宽 15：
    # 扇区 3 目标 70 落在上一桥（57.5）的排斥区内，被推到 ≈72.5
    _, res = _tabs("M0,0h40v10h-40Z", 5, 15.0)
    c0 = res["contours"][0]
    assert c0["placed"] == 4
    assert c0["unplaced"] == 1  # 扇区 4 的可行域被排斥区耗尽
    centers = _centers(res)
    assert centers[0] == 10.0
    assert centers[1] == 30.0
    assert centers[2] == 57.5
    assert centers[3] == pytest.approx(72.5, abs=1e-6)
    # 恰好相接（距离 == width，容差内）视为不重叠
    assert centers[3] - centers[2] == pytest.approx(15.0, abs=1e-6)


def test_tabs_never_overlap_and_stay_in_sector():
    # 一般性不变量：中心落在各自扇区内，且两两环向距离 >= width（容差内）
    d = "M0,0h30v10h-30Z"
    count, width = 4, 12.0
    _, res = _tabs(d, count, width)
    c0 = res["contours"][0]
    perimeter = c0["perimeter"]
    sector = perimeter / count
    centers = _centers(res)
    assert centers == [10.0, 24.0, 50.0, 64.0]
    for t in c0["tabs"]:
        assert t["sector"] * sector <= t["center_s"] <= (t["sector"] + 1) * sector
    for i, ci in enumerate(centers):
        for cj in centers[i + 1:]:
            gap = min(abs(ci - cj), perimeter - abs(ci - cj))
            assert gap >= width - 1e-6


# ---------------------------------------------------------------------------
# 参与/跳过规则
# ---------------------------------------------------------------------------

def test_skips_hole_open_and_degenerate():
    d = ("M0,0h40v40h-40Z "          # 0: outer
         "M10,10h20v20h-20Z "        # 1: hole
         "M5,5 L8,5 L5,5 Z "         # 2: degenerate
         "M50,0 L60,0 L60,10")       # 3: open
    _, res = _tabs(d, 2, 1.0)
    assert res["total_contours"] == 1
    assert [c["original_index"] for c in res["contours"]] == [0]
    skipped = {(s["original_index"], s["role"], s["reason"])
               for s in res["skipped_contours"]}
    assert skipped == {(1, "hole", "hole"),
                       (2, "degenerate", "degenerate"),
                       (3, "open", "open")}


def test_island_outer_participates_hole_skipped():
    # 外环 -> 孔 -> 孔中岛：岛（深度 2）也是 outer，参与布置
    d = "M0,0h100v100h-100Z M10,10h80v80h-80Z M40,40h20v20h-20Z"
    _, res = _tabs(d, 2, 1.0)
    assert res["total_contours"] == 2
    assert [c["original_index"] for c in res["contours"]] == [0, 2]
    assert [c["role"] for c in res["contours"]] == ["outer", "outer"]
    assert [c["depth"] for c in res["contours"]] == [0, 2]
    assert res["skipped_contours"] == [
        {"original_index": 1, "role": "hole", "reason": "hole"}]
    assert res["total_tabs"] == 4
    # 岛：周长 80，两扇区中点 20 / 60 恰为顶点，宽 1 时退半宽，等距取小弧长
    island = res["contours"][1]
    assert island["perimeter"] == 80.0
    assert [t["center_s"] for t in island["tabs"]] == [19.5, 59.5]


def test_two_independent_outers_processed_by_original_index():
    _, res = _tabs("M0,0h10v10h-10Z M100,0h10v10h-10Z", 2, 1.0)
    assert res["total_contours"] == 2
    assert [c["original_index"] for c in res["contours"]] == [0, 1]
    # 周长 40、2 扇区：目标 10 / 30 落在顶点，退半宽后取小弧长 9.5 / 29.5
    assert _centers(res, 0) == [9.5, 29.5]
    assert _centers(res, 1) == [9.5, 29.5]
    # 第二外环平移 100：中心坐标随之平移
    assert res["contours"][1]["tabs"][0]["center"] == {"x": 109.5, "y": 0.0}
    assert res["total_tabs"] == 4


def test_curve_contour_tabs_on_flattened_polyline():
    # 单位圆（两条半圆弧细分）：微桥落在细分折线上，中心距原点 ≈ 半径
    d = "M-1,0 A1,1 0 1,0 1,0 A1,1 0 1,0 -1,0 Z"
    _, res = _tabs(d, 4, 0.1, corner_margin=0.01)
    c0 = res["contours"][0]
    assert c0["placed"] == 4
    assert c0["unplaced"] == 0
    # 细分折线周长略小于真实圆周（弦近似）
    assert 6.2 < c0["perimeter"] < 2 * math.pi
    centers = [t["center_s"] for t in c0["tabs"]]
    assert centers == sorted(centers)  # 扇区顺序 -> 弧长递增
    for t in c0["tabs"]:
        assert t["end_s"] - t["start_s"] == pytest.approx(0.1)
        r = math.hypot(t["center"]["x"], t["center"]["y"])
        assert 0.99 < r <= 1.0 + 1e-9
        assert 0.0 <= t["start_s"] < t["center_s"] < t["end_s"] <= c0["perimeter"]


# ---------------------------------------------------------------------------
# 空结果 / 确定性 / 填充规则
# ---------------------------------------------------------------------------

def test_empty_path_returns_empty_result():
    _, res = _tabs("", 3, 1.0)
    assert res["total_contours"] == 0
    assert res["contours"] == []
    assert res["total_tabs"] == 0
    assert res["total_unplaced"] == 0
    assert res["skipped_contours"] == []


def test_only_open_contour_returns_empty_result():
    _, res = _tabs("M0,0 L10,0 L10,10", 3, 1.0)
    assert res["total_contours"] == 0
    assert res["contours"] == []
    assert res["skipped_contours"] == [
        {"original_index": 0, "role": "open", "reason": "open"}]


def test_tabs_deterministic():
    d = ("M0,0h100v100h-100Z M10,10h80v80h-80Z M40,40h20v20h-20Z "
         "M200,0 Q210,10 200,20 T200,40 Z")
    contours, _ = analyze_contours(d, "nonzero")
    r1 = compute_tabs(contours, 5, 2.0, 0.5)
    r2 = compute_tabs(contours, 5, 2.0, 0.5)
    assert r1 == r2


def test_tabs_same_for_both_fill_rules():
    d = "M0,0h40v40h-40Z M10,10h20v20h-20Z"
    _, r_nonzero = _tabs(d, 4, 2.0, 0.5, "nonzero")
    _, r_evenodd = _tabs(d, 4, 2.0, 0.5, "evenodd")
    assert r_nonzero == r_evenodd


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------

def test_invalid_params_raise_value_error():
    contours, _ = analyze_contours("M0,0h10v10h-10Z", "nonzero")
    for bad_count in (0, -1, 2.5, "3", True, None):
        with pytest.raises(ValueError):
            compute_tabs(contours, bad_count, 1.0, 0.0)
    for bad_width in (0, -1.0, float("nan"), float("inf"), "x"):
        with pytest.raises(ValueError):
            compute_tabs(contours, 2, bad_width, 0.0)
    for bad_margin in (-0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            compute_tabs(contours, 2, 1.0, bad_margin)
