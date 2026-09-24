"""微桥（tabs）布置：闭合外环轮廓上的弧长等分扇区选点（只读即时计算）。

对一份已存 SVG path 即时解析并完成轮廓包含层级分类后（与
metrics / clearance / scanlines / cut-plan 同一份解析与曲线细分折线，见
:func:`app.svggeo.analyze_contours`），仅对**闭合且非退化的外环
（outer）**轮廓沿细分折线建立首尾相接的弧长坐标 ``s ∈ [0, L)``
（``L`` 为轮廓周长），按以下规则布置 ``count`` 条微桥：

- 周长等分为 ``count`` 个扇区，第 ``j`` 个扇区为
  ``[j*L/count, (j+1)*L/count)``，以扇区**中点**为桥段中心的目标弧长；
- 桥宽 ``width``：中心位于弧长 ``c`` 的桥段占据弧长区间
  ``[c - width/2, c + width/2]``；
- **拐角余量**：每个细分顶点两侧各 ``corner_margin`` 弧长为余量区，
  桥段不得跨入（桥端恰好落在余量区边界视为可行）；
- **互不重叠**：两桥中心的环向弧长距离不得小于 ``width``
  （恰好相接视为可行，含 1e-9 相对容差以吸收浮点误差）；
- 按**轮廓原始索引**、再按**扇区顺序**依次布置：在扇区内的可行
  桥中心中选取距目标最近者，距离相同时取较小弧长；每放置一桥
  立即从可行域扣除其排斥区（中心两侧各 ``width`` 的环向邻域），
  影响后续扇区的选择；
- 扇区内不存在可行桥中心时，该桥计入**未能放置**并继续下一扇区。

孔洞（hole）、开放（open）与退化（degenerate）轮廓跳过并列入
``skipped_contours``。整个计算只读、即时完成：不依赖已保存的
规范化结果，也不写库。
"""

from __future__ import annotations

import bisect
import math
from typing import Any

from .svggeo import Contour

# 输出量化位数（仅影响 JSON 展示，不改变选点判定）
_DECIMALS = 9
# 可行性/重叠判定的相对容差：保证“桥端恰落在余量区边界、两桥恰好相接”可行
_EPS_REL = 1e-9


def compute_tabs(contours: list[Contour],
                 count: int,
                 width: float,
                 corner_margin: float) -> dict[str, Any]:
    """对已分类的轮廓列表在闭合外环上布置微桥。

    参数:
        contours: :func:`app.svggeo.analyze_contours` 返回的轮廓
            （已带 ``role`` / ``depth``）。
        count: 每条外环轮廓的微桥数量（> 0 的整数）。
        width: 桥宽（> 0 的有限数）。
        corner_margin: 顶点两侧拐角余量（≥ 0 的有限数）。

    返回:
        ``{count, width, corner_margin, total_contours, total_tabs,
        total_unplaced, contours, skipped_contours}``；``contours``
        每项含轮廓 ``original_index`` / ``role`` / ``depth``、周长、
        扇区长、请求/已放置/未放置数量与逐桥段的起点、终点、中心
        坐标及对应弧长。

    抛出:
        ValueError: 参数非法（count 非正整数、width 非正或非有限、
            corner_margin 为负或非有限）。
    """
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count 必须是大于 0 的整数")
    if not (isinstance(width, (int, float))
            and math.isfinite(width) and width > 0):
        raise ValueError("width 必须是大于 0 的有限数")
    if not (isinstance(corner_margin, (int, float))
            and math.isfinite(corner_margin) and corner_margin >= 0):
        raise ValueError("corner_margin 必须是非负的有限数")

    outers = [c for c in contours if c.role == "outer"]
    outers.sort(key=lambda c: c.original_index)

    skipped = [
        {"original_index": c.original_index, "role": c.role,
         "reason": c.role}
        for c in sorted(contours, key=lambda c: c.original_index)
        if c.role != "outer"
    ]

    items: list[dict[str, Any]] = []
    total_tabs = 0
    total_unplaced = 0
    for c in outers:
        item = _contour_tabs(c, count, float(width), float(corner_margin))
        total_tabs += item["placed"]
        total_unplaced += item["unplaced"]
        items.append(item)

    return {
        "count": count,
        "width": _num(float(width)),
        "corner_margin": _num(float(corner_margin)),
        "total_contours": len(outers),
        "total_tabs": total_tabs,
        "total_unplaced": total_unplaced,
        "contours": items,
        "skipped_contours": skipped,
    }


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _contour_tabs(c: Contour,
                  count: int,
                  width: float,
                  corner_margin: float) -> dict[str, Any]:
    """在单个闭合外环上按扇区顺序布置 ``count`` 条微桥。"""
    pts = c.points
    n = len(pts)
    xs = [p[0] for p in pts] + [pts[0][0]]
    ys = [p[1] for p in pts] + [pts[0][1]]

    # 首尾相接的弧长坐标：cum[k] 为第 k 个顶点的弧长，cum[n] 为周长
    cum = [0.0] * (n + 1)
    for k in range(n):
        cum[k + 1] = cum[k] + math.hypot(xs[k + 1] - xs[k],
                                         ys[k + 1] - ys[k])
    perimeter = cum[n]
    half = width / 2.0
    tol = _EPS_REL * max(1.0, perimeter, width, corner_margin)

    # 可行桥中心区间：每条边跨 (cum[k], cum[k+1]) 上，桥两端分别避开
    # 两侧顶点的余量区后剩余的中心活动范围（恰抵余量区边界可行）
    feasible: list[tuple[float, float]] = []
    for k in range(n):
        lo = cum[k] + corner_margin + half
        hi = cum[k + 1] - corner_margin - half
        if hi >= lo - tol:
            feasible.append((lo, max(lo, hi)))

    sector = perimeter / count
    tabs: list[dict[str, Any]] = []
    unplaced = 0

    for j in range(count):
        a = j * sector
        # 末扇区右端点直接取周长，消除 sector 累乘的浮点尾差
        b = perimeter if j == count - 1 else a + sector
        target = (j + 0.5) * sector

        # 扇区内距目标最近的可行中心；距离相同取较小弧长
        best: tuple[float, float] | None = None
        for lo, hi in feasible:
            lo2 = lo if lo > a else a
            hi2 = hi if hi < b else b
            if lo2 > hi2:
                continue
            c_star = target if lo2 <= target <= hi2 else (
                lo2 if target < lo2 else hi2)
            cand = (abs(c_star - target), c_star)
            if best is None or cand < best:
                best = cand

        if best is None:
            unplaced += 1
            continue

        center_s = best[1]
        start_s = center_s - half
        end_s = center_s + half
        cx, cy = _point_at(xs, ys, cum, center_s)
        sx, sy = _point_at(xs, ys, cum, start_s)
        ex, ey = _point_at(xs, ys, cum, end_s)
        tabs.append({
            "sector": j,
            "target_s": _num(target),
            "center_s": _num(center_s),
            "start_s": _num(start_s),
            "end_s": _num(end_s),
            "center": {"x": _num(cx), "y": _num(cy)},
            "start": {"x": _num(sx), "y": _num(sy)},
            "end": {"x": _num(ex), "y": _num(ey)},
        })

        # 扣除本桥的排斥区（中心两侧各 width 的环向邻域），后续扇区不得重叠
        feasible = _subtract_exclusion(feasible, center_s, width,
                                       perimeter, tol)

    return {
        "original_index": c.original_index,
        "role": c.role,
        "depth": c.depth,
        "perimeter": _num(perimeter),
        "sector_length": _num(sector),
        "requested": count,
        "placed": len(tabs),
        "unplaced": unplaced,
        "tabs": tabs,
    }


def _subtract_exclusion(feasible: list[tuple[float, float]],
                        center: float,
                        width: float,
                        perimeter: float,
                        tol: float) -> list[tuple[float, float]]:
    """从可行中心区间中扣除已放置桥的排斥区（返回新列表，保持升序）。

    排斥区为环向弧长距离 < ``width`` 的中心范围（两端各收缩 ``tol``，
    使恰好相接、距离 == ``width`` 的中心保持可行）；排斥区跨过
    0 / 周长边界时按环形补另一端。
    """
    radius = width - tol
    if radius <= 0.0:
        return feasible
    spans = [(center - radius, center + radius)]
    if center - radius < 0.0:
        spans.append((center - radius + perimeter, perimeter))
    if center + radius > perimeter:
        spans.append((0.0, center + radius - perimeter))

    out: list[tuple[float, float]] = []
    for lo, hi in feasible:
        pieces = [(lo, hi)]
        for elo, ehi in spans:
            nxt: list[tuple[float, float]] = []
            for plo, phi in pieces:
                if ehi <= plo or elo >= phi:
                    nxt.append((plo, phi))
                    continue
                if elo > plo:
                    nxt.append((plo, min(elo, phi)))
                if ehi < phi:
                    nxt.append((max(ehi, plo), phi))
            pieces = nxt
        out.extend(pieces)
    return out


def _point_at(xs: list[float],
              ys: list[float],
              cum: list[float],
              s: float) -> tuple[float, float]:
    """闭合折线上弧长 ``s`` 处的插值点（``s`` 截断到 ``[0, 周长]``）。"""
    perimeter = cum[-1]
    if s <= 0.0:
        s = 0.0
    elif s >= perimeter:
        s = perimeter
    k = bisect.bisect_right(cum, s) - 1
    if k >= len(xs) - 1:
        k = len(xs) - 2
    seg = cum[k + 1] - cum[k]
    t = 0.0 if seg <= 0.0 else (s - cum[k]) / seg
    return (xs[k] + t * (xs[k + 1] - xs[k]),
            ys[k] + t * (ys[k + 1] - ys[k]))


def _num(v: float) -> float:
    """量化展示值并消除 -0.0。"""
    r = round(v, _DECIMALS)
    return r + 0.0 if r == 0 else r
