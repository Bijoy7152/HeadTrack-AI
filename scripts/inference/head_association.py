#!/usr/bin/env python3
"""
Head-to-Player Association (Phase 3, section 3.7)

Determines which detected head belongs to which tracked player, for a
single frame. Pure geometry, no video/model dependency, so it is
unit-testable in isolation.

Association criteria (all folded into one derived confidence score, not an
arbitrary constant):
  - containment: does the player's box contain the head center?
  - upper-body priority: is the head center within the top fraction of the
    player's box (upper_fraction), where a real head should sit?
  - spatial distance: how close is the head to the player's head anchor
    (top-center of the player box), normalized by the player's box size so
    the score is scale-invariant across near/far players.

A head that is not contained by any player box may still be associated to
the nearest player within a distance gate (handles head boxes that slightly
overflow the player box), but at a lower confidence than a contained match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class HeadBox:
    head_id: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass
class TrackedPlayer:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)

    @property
    def head_anchor(self) -> tuple[float, float]:
        """Top-center of the player box: where a head should be."""
        return ((self.x1 + self.x2) / 2.0, self.y1)

    def contains(self, point: tuple[float, float]) -> bool:
        px, py = point
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2

    def is_upper_region(self, point: tuple[float, float], upper_fraction: float) -> bool:
        _, py = point
        return py <= self.y1 + upper_fraction * self.height


@dataclass
class HeadAssociation:
    head_id: str
    player_track_id: Optional[int]
    association_confidence: float
    distance_px: float
    contained: bool
    notes: str = ""


def _score(head_center: tuple[float, float], player: TrackedPlayer, upper_fraction: float) -> float:
    """Derive a match score in [0, 1] from containment, upper-body priority,
    and normalized distance to the player's head anchor."""
    d = math.hypot(head_center[0] - player.head_anchor[0], head_center[1] - player.head_anchor[1])
    scale = max(player.width, player.height, 1.0)
    proximity = max(0.0, 1.0 - d / scale)  # 1.0 = right at anchor, 0.0 = >= 1 box-size away
    upper_weight = 1.0 if player.is_upper_region(head_center, upper_fraction) else 0.5
    return upper_weight * proximity


def associate_heads_to_players(
    heads: list[HeadBox],
    players: list[TrackedPlayer],
    upper_fraction: float = 0.4,
    max_distance_factor: float = 1.0,
) -> list[HeadAssociation]:
    """Associate each head to the best-matching tracked player in this frame.

    Parameters
    ----------
    heads : detected head boxes in the current frame.
    players : tracked player boxes in the current frame (from ByteTrack).
    upper_fraction : fraction of the player box height (from the top)
        considered the 'upper body' region where a head should be.
    max_distance_factor : for heads not contained by any player box, the
        maximum allowed distance to a player's head anchor, expressed as a
        multiple of that player's box size (width/height max). Candidates
        beyond this are not associated.
    """
    results: list[HeadAssociation] = []
    for head in heads:
        hc = head.center

        if not players:
            results.append(HeadAssociation(head.head_id, None, 0.0, math.inf, False,
                                           "no tracked players in frame"))
            continue

        containing = [p for p in players if p.contains(hc)]
        pool = containing if containing else players
        gate_ok = containing is pool  # contained candidates are never gated

        scored = []
        for p in pool:
            d = math.hypot(hc[0] - p.head_anchor[0], hc[1] - p.head_anchor[1])
            if not gate_ok and d > max_distance_factor * max(p.width, p.height, 1.0):
                continue
            scored.append((p, d, _score(hc, p, upper_fraction)))

        if not scored:
            results.append(HeadAssociation(head.head_id, None, 0.0, math.inf, False,
                                           "no player contains head and none within distance gate"))
            continue

        best_player, best_dist, best_score = max(scored, key=lambda t: t[2])
        results.append(HeadAssociation(
            head_id=head.head_id,
            player_track_id=best_player.track_id,
            association_confidence=round(max(0.0, min(1.0, best_score)), 4),
            distance_px=round(best_dist, 2),
            contained=bool(containing),
            notes="" if containing else "associated via nearest-player fallback (not contained)",
        ))

    return results
