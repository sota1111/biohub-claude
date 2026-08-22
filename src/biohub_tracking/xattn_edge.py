"""Learned cross-attention edge-linking cost (SOT-2994, SimpleNodeTransformer port).

The champion links each ``t -> t+1`` transition with the ARGUS motion-model LAP
cost (SOT-2864): a source is matched to the distance/motion-nearest feasible
successor. Two prior *learned* linking axes on the classical detections were
**REJECTED** and frame the new-evidence angle this module tests:

* **SOT-2841** — a per-edge *logistic* re-ranker over the joint edge-feature vector
  (:data:`biohub_tracking.edge_linker.EDGE_FEATURE_NAMES`). Held-out ``p_edge`` gap
  0.42 (genuinely discriminative) but **zero CV gain**: a fixed raw-distance
  feasible set is saturated (near ≈ GT), so re-ordering it recovers nothing.
* **SOT-2870** — the same logistic driving a motion+shape **gate expansion**; still
  strictly worse than the plain SOT-2864 motion gate (regressed the clean 44b6
  family), 4/4 non-regression failed.

Both used an **independent per-edge** classifier: the score of pair ``(i, j)`` is a
function of that pair's feature vector alone. The official royerlab baseline instead
scores an edge with a **cross-attention transformer** (``SimpleNodeTransformer``):
each node pools information from the *other candidate nodes* before an edge is
scored, so the assignment sees the whole competing candidate set, not one pair in
isolation. This module ports that idea to the classical detections as the structural
differentiator from SOT-2841/2870 (and from switching a fixed operating point,
SOT-2922/2923/2931):

* the per-edge feature vector is the **proven** :func:`edge_feature_planes` (F=10),
  reused verbatim so there is no train/infer feature skew;
* each edge embedding is then **contextualised by attention** over its source's
  competing successors (row attention) *and* its destination's competing
  predecessors (column attention) — a lightweight single-query cross-attention
  pooling that injects the candidate-set distribution into every edge score;
* the head maps ``[edge_embed, row_context, col_context]`` to ``p_edge``.

The learned ``p_edge`` enters the link cost as ``dist + weight * (1 - p_edge)`` —
**re-rank only, metric-valid**: the ``<= max_distance`` feasibility gate stays on the
raw/motion distance (exactly like SOT-2841), so it reorders the champion's feasible
set and never admits a new long-range edge.

**Offline / exec-compat by construction.** Training uses torch (GPU/CPU, dev-time
only) but the shipped weights are small dense arrays embedded in the link config and
**inference is pure numpy** — a standardize → embed → masked-softmax attention →
2-layer head, no torch / sklearn / pickle / filesystem at inference, ``exec()`` /
no-``__file__`` safe. A parity test pins the numpy forward to the torch forward.
``weight == 0`` (or ``xattn_edge_model is None`` / no descriptors) drops the term and
reproduces the champion **byte-for-byte** (a strict default-off superset), so the
frozen champion config stays byte-identical.

Node features consume the SOT-2993 learned detector's node features when a config
supplies them; absent that (SOT-2993 unmerged) it falls back to the handcrafted
geometric+intensity :func:`edge_feature_planes` on the classical detections, so this
axis is **evaluable standalone** without hard-blocking on SOT-2993.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .edge_linker import (
    EDGE_FEATURE_NAMES,
    N_EDGE_FEATURES,
    _sigmoid,
    edge_feature_planes,
)

# The learned scorer reuses the SOT-2841 edge feature extraction verbatim (no skew).
XATTN_FEATURE_NAMES: tuple[str, ...] = EDGE_FEATURE_NAMES
# Architecture tag stored in the config so a loaded model self-describes / a future
# architecture change is caught at load time rather than silently mis-scoring.
XATTN_ARCH_VERSION: str = "xattn-edge-v1"


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _masked_softmax(scores: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    """Softmax of ``scores`` over ``axis``, restricted to the ``mask`` entries.

    Infeasible entries get weight exactly 0; a row/column with no feasible entry
    yields all-zero weights (its context vector is then the zero vector — a
    well-defined "no competition" signal), never a divide-by-zero or NaN.
    """
    neg_inf = np.float64(-1e30)
    masked_scores = np.where(mask, scores, neg_inf)
    m = np.max(masked_scores, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.where(mask, np.exp(masked_scores - m), 0.0)
    s = e.sum(axis=axis, keepdims=True)
    s = np.where(s > 0.0, s, 1.0)
    return e / s


@dataclass(frozen=True)
class CrossAttentionEdgeCost:
    """Embedded pure-numpy cross-attention edge-linking cost (SOT-2994).

    For each feasible ``(P, Q)`` edge the forward pass is::

        z   = (x - mean) / std                       # standardized edge features
        h   = relu(z @ W1 + b1)                      # (P, Q, d) edge embedding
        a_r = softmax_j( h . a_row )  over feasible  # row (successor) attention
        c_r = sum_j a_r * h                          # (P, d) source context
        a_c = softmax_i( h . a_col )  over feasible  # col (predecessor) attention
        c_c = sum_i a_c * h                          # (Q, d) dest context
        g   = concat([h, c_r broadcast, c_c broadcast])  # (P, Q, 3d)
        p   = sigmoid( relu(g @ W2 + b2) @ w3 + b3 ) # (P, Q) edge probability

    and the added link cost is ``weight * (1 - p)`` (re-rank only). All weights are
    embedded in the link config; inference is numpy-only (Kaggle-kernel / exec safe).
    ``weight == 0`` makes the penalty identically zero → the linker is byte-for-byte
    the champion.
    """

    feature_names: tuple[str, ...]
    d: int
    h2: int
    mean: np.ndarray  # (F,)
    std: np.ndarray  # (F,)
    W1: np.ndarray  # (F, d)
    b1: np.ndarray  # (d,)
    a_row: np.ndarray  # (d,)
    a_col: np.ndarray  # (d,)
    W2: np.ndarray  # (3d, h2)
    b2: np.ndarray  # (h2,)
    w3: np.ndarray  # (h2,)
    b3: float
    weight: float
    arch: str = XATTN_ARCH_VERSION

    # -- forward -----------------------------------------------------------------
    def _forward(self, planes: np.ndarray, feasible: np.ndarray) -> np.ndarray:
        """``(P, Q)`` edge probability from ``(P, Q, F)`` planes + feasibility mask."""
        std = np.where(self.std > 1e-12, self.std, 1.0)
        z = (planes - self.mean) / std  # (P, Q, F)
        h = _relu(z @ self.W1 + self.b1)  # (P, Q, d)

        # Row (successor) attention: for each source i, pool over its feasible j.
        e_row = h @ self.a_row  # (P, Q)
        alpha_row = _masked_softmax(e_row, feasible, axis=1)  # (P, Q)
        c_row = np.einsum("pq,pqd->pd", alpha_row, h)  # (P, d)
        c_row_b = np.broadcast_to(c_row[:, None, :], h.shape)  # (P, Q, d)

        # Column (predecessor) attention: for each dest j, pool over its feasible i.
        e_col = h @ self.a_col  # (P, Q)
        alpha_col = _masked_softmax(e_col, feasible, axis=0)  # (P, Q)
        c_col = np.einsum("pq,pqd->qd", alpha_col, h)  # (Q, d)
        c_col_b = np.broadcast_to(c_col[None, :, :], h.shape)  # (P, Q, d)

        g = np.concatenate([h, c_row_b, c_col_b], axis=2)  # (P, Q, 3d)
        hid = _relu(g @ self.W2 + self.b2)  # (P, Q, h2)
        logit = hid @ self.w3 + self.b3  # (P, Q)
        return _sigmoid(logit)

    def _feasible(
        self, dist_scaled: np.ndarray, max_distance: float, dist_pred: np.ndarray | None
    ) -> np.ndarray:
        """The candidate set the attention pools over — identical to the feasible set
        :func:`edge_feature_planes` computes its ranks/rivals over, so features and
        attention describe the same competition."""
        if dist_pred is None:
            return dist_scaled <= max_distance
        return (dist_scaled <= max_distance) | (dist_pred <= max_distance)

    def probability_planes(
        self,
        dist_scaled: np.ndarray,
        src_desc: np.ndarray,
        dst_desc: np.ndarray,
        max_distance: float,
        dist_pred: np.ndarray | None = None,
    ) -> np.ndarray:
        """``(P, Q)`` learned edge probability ``p_edge`` for every pair."""
        planes = edge_feature_planes(
            dist_scaled, src_desc, dst_desc, max_distance, dist_pred=dist_pred
        )
        feasible = self._feasible(dist_scaled, max_distance, dist_pred)
        return self._forward(planes, feasible)

    def penalty(
        self,
        dist_scaled: np.ndarray,
        src_desc: np.ndarray,
        dst_desc: np.ndarray,
        max_distance: float,
        dist_pred: np.ndarray | None = None,
    ) -> np.ndarray:
        """``weight * (1 - p_edge)`` added-cost matrix ``(P, Q)`` (0 when weight 0)."""
        if self.weight == 0.0:
            return np.zeros_like(dist_scaled, dtype=np.float64)
        p_edge = self.probability_planes(
            dist_scaled, src_desc, dst_desc, max_distance, dist_pred=dist_pred
        )
        return self.weight * (1.0 - p_edge)

    def score_transition(
        self, planes: np.ndarray, feasible: np.ndarray
    ) -> np.ndarray:
        """``p_edge`` from precomputed planes + feasibility (screen/parity helper)."""
        return self._forward(np.asarray(planes, dtype=np.float64), np.asarray(feasible))

    # -- (de)serialization -------------------------------------------------------
    def to_dict(self) -> dict:
        """JSON-serialisable config block (embeddable in ``champion/config.json``)."""
        return {
            "arch": self.arch,
            "feature_names": list(self.feature_names),
            "d": int(self.d),
            "h2": int(self.h2),
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "W1": self.W1.astype(float).tolist(),
            "b1": [float(v) for v in self.b1],
            "a_row": [float(v) for v in self.a_row],
            "a_col": [float(v) for v in self.a_col],
            "W2": self.W2.astype(float).tolist(),
            "b2": [float(v) for v in self.b2],
            "w3": [float(v) for v in self.w3],
            "b3": float(self.b3),
            "weight": float(self.weight),
        }

    @classmethod
    def from_dict(cls, dd: dict) -> "CrossAttentionEdgeCost":
        """Rebuild a model from its serialized block (validates feature order + arch)."""
        names = tuple(dd["feature_names"])
        if names != XATTN_FEATURE_NAMES:
            raise ValueError(
                "xattn_edge_model feature_names do not match XATTN_FEATURE_NAMES "
                "(feature extraction changed; retrain the model)"
            )
        arch = dd.get("arch", XATTN_ARCH_VERSION)
        if arch != XATTN_ARCH_VERSION:
            raise ValueError(
                f"xattn_edge_model arch {arch!r} != {XATTN_ARCH_VERSION!r} "
                "(architecture changed; retrain the model)"
            )
        d = int(dd["d"])
        h2 = int(dd["h2"])
        W1 = np.asarray(dd["W1"], dtype=np.float64).reshape(N_EDGE_FEATURES, d)
        W2 = np.asarray(dd["W2"], dtype=np.float64).reshape(3 * d, h2)
        return cls(
            feature_names=names,
            d=d,
            h2=h2,
            mean=np.asarray(dd["mean"], dtype=np.float64),
            std=np.asarray(dd["std"], dtype=np.float64),
            W1=W1,
            b1=np.asarray(dd["b1"], dtype=np.float64),
            a_row=np.asarray(dd["a_row"], dtype=np.float64),
            a_col=np.asarray(dd["a_col"], dtype=np.float64),
            W2=W2,
            b2=np.asarray(dd["b2"], dtype=np.float64),
            w3=np.asarray(dd["w3"], dtype=np.float64),
            b3=float(dd["b3"]),
            weight=float(dd.get("weight", 0.0)),
            arch=arch,
        )

    def with_weight(self, weight: float) -> "CrossAttentionEdgeCost":
        """A copy at a different re-rank ``weight`` (the screen's strength sweep)."""
        return CrossAttentionEdgeCost(
            feature_names=self.feature_names,
            d=self.d,
            h2=self.h2,
            mean=self.mean,
            std=self.std,
            W1=self.W1,
            b1=self.b1,
            a_row=self.a_row,
            a_col=self.a_col,
            W2=self.W2,
            b2=self.b2,
            w3=self.w3,
            b3=self.b3,
            weight=float(weight),
            arch=self.arch,
        )


def torch_available() -> bool:
    """True iff torch can be imported (training only; inference is numpy-only)."""
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _standardization(transitions: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Feature mean/std over all TRAINABLE edge vectors across transitions."""
    rows = []
    for tr in transitions:
        planes = tr["planes"]
        trainable = tr["trainable"]
        if trainable.any():
            rows.append(planes[trainable])
    if not rows:
        return np.zeros(N_EDGE_FEATURES), np.ones(N_EDGE_FEATURES)
    X = np.concatenate(rows, axis=0)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return mean, std


def fit_cross_attention(
    transitions: list[dict],
    *,
    weight: float = 0.0,
    d: int = 12,
    h2: int = 12,
    epochs: int = 200,
    lr: float = 5e-3,
    l2: float = 1e-4,
    seed: int = 0,
) -> CrossAttentionEdgeCost:
    """Fit a :class:`CrossAttentionEdgeCost` from per-transition training tensors.

    ``transitions`` is a list of dicts, one per ``t -> t+1`` frame pair::

        {"planes": (P, Q, F) float, "feasible": (P, Q) bool,
         "trainable": (P, Q) bool, "labels": (P, Q) 0/1 float}

    where ``feasible`` is the candidate set the attention pools over and
    ``trainable`` marks the masked-sparse-supervision pairs (feasible pairs of a
    GT-matched source; positive = GT edge, negative = its other feasible
    successors — the official baseline's *masked* loss, which avoids the SOT-2828
    node positive-unlabeled contamination). Class imbalance is handled by inverse
    class frequency over the trainable pairs. Training is torch on CPU/GPU
    (deterministic under ``seed``); the returned model is pure-numpy for inference.

    Raises ``RuntimeError`` if torch is unavailable (callers feature-detect with
    :func:`torch_available`).
    """
    if not torch_available():
        raise RuntimeError(
            "fit_cross_attention needs torch (training only); inference is numpy-only"
        )
    import torch

    torch.manual_seed(seed)
    # Training runs in float32 (2×+ faster on CPU, half the memory for the dense
    # ~750×750 frames); the exported weights are cast to float64 for the numpy
    # inference path, and the numpy↔torch parity test recomputes in float64 from
    # those exported weights, so the shipped precision is unaffected.
    dt = torch.float32
    mean, std = _standardization(transitions)
    mean_t = torch.tensor(mean, dtype=dt)
    std_t = torch.tensor(std, dtype=dt)

    # Xavier-ish init from a seeded generator (reproducible).
    g = torch.Generator().manual_seed(seed)

    def _init(*shape, scale):
        return (torch.randn(*shape, generator=g, dtype=dt) * scale).requires_grad_(True)

    W1 = _init(N_EDGE_FEATURES, d, scale=(1.0 / N_EDGE_FEATURES) ** 0.5)
    b1 = torch.zeros(d, dtype=dt, requires_grad=True)
    a_row = _init(d, scale=(1.0 / d) ** 0.5)
    a_col = _init(d, scale=(1.0 / d) ** 0.5)
    W2 = _init(3 * d, h2, scale=(1.0 / (3 * d)) ** 0.5)
    b2 = torch.zeros(h2, dtype=dt, requires_grad=True)
    w3 = _init(h2, scale=(1.0 / h2) ** 0.5)
    b3 = torch.zeros(1, dtype=dt, requires_grad=True)
    params = [W1, b1, a_row, a_col, W2, b2, w3, b3]

    # Precompute torch tensors per transition (only those with trainable pairs).
    tt = []
    n_pos = 0.0
    n_neg = 0.0
    for tr in transitions:
        if not tr["trainable"].any():
            continue
        planes = torch.tensor(tr["planes"], dtype=dt)
        feasible = torch.tensor(tr["feasible"], dtype=torch.bool)
        trainable = torch.tensor(tr["trainable"], dtype=torch.bool)
        labels = torch.tensor(tr["labels"], dtype=dt)
        tt.append((planes, feasible, trainable, labels))
        n_pos += float(labels[trainable].sum())
        n_neg += float((1.0 - labels)[trainable].sum())
    if not tt:  # no trainable data → an inert model (weight applied, p_edge≈const)
        return _from_torch(
            mean, std, d, h2, W1, b1, a_row, a_col, W2, b2, w3, b3, weight
        )

    w_pos = (n_pos + n_neg) / (2.0 * n_pos) if n_pos > 0 else 0.0
    w_neg = (n_pos + n_neg) / (2.0 * n_neg) if n_neg > 0 else 0.0
    # Total sample weight over all trainable pairs, precomputed once (data fixed), so
    # each transition can back-prop its own loss (freeing its graph) while the
    # accumulated gradient still equals that of the class-weighted mean BCE.
    total_w = 0.0
    for _planes, _feas, trainable, labels in tt:
        yy = labels[trainable]
        total_w += float(torch.where(yy > 0.5, w_pos, w_neg).sum())
    total_w = max(total_w, 1.0)

    neg_inf = torch.tensor(-1e30, dtype=dt)
    opt = torch.optim.Adam(params, lr=lr, weight_decay=l2)

    def forward(planes, feasible):
        z = (planes - mean_t) / std_t
        h = torch.relu(z @ W1 + b1)  # (P,Q,d)
        e_row = h @ a_row  # (P,Q)
        e_row = torch.where(feasible, e_row, neg_inf)
        alpha_row = torch.softmax(e_row, dim=1)
        alpha_row = torch.where(feasible, alpha_row, torch.zeros_like(alpha_row))
        c_row = torch.einsum("pq,pqd->pd", alpha_row, h)
        c_row_b = c_row.unsqueeze(1).expand(-1, h.shape[1], -1)
        e_col = h @ a_col
        e_col = torch.where(feasible, e_col, neg_inf)
        alpha_col = torch.softmax(e_col, dim=0)
        alpha_col = torch.where(feasible, alpha_col, torch.zeros_like(alpha_col))
        c_col = torch.einsum("pq,pqd->qd", alpha_col, h)
        c_col_b = c_col.unsqueeze(0).expand(h.shape[0], -1, -1)
        gg = torch.cat([h, c_row_b, c_col_b], dim=2)
        hid = torch.relu(gg @ W2 + b2)
        logit = hid @ w3 + b3
        return logit

    for _ in range(epochs):
        opt.zero_grad()
        for planes, feasible, trainable, labels in tt:
            logit = forward(planes, feasible)
            lg = logit[trainable]
            yy = labels[trainable]
            sw = torch.where(yy > 0.5, w_pos, w_neg)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                lg, yy, weight=sw, reduction="sum"
            )
            # Back-prop each transition immediately so only one transition's autograd
            # graph is ever live (bounded memory for dense ~750×750 frames); grads
            # accumulate into .grad and equal the gradient of the class-weighted mean
            # BCE over all transitions.
            (bce / total_w).backward()
        opt.step()

    return _from_torch(mean, std, d, h2, W1, b1, a_row, a_col, W2, b2, w3, b3, weight)


def _from_torch(mean, std, d, h2, W1, b1, a_row, a_col, W2, b2, w3, b3, weight):
    """Detach a trained torch model into an embedded numpy :class:`CrossAttentionEdgeCost`."""
    def npy(t):
        return t.detach().cpu().numpy().astype(np.float64)

    return CrossAttentionEdgeCost(
        feature_names=XATTN_FEATURE_NAMES,
        d=int(d),
        h2=int(h2),
        mean=np.asarray(mean, dtype=np.float64),
        std=np.asarray(std, dtype=np.float64),
        W1=npy(W1),
        b1=npy(b1),
        a_row=npy(a_row),
        a_col=npy(a_col),
        W2=npy(W2),
        b2=npy(b2),
        w3=npy(w3),
        b3=float(npy(b3).reshape(-1)[0]),
        weight=float(weight),
    )
