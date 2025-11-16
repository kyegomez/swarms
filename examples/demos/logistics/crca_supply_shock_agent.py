import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from loguru import logger
import json
import re
import warnings

from swarms import Agent
from swarms.agents import CRCAAgent


SUPPLY_AGENT_PROMPT = """
You are an expert supply chain analyst with deep knowledge of multi-echelon inventory, logistics, and robust control.
You analyze telemetry and SCM outputs, apply causal reasoning, and propose safe, actionable interventions.

Data-awareness:
- Consider lead time (L), survival/transport factor (phi), capacity (K), backlog (B), inventory (I), demand (D), orders (O), receipts (R), price (p).
- Reference learned causal edges when explaining recommendations (e.g., B→p+, I→p−, dcost→p+, L→R−, D→O+).

Uncertainty and risk:
- Note regime shifts or drift (EWMA/CUSUM/BOCPD). Prefer conservative actions when uncertainty is high.
- Propose z_alpha updates, reroute/expedite shares with CVaR-style caution and execution realism.

Guardrails:
- Avoid claims beyond the data window. Keep recommendations feasible under capacity and service constraints.
- Prefer stable signals and explain trade-offs (service vs. cost vs. bullwhip).

Output structure (always include these 7 sections):
1) Drivers: succinct causal drivers (e.g., L↑→R↓→I↓→B↑→p↑; dcost↑→p↑→D↓)
2) Regime/Alerts: note EWMA/CUSUM/BOCPD, utilization, and stability
3) Proposal: recommended z_alpha, reroute/expedite, and rationale
4) Expected Impact: service, cost proxy, bullwhip changes (direction and rough magnitude)
5) Risks/Uncertainty: cite instability or wide uncertainty; suggest mitigations
6) Counterfactuals: 1–2 do()-style scenarios and expected KPI shifts
7) Actionables: concrete next steps and monitoring items

Learned DAG alignment:
- Mirror the learned DAG strengths exactly in your explanation. Do not claim effects whose learned strength is ~0. If L→R≈0 or R→I≈0, reflect that and avoid relying on those edges. Base rationale on the actual learned edges provided.

After the 7 sections, you MUST output a final JSON object on a new line with the exact schema:
{
  "proposal": {"z_alpha_new": number, "reroute_share": number, "expedite_share": number, "dual_source_share": number},
  "notes": string,
  "expected_deltas": {"service": number, "cost_proxy": number, "bullwhip": number},
  "ci80": {"service": [low, high], "cost_proxy": [low, high], "bullwhip": [low, high]},
  "feasible": boolean
}
Numbers should be in [0, 3] for z_alpha_new, [0, 1] for shares.
"""


@dataclass
class SKUKey:
    sku: str
    facility: str


@dataclass
class PolicyParams:
    z_alpha: float = 1.28  # base safety factor (approx 90%)
    theta_smoothing: float = 0.35  # order smoothing parameter


@dataclass
class Elasticities:
    eta_c: float = 0.3  # pass-through elasticity to price from cost changes
    eta_B: float = 0.5  # price increases with backlog
    eta_I: float = 0.3  # price decreases with inventory


class SupplyShockCRCAgent:
    """CR-CA Supply-Shock Agent for multi-period inventory flows and interventions.

    Implements SCM flow, queueing-derived lead times, pricing pass-through, and
    integrates CRCAAgent for causal analysis and do()-counterfactuals.
    """

    def __init__(
        self,
        skus: List[SKUKey],
        T: int = 60,
        policy: PolicyParams = PolicyParams(),
        el: Elasticities = Elasticities(),
        seed: int = 7,
    ) -> None:
        self.skus = skus
        self.T = T
        self.policy = policy
        self.el = el
        self.rng = np.random.default_rng(seed)

        # Core data containers (panel: period x (sku,facility))
        index = pd.MultiIndex.from_tuples(
            [(t, k.sku, k.facility) for t in range(T) for k in skus],
            names=["t", "sku", "facility"],
        )
        self.df = pd.DataFrame(index=index)

        # Initialize states with simple priors
        self.df["D"] = self.rng.poisson(100, len(self.df))  # demand
        self.df["I"] = 120.0  # on-hand
        self.df["B"] = 0.0  # backorder
        self.df["P"] = 80.0  # pipeline
        self.df["O"] = 0.0  # orders placed
        self.df["R"] = 0.0  # receipts
        self.df["K"] = 1e6  # facility capacity (big default)
        self.df["phi"] = 1.0  # spoilage/survival fraction
        self.df["L"] = self.rng.integers(1, 4, len(self.df)).astype(float)  # lead time in periods (float for updates)
        self.df["c"] = 10.0  # unit cost
        self.df["dcost"] = 0.0  # cost change
        self.df["p_bar"] = 15.0
        self.df["p"] = self.df["p_bar"]
        self.df["D_ref"] = 100.0

        # CR-CA causal layer
        self.crca = CRCAAgent(
            name="crca-supply-shock",
            description="Causal layer for supply shocks and policy",
            model_name="gpt-4o-mini",
            max_loops=2,
        )
        self._build_causal_graph()

        # VSM overlay components (S2-S5)
        self.vsm = VSMOverlay()

        # Narrative LLM agent (graceful init)
        try:
            self.agent = Agent(
                agent_name="CRCA-Supply-Shock-Agent",
                system_prompt=SUPPLY_AGENT_PROMPT,
                model_name="gpt-4o",
                max_loops=1,
                autosave=False,
                dashboard=True,
                verbose=False,
                dynamic_temperature_enabled=True,
                context_length=200000,
                output_type="string",
                streaming_on=False,
            )
        except Exception as e:
            logger.warning(f"LLM Agent init failed (narrative disabled): {e}")
            self.agent = None

        # Feasibility config and last applied controls
        self.lane_capacity: float = 0.35  # max share sum for reroute+expedite+dual_source per cycle
        self.max_weekly_change: Dict[str, float] = {"z_alpha_new": 0.3, "reroute_share": 0.15, "expedite_share": 0.15, "dual_source_share": 0.2}
        self.last_controls: Dict[str, float] = {"z_alpha_new": self.policy.z_alpha, "reroute_share": 0.0, "expedite_share": 0.0, "dual_source_share": 0.0}
        # KPI normalization scalers
        self.kpi_units = {"service": "%", "cost_proxy": "currency", "bullwhip": "ratio"}
        self.cost_unit_multiplier = 1.0
        # KPI/action history and RL weights
        self.kpi_history: List[Dict[str, float]] = []
        self.action_history: List[Dict[str, float]] = []
        self.mpc_weights: Dict[str, float] = {"service": 1.0, "cost": 1.0, "bullwhip": 0.5}
        # Dynamic data feeds
        self._feeds: Dict[str, Any] = {}
        # Direct pricing lever (applied as an offset to p_bar during simulate)
        self._price_adjust: float = 0.0

    # ===== Helper: logistics lever support from learned DAG =====
    def _logistics_support_from_strengths(self, strengths: Dict[str, float], tol: float = 0.05) -> bool:
        keys = ["L->R", "phi->R", "K->R"]
        return any(abs(float(strengths.get(k, 0.0))) > tol for k in keys)

    # ===== Helper: calibrate z to target service via short grid search =====
    def calibrate_z_to_service(self, target_service: float = 0.95, z_grid: Optional[np.ndarray] = None) -> float:
        if z_grid is None:
            z_grid = np.linspace(0.8, 2.6, 10)
        best_z = self.policy.z_alpha
        best_val = float("inf")
        # Snapshot current policy
        z_prev = self.policy.z_alpha
        try:
            for z in z_grid:
                self.policy.z_alpha = float(z)
                df = self.simulate()
                kpi = self.summarize(df)
                # distance to target with slight cost penalty to avoid extreme z
                val = abs(float(kpi.get("service_level", 0.0)) - target_service) + 0.01 * max(0.0, float(kpi.get("cost_proxy", 0.0)))
                if val < best_val:
                    best_val = val
                    best_z = float(z)
        finally:
            self.policy.z_alpha = z_prev
        return best_z

    # ===== Helper: minimal Pareto grid on (z, r, e) =====
    def pareto_front(self, base_z: float, allow_logistics: bool, trials: int = 30, allow_price: bool = False) -> List[Dict[str, Any]]:
        z_vals = np.clip(np.linspace(base_z - 0.2, base_z + 0.2, 5), 0.5, 3.0)
        r_vals = [0.0, 0.05, 0.1] if allow_logistics else [0.0]
        e_vals = [0.0, 0.05, 0.1] if allow_logistics else [0.0]
        p_vals = [0.0, -0.5, -1.0] if allow_price else [0.0]
        points: List[Dict[str, Any]] = []
        for z in z_vals:
            for r in r_vals:
                for e in e_vals:
                    for p in p_vals:
                        imp = self._quantified_impact(float(z), float(r), float(e), trials=trials, price_adjust=float(p))
                        exp = imp.get("expected", {})
                        points.append({
                            "z_alpha_new": float(z),
                            "reroute_share": float(r),
                            "expedite_share": float(e),
                            "price_adjust": float(p),
                            "expected": exp,
                            "ci80": imp.get("ci80", {}),
                            "cvar_loss": imp.get("cvar_loss", 0.0),
                        })
        # Pareto filter: maximize service, minimize cost and bullwhip
        def dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            ea, eb = a["expected"], b["expected"]
            svc_a, cost_a, bw_a = ea.get("service", 0.0), ea.get("cost_proxy", 0.0), ea.get("bullwhip", 0.0)
            svc_b, cost_b, bw_b = eb.get("service", 0.0), eb.get("cost_proxy", 0.0), eb.get("bullwhip", 0.0)
            return (svc_a >= svc_b and cost_a <= cost_b and bw_a <= bw_b) and (svc_a > svc_b or cost_a < cost_b or bw_a < bw_b)
        pareto: List[Dict[str, Any]] = []
        for p in points:
            if not any(dominates(q, p) for q in points if q is not p):
                pareto.append(p)
        return pareto

    def z_service_curve(self, z_values: Optional[np.ndarray] = None) -> List[Dict[str, float]]:
        if z_values is None:
            z_values = np.linspace(0.8, 2.6, 10)
        curve: List[Dict[str, float]] = []
        z_prev = self.policy.z_alpha
        try:
            for z in z_values:
                self.policy.z_alpha = float(z)
                df = self.simulate()
                k = self.summarize(df)
                curve.append({"z": float(z), "service": float(k.get("service_level", 0.0)), "cost_proxy": float(k.get("cost_proxy", 0.0))})
        finally:
            self.policy.z_alpha = z_prev
        return curve

    # ===== Real-world KPI ingestion =====
    def ingest_kpis(self, kpis: Dict[str, float]) -> None:
        """Ingest external KPIs (e.g., service %, cost, bullwhip) and store history."""
        safe = {
            "service": float(kpis.get("service", np.nan)),
            "cost_proxy": float(kpis.get("cost_proxy", np.nan)),
            "bullwhip": float(kpis.get("bullwhip", np.nan)),
        }
        self.kpi_history.append(safe)

    # ===== Dynamic data feeds =====
    def register_feed(self, name: str, fetch_fn: Any) -> None:
        """Register a callable that returns dicts to merge into df or KPIs."""
        self._feeds[name] = fetch_fn

    def poll_feeds(self) -> Dict[str, Any]:
        """Poll all feeds and merge into state; return a snapshot of updates."""
        updates: Dict[str, Any] = {}
        for name, fn in list(self._feeds.items()):
            try:
                data = fn()
                updates[name] = data
                # If KPI-like, ingest; else if dataframe-like keys, merge shallowly
                if isinstance(data, dict) and set(["service", "cost_proxy", "bullwhip"]).issubset(set(data.keys())):
                    self.ingest_kpis(data)
                # Extend here to merge time-series; keeping simple for now
            except Exception as e:
                updates[name] = {"error": str(e)}
        return updates

    # ===== Action validation and rollback =====
    def validate_and_rollback(
        self,
        new_kpis: Dict[str, float],
        thresholds: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """Validate last applied action against fresh KPIs; rollback if violated.

        thresholds: {"min_service_gain": 0.0, "max_cost_increase": 0.0, "max_bullwhip_increase": 0.0}
        """
        if thresholds is None:
            thresholds = {"min_service_gain": 0.0, "max_cost_increase": 0.0, "max_bullwhip_increase": 0.0}
        if not self.kpi_history:
            self.ingest_kpis(new_kpis)
            return {"rolled_back": False, "reason": "no_baseline"}
        prev = self.kpi_history[-1]
        self.ingest_kpis(new_kpis)
        ds = float(new_kpis.get("service", 0.0) - (prev.get("service", 0.0) if prev.get("service") is not None else 0.0))
        dc = float(new_kpis.get("cost_proxy", 0.0) - (prev.get("cost_proxy", 0.0) if prev.get("cost_proxy") is not None else 0.0))
        db = float(new_kpis.get("bullwhip", 0.0) - (prev.get("bullwhip", 0.0) if prev.get("bullwhip") is not None else 0.0))
        violate = (ds < thresholds["min_service_gain"]) or (dc > thresholds["max_cost_increase"]) or (db > thresholds["max_bullwhip_increase"]) 
        if violate and self.action_history:
            # rollback: revert to previous controls, dampen risky knobs
            last = self.action_history[-1]
            self.policy.z_alpha = float(last.get("z_alpha_prev", self.policy.z_alpha))
            self.last_controls["z_alpha_new"] = self.policy.z_alpha
            for k in ["reroute_share", "expedite_share", "dual_source_share"]:
                self.last_controls[k] = max(0.0, self.last_controls.get(k, 0.0) * 0.5)
            return {"rolled_back": True, "reason": "threshold_violation", "delta": {"service": ds, "cost": dc, "bullwhip": db}}
        return {"rolled_back": False, "reason": "ok", "delta": {"service": ds, "cost": dc, "bullwhip": db}}

    # ===== Reinforcement-based self-tuning =====
    def reinforce_from_outcome(self, expected: Dict[str, float]) -> Dict[str, float]:
        """Update MPC weights from outcome using a simple reward: r = dS - a*max(0,dC) - b*max(0,dB)."""
        dS = float(expected.get("service", 0.0))
        dC = float(expected.get("cost_proxy", 0.0))
        dB = float(expected.get("bullwhip", 0.0))
        a, b, lr = 1.0, 0.5, 0.1
        reward = dS - a * max(0.0, dC) - b * max(0.0, dB)
        # Increase emphasis on service if reward positive; else increase cost/bullwhip penalty
        if reward >= 0:
            self.mpc_weights["service"] = float(min(2.0, self.mpc_weights.get("service", 1.0) + lr * reward))
            self.mpc_weights["cost"] = float(max(0.2, self.mpc_weights.get("cost", 1.0) * (1.0 - 0.05)))
        else:
            self.mpc_weights["cost"] = float(min(2.0, self.mpc_weights.get("cost", 1.0) + lr * (-reward)))
            self.mpc_weights["bullwhip"] = float(min(2.0, self.mpc_weights.get("bullwhip", 0.5) + lr * 0.5 * (-reward)))
        return dict(self.mpc_weights)

    def _feasible(self, proposal: Dict[str, Any]) -> bool:
        rr = float(proposal.get("reroute_share", 0.0))
        ex = float(proposal.get("expedite_share", 0.0))
        ds = float(proposal.get("dual_source_share", 0.0))
        if rr < 0 or ex < 0:
            return False
        if rr + ex + ds > self.lane_capacity:
            return False
        # rate limits
        for k, cap in self.max_weekly_change.items():
            prev = float(self.last_controls.get(k, 0.0))
            cur = float(proposal.get(k, prev))
            if abs(cur - prev) > cap:
                return False
        return True

    def _quantified_impact(self, z_new: float, rr: float, ex: float, trials: int = 100, price_adjust: float = 0.0, alpha: float = 0.9) -> Dict[str, Any]:
        # Run small Monte Carlo by injecting noise on D and L; measure KPI deltas
        base = self.simulate()
        base_kpi = self.summarize(base)
        deltas = []
        rng = self.rng
        saved_price_adj = self._price_adjust
        for _ in range(trials):
            # stochastic shocks via temporary tweaks
            shock = {"stochastic": True}
            self.policy.z_alpha = z_new
            # approximate expedite/reroute: reduce L and increase phi within shares
            self.df["phi"] = np.clip(self.df["phi"] * (1.0 + 0.1 * rr), 0.2, 1.2)
            self.df["L"] = np.clip(self.df["L"] * (1.0 - 0.2 * ex), 1.0, None)
            self._price_adjust = float(price_adjust)
            sim = self.simulate(interventions=None)
            kpi = self.summarize(sim)
            deltas.append({
                "service": float(kpi["service_level"] - base_kpi["service_level"]),
                "cost_proxy": float(kpi["cost_proxy"] - base_kpi["cost_proxy"]),
                "bullwhip": float(kpi["bullwhip"] - base_kpi["bullwhip"]),
            })
        # restore controls
        self._price_adjust = saved_price_adj
        # Aggregate
        svc = np.array([d["service"] for d in deltas])
        cst = np.array([d["cost_proxy"] for d in deltas])
        bwe = np.array([d["bullwhip"] for d in deltas])
        def ci80(arr: np.ndarray) -> Tuple[float, float]:
            return float(np.quantile(arr, 0.1)), float(np.quantile(arr, 0.9))
        def ci95(arr: np.ndarray) -> Tuple[float, float]:
            return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))
        loss = cst - svc + bwe
        thr = float(np.quantile(loss, alpha))
        cvar = float(loss[loss >= thr].mean()) if np.any(loss >= thr) else float(loss.mean())
        return {
            "expected": {"service": float(np.mean(svc)), "cost_proxy": float(np.mean(cst)), "bullwhip": float(np.mean(bwe))},
            "ci80": {"service": ci80(svc), "cost_proxy": ci80(cst), "bullwhip": ci80(bwe)},
            "ci95": {"service": ci95(svc), "cost_proxy": ci95(cst), "bullwhip": ci95(bwe)},
            "samples": {"service": svc.tolist(), "cost_proxy": cst.tolist(), "bullwhip": bwe.tolist()},
            "cvar_alpha": alpha,
            "cvar_loss": cvar,
        }

    # ===== CVaR grid minimizer (discrete neighborhood search) =====
    def cvar_select_from_grid(
        self,
        base: Dict[str, float],
        alpha: float = 0.9,
        eps: float = 0.05,
        trials: int = 40,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Pick among a small grid around base (z,r,e,ds) to minimize CVaR_alpha of loss.

        loss = w_cost*Δcost - w_service*Δservice + w_bw*Δbullwhip
        """
        if weights is None:
            weights = {"cost": 1.0, "service": 1.0, "bullwhip": 0.5}
        z0 = float(base.get("z_alpha_new", self.policy.z_alpha))
        r0 = float(base.get("reroute_share", 0.0))
        e0 = float(base.get("expedite_share", 0.0))
        ds0 = float(base.get("dual_source_share", 0.0))
        cand = [
            {"z": z0, "r": r0, "e": e0},
            {"z": z0+eps, "r": r0, "e": e0},
            {"z": z0, "r": r0+eps, "e": e0},
            {"z": z0, "r": r0, "e": e0+eps},
            {"z": z0-eps, "r": r0, "e": e0},
        ]
        best = None
        best_val = float("inf")
        tail_q = alpha
        for c in cand:
            imp = self._quantified_impact(c["z"], c["r"], c["e"], trials=trials)
            svc = np.array(imp["samples"]["service"])  # Δservice
            cst = np.array(imp["samples"]["cost_proxy"])  # Δcost
            bwe = np.array(imp["samples"]["bullwhip"])  # Δbullwhip
            loss = weights["cost"] * cst - weights["service"] * svc + weights["bullwhip"] * bwe
            thresh = np.quantile(loss, tail_q)
            cvar = float(loss[loss >= thresh].mean()) if np.any(loss >= thresh) else float(loss.mean())
            if cvar < best_val:
                best_val = cvar
                best = {"z_alpha_new": c["z"], "reroute_share": c["r"], "expedite_share": c["e"], "dual_source_share": ds0}
        return {"choice": best, "cvar": best_val, "alpha": alpha}

    @staticmethod
    def _extract_final_json(text: str) -> Optional[Dict[str, Any]]:
        try:
            # Find the last JSON-like block with required keys
            matches = re.findall(r"\{[\s\S]*?\}", text)
            for chunk in reversed(matches):
                if '"proposal"' in chunk and '"z_alpha_new"' in chunk:
                    return json.loads(chunk)
        except Exception:
            return None
        return None

    @staticmethod
    def _validate_proposal(p: Dict[str, Any]) -> Optional[Dict[str, float]]:
        try:
            z = float(np.clip(float(p.get("z_alpha_new")), 0.0, 3.0))
            rr = float(np.clip(float(p.get("reroute_share")), 0.0, 1.0))
            ex = float(np.clip(float(p.get("expedite_share")), 0.0, 1.0))
            return {"z_alpha_new": z, "reroute_share": rr, "expedite_share": ex}
        except Exception:
            return None

    def _llm_decision_loop(
        self,
        panel: pd.DataFrame,
        s2: Dict[str, Any],
        proposal: Optional[Dict[str, Any]],
        decision: Dict[str, Any],
        strengths: Dict[str, float],
        kpis: Dict[str, Any],
        rounds: int = 3,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if self.agent is None:
            return "analysis_unavailable: agent_not_initialized", proposal

        alerts = s2.get("alerts", [])
        ai_out: Any = ""
        for r in range(rounds):
            # Edge guidance for narration
            top_edges = sorted(strengths.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
            weak_edges = [k for k, v in strengths.items() if abs(v) <= 0.05]
            narrative_prompt = (
                f"Round {r+1}/{rounds}. Analyze state and propose safe actions.\n"
                f"KPIs: service={kpis.get('service_level', 0.0):.2f}, cost_proxy={kpis.get('cost_proxy', 0.0):.2f}, bullwhip={kpis.get('bullwhip', 0.0):.2f}.\n"
                f"Alerts: {alerts}.\n"
                f"Current proposal: {proposal}. Decision: {decision}.\n"
                f"Causal edges (use these): {dict(top_edges)}; avoid near-zero edges: {weak_edges}.\n"
                f"Provide the 7-section output and the final JSON exactly per schema."
            )
            try:
                ai_out = self.agent.run(narrative_prompt)
            except Exception as e:
                return f"analysis_unavailable: {e}", proposal
            llm_json = self._extract_final_json(str(ai_out)) if isinstance(ai_out, str) else None
            if llm_json and isinstance(llm_json.get("proposal"), dict):
                validated = self._validate_proposal(llm_json["proposal"])
                if validated:
                    proposal = {**validated, "source": "llm"}
                    decision = self.vsm.s5_policy_gate(proposal, s2, self.vsm.s3_star_audit(panel), risk_cap_cvar=0.15)
        return str(ai_out), proposal

    def _build_causal_graph(self) -> None:
        g = self.crca.causal_graph
        g.clear()
        vars_graph = [
            "L",  # lead time
            "phi",  # survival/transport factor
            "K",  # capacity
            "dcost",
            "B",  # backlog
            "I",  # inventory
            "p",  # price
            "D",  # demand
            "O",  # orders
            "R",  # receipts
        ]
        g.add_nodes_from(vars_graph)
        # Structural influences (signs where sensible)
        self.crca.add_causal_relationship("K", "R", strength=0.0)  # more cap -> more receipts
        self.crca.add_causal_relationship("phi", "R", strength=0.0)  # more survival -> more receipts
        self.crca.add_causal_relationship("L", "R", strength=0.0)  # longer lead -> lower timely receipts
        self.crca.edge_sign_constraints[("L", "R")] = -1

        self.crca.add_causal_relationship("B", "p", strength=0.0)  # backlog -> higher price
        self.crca.edge_sign_constraints[("B", "p")] = 1
        self.crca.add_causal_relationship("I", "p", strength=0.0)  # inventory -> lower price
        self.crca.edge_sign_constraints[("I", "p")] = -1
        self.crca.add_causal_relationship("dcost", "p", strength=0.0)  # cost pass-through
        self.crca.edge_sign_constraints[("dcost", "p")] = 1

        self.crca.add_causal_relationship("p", "D", strength=0.0)  # pricing impacts demand
        self.crca.edge_sign_constraints[("p", "D")] = -1
        self.crca.add_causal_relationship("D", "O", strength=0.0)  # more demand -> more orders
        self.crca.edge_sign_constraints[("D", "O")] = 1
        self.crca.add_causal_relationship("R", "I", strength=0.0)
        self.crca.edge_sign_constraints[("R", "I")] = 1
        self.crca.add_causal_relationship("D", "B", strength=0.0)
        self.crca.edge_sign_constraints[("D", "B")] = 1

    @staticmethod
    def _relu(x: float) -> float:
        return float(max(0.0, x))

    def _arrivals(self, O_hist: List[float], L_hist: List[int], phi_hist: List[float], t: int) -> float:
        """Receipts at t: sum 1[L=ell]*O[t-ell]*phi[t-ell->t]."""
        total = 0.0
        for ell in range(1, min(10, t + 1)):
            if L_hist[t] == ell:
                total += O_hist[t - ell] * phi_hist[t - ell]
        return total

    def _queueing_leadtime(self, lam: float, mu: float, transport: float = 0.0) -> float:
        rho = min(0.95, lam / max(mu, 1e-6))
        wq = (rho / (mu * max(1e-6, (1 - rho)))) if rho < 0.999 else 10.0
        return wq + (1.0 / max(mu, 1e-6)) + transport

    def _price_pass_through(self, p_bar: float, dcost: float, B: float, I: float, D_ref: float) -> float:
        return (
            p_bar
            + self.el.eta_c * dcost
            + self.el.eta_B * (B / max(D_ref, 1e-6))
            - self.el.eta_I * (I / max(D_ref, 1e-6))
        )

    def simulate(self, interventions: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Run minimal multi-period SCM with optional do()-style interventions.

        interventions: dict with keys like {'outage_facility': {'F1': 1}, 'disruption_route': {'R1': 1}}
        For simplicity, treat interventions as shocks to K (capacity), L (lead), and phi.
        """
        df = self.df.copy()
        # Initialize per (sku,facility)
        for (sku, fac), sub in df.groupby(level=["sku", "facility" ]):
            O = [0.0] * self.T
            R = [0.0] * self.T
            I = [float(sub.iloc[0]["I"])] + [0.0] * (self.T - 1)
            B = [float(sub.iloc[0]["B"])] + [0.0] * (self.T - 1)
            P = [float(sub.iloc[0]["P"])] + [0.0] * (self.T - 1)
            L_hist = [int(sub.iloc[min(i, len(sub)-1)]["L"]) for i in range(self.T)]
            phi_hist = [float(sub.iloc[min(i, len(sub)-1)]["phi"]) for i in range(self.T)]
            K_hist = [float(sub.iloc[min(i, len(sub)-1)]["K"]) for i in range(self.T)]

            # Apply high-level interventions (if any)
            for t in range(self.T):
                if interventions:
                    if interventions.get("outage_facility", {}).get(fac, 0) == 1:
                        K_hist[t] = K_hist[t] * 0.7  # 30% capacity loss
                    if interventions.get("disruption_route", {}).get(fac, 0) == 1:
                        L_hist[t] = max(1, L_hist[t] + 1)
                        phi_hist[t] = max(0.2, phi_hist[t] * 0.8)

            # Period loop
            for t in range(self.T):
                # Demand and cost dynamics (toy):
                D_t = float(sub.iloc[t]["D"]) if t < len(sub) else 100.0
                dcost_t = float(sub.iloc[t]["dcost"]) if t < len(sub) else 0.0
                pbar = (float(sub.iloc[t]["p_bar"]) if t < len(sub) else 15.0) + float(self._price_adjust)

                # Receipts from earlier orders
                R[t] = self._arrivals(O, L_hist, phi_hist, t)

                # Shipments and inventory/backorders
                S_t = min(I[t - 1] + (R[t] if t > 0 else 0.0), D_t + (B[t - 1] if t > 0 else 0.0))
                if t > 0:
                    I[t] = I[t - 1] + R[t] - S_t
                    B[t] = max(0.0, D_t + B[t - 1] - (I[t - 1] + R[t]))

                # Demand response to price
                p_t = self._price_pass_through(pbar, dcost_t, B[t], I[t], float(sub.iloc[0]["D_ref"]))
                # Simple log-linear elasticity around reference
                D_t_eff = max(0.0, D_t * math.exp(-0.01 * (p_t - pbar)))

                # Lead-time estimate (M/M/1 rough cut)
                lam = D_t_eff
                mu = max(1e-6, K_hist[t] / max(1.0, len(self.skus)))
                L_eff = max(1.0, self._queueing_leadtime(lam, mu, transport=float(L_hist[t]) - 1.0))

                # Base-stock target and orders
                POS_t = I[t] + P[t]
                muL = L_eff
                sigL = 0.5 * muL
                target = muL * D_t_eff + self.policy.z_alpha * sigL * math.sqrt(max(1e-6, D_t_eff))
                O_policy = self._relu(target - POS_t)
                # Order smoothing
                O_prev = O[t - 1] if t > 0 else 0.0
                O[t] = (1 - self.policy.theta_smoothing) * O_prev + self.policy.theta_smoothing * O_policy

                # Update pipeline (very simplified)
                if t < self.T - 1:
                    P[t + 1] = max(0.0, P[t] + O[t] - R[t])

                # Capacity constraint (aggregate receipts)
                R[t] = min(R[t], K_hist[t])

                # Write back to df
                df.loc[(t, sku, fac), ["R", "I", "B", "O", "p", "L", "phi", "K"]] = [
                    R[t], I[t], B[t], O[t], p_t, L_eff, phi_hist[t], K_hist[t]
                ]

        return df

    # ===== Reporting =====
    def summarize(self, df: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # Simple KPIs
        service = 1.0 - (df["B"].groupby(level="t").mean() > 0).mean()
        holding_cost = df["I"].clip(lower=0).mean() * 0.1
        shortage_penalty = df["B"].mean() * 1.0
        ordering_cost = df["O"].mean() * df["c"].mean()
        cost = holding_cost + shortage_penalty + ordering_cost
        bwe = float((df["O"].var() / max(df["D"].var(), 1e-6)))
        out["service_level"] = float(service)
        out["cost_proxy"] = float(cost)
        out["bullwhip"] = bwe
        return out

    # ===== Causal runs (Upgraded) =====
    def causal_edges(self, df: pd.DataFrame) -> Dict[str, float]:
        """Fit causal edges and return strengths, now with enhanced analysis."""
        # Fit on panel averages per t
        panel = df.groupby(level="t").mean(numeric_only=True)
        vars_fit = [c for c in ["L", "phi", "K", "dcost", "B", "I", "p", "D", "O", "R"] if c in panel.columns]
        try:
            self.crca.fit_from_dataframe(panel, variables=vars_fit, window=min(30, len(panel)), decay_alpha=0.9, ridge_lambda=0.1)
        except Exception as e:
            logger.warning(f"CRCA fit skipped: {e}")
        strengths = {}
        for u, v in self.crca.causal_graph.edges():
            strengths[f"{u}->{v}"] = float(self.crca.causal_graph[u][v].get("strength", 0.0))
        return strengths

    def advanced_causal_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive advanced causal analysis using all upgraded CR-CA methods."""
        panel = df.groupby(level="t").mean(numeric_only=True)
        vars_fit = [c for c in ["L", "phi", "K", "dcost", "B", "I", "p", "D", "O", "R"] if c in panel.columns]
        
        # Fit causal graph first
        try:
            self.crca.fit_from_dataframe(panel, variables=vars_fit, window=min(30, len(panel)), decay_alpha=0.9, ridge_lambda=0.1)
        except Exception as e:
            logger.warning(f"CRCA fit skipped: {e}")
            return {"error": str(e)}
        
        # Get latest state for analysis
        latest = panel.iloc[-1] if len(panel) > 0 else {}
        factual_state = {var: float(latest.get(var, 0.0)) for var in vars_fit if var in latest}
        
        # ========== UPGRADED METHODS ==========
        results: Dict[str, Any] = {
            "causal_strengths": {},
            "sensitivity_analysis": {},
            "granger_causality": {},
            "information_theory": {},
            "var_model": {},
            "bayesian_edges": {},
            "root_causes": {},
            "shapley_attribution": {},
            "whatif_analysis": {},
            "optimal_intervention": {},
            "alternate_realities": {},
        }
        
        # Get causal strengths
        for u, v in self.crca.causal_graph.edges():
            results["causal_strengths"][f"{u}->{v}"] = float(self.crca.causal_graph[u][v].get("strength", 0.0))
        
        # 1. Sensitivity Analysis: What drives service/cost changes most?
        try:
            if len(factual_state) >= 4:
                intervention_vars = [v for v in ["L", "phi", "K", "dcost", "B"] if v in factual_state][:4]
                test_intervention = {k: factual_state.get(k, 0.0) for k in intervention_vars}
                
                # Analyze sensitivity to service (via B→p→D chain) and cost
                sensitivity_service = self.crca.sensitivity_analysis(
                    intervention=test_intervention,
                    target="B",  # Backlog affects service
                    perturbation_size=0.01,
                )
                sensitivity_cost = self.crca.sensitivity_analysis(
                    intervention={k: v for k, v in test_intervention.items() if k in ["dcost", "L", "phi"]},
                    target="p",  # Price affects cost
                    perturbation_size=0.01,
                )
                results["sensitivity_analysis"] = {
                    "service_drivers": sensitivity_service,
                    "cost_drivers": sensitivity_cost,
                }
                logger.info(f"Sensitivity: service={sensitivity_service.get('most_influential_variable', 'N/A')}, cost={sensitivity_cost.get('most_influential_variable', 'N/A')}")
        except Exception as e:
            logger.debug(f"Sensitivity analysis failed: {e}")
        
        # 2. Granger Causality: Temporal causal relationships
        try:
            if len(panel) >= 20:
                # Test if demand Granger-causes orders
                granger_d_o = self.crca.granger_causality_test(
                    df=panel,
                    var1="D",
                    var2="O",
                    max_lag=3,
                )
                # Test if backlog Granger-causes price
                granger_b_p = self.crca.granger_causality_test(
                    df=panel,
                    var1="B",
                    var2="p",
                    max_lag=2,
                )
                results["granger_causality"] = {
                    "demand_granger_causes_orders": granger_d_o.get("granger_causes", False),
                    "backlog_granger_causes_price": granger_b_p.get("granger_causes", False),
                    "d_o_f_stat": granger_d_o.get("f_statistic", 0.0),
                    "b_p_f_stat": granger_b_p.get("f_statistic", 0.0),
                }
                logger.info(f"Granger causality: D→O={granger_d_o.get('granger_causes', False)}, B→p={granger_b_p.get('granger_causes', False)}")
        except Exception as e:
            logger.debug(f"Granger causality test failed: {e}")
        
        # 3. Information Theoretic Measures
        try:
            if len(panel) >= 10:
                core_vars = [v for v in ["D", "O", "B", "I", "p", "L"] if v in panel.columns]
                if len(core_vars) >= 3:
                    info_theory = self.crca.compute_information_theoretic_measures(
                        df=panel,
                        variables=core_vars,
                    )
                    results["information_theory"] = info_theory
                    logger.info(f"Information theory: {len(info_theory.get('entropies', {}))} entropies computed")
        except Exception as e:
            logger.debug(f"Information theory computation failed: {e}")
        
        # 4. VAR Model: Vector Autoregression
        try:
            if len(panel) >= 30:
                var_vars = [v for v in ["D", "O", "I", "B", "p"] if v in panel.columns]
                if len(var_vars) >= 2:
                    var_model = self.crca.vector_autoregression_estimation(
                        df=panel,
                        variables=var_vars,
                        max_lag=2,
                    )
                    results["var_model"] = var_model
                    logger.info(f"VAR model: {var_model.get('n_variables', 0)} variables, lag={var_model.get('max_lag', 0)}")
        except Exception as e:
            logger.debug(f"VAR estimation failed: {e}")
        
        # 5. Bayesian Edge Inference
        try:
            bayesian_edges = {}
            key_edges = [("B", "p"), ("I", "p"), ("dcost", "p"), ("D", "O"), ("L", "R")]
            for parent, child in key_edges:
                if parent in panel.columns and child in panel.columns:
                    bayes_result = self.crca.bayesian_edge_inference(
                        df=panel,
                        parent=parent,
                        child=child,
                        prior_mu=0.0,
                        prior_sigma=1.0,
                    )
                    if "error" not in bayes_result:
                        bayesian_edges[f"{parent}->{child}"] = {
                            "posterior_mean": bayes_result.get("posterior_mean", 0.0),
                            "posterior_std": bayes_result.get("posterior_std", 0.0),
                            "credible_interval": bayes_result.get("credible_interval_95", (0.0, 0.0)),
                        }
            results["bayesian_edges"] = bayesian_edges
            if bayesian_edges:
                logger.info(f"Bayesian inference for {len(bayesian_edges)} edges")
        except Exception as e:
            logger.debug(f"Bayesian inference failed: {e}")
        
        # 6. Deep Root Cause Analysis: Find ultimate drivers of service/cost issues
        try:
            root_causes_service = self.crca.deep_root_cause_analysis(
                problem_variable="B",  # Backlog is service issue
                max_depth=8,
                min_path_strength=0.01,
            )
            root_causes_cost = self.crca.deep_root_cause_analysis(
                problem_variable="p",  # Price is cost proxy
                max_depth=8,
                min_path_strength=0.01,
            )
            results["root_causes"] = {
                "service_issues": root_causes_service,
                "cost_issues": root_causes_cost,
            }
            if root_causes_service.get("ultimate_root_causes"):
                logger.info(f"Root causes (service): {[rc.get('root_cause') for rc in root_causes_service.get('ultimate_root_causes', [])[:3]]}")
        except Exception as e:
            logger.debug(f"Root cause analysis failed: {e}")
        
        # 7. Shapley Value Attribution: Fair attribution of KPI drivers
        try:
            if len(panel) >= 7:
                # Baseline: average over last week
                baseline_state = {
                    k: float(panel[k].tail(7).mean()) 
                    for k in factual_state.keys() 
                    if k in panel.columns
                }
                if baseline_state and "B" in baseline_state:
                    shapley_backlog = self.crca.shapley_value_attribution(
                        baseline_state=baseline_state,
                        target_state=factual_state,
                        target="B",
                    )
                    if "p" in baseline_state:
                        shapley_price = self.crca.shapley_value_attribution(
                            baseline_state=baseline_state,
                            target_state=factual_state,
                            target="p",
                        )
                        results["shapley_attribution"] = {
                            "backlog_drivers": shapley_backlog,
                            "price_drivers": shapley_price,
                        }
                        logger.info(f"Shapley attribution computed for B and p")
        except Exception as e:
            logger.debug(f"Shapley attribution failed: {e}")
        
        # 8. Multi-layer What-If Analysis: Cascading effects of disruptions
        try:
            test_scenarios = [
                {"L": factual_state.get("L", 2.0) * 1.5},  # Lead time disruption
                {"phi": max(0.2, factual_state.get("phi", 1.0) * 0.7)},  # Survival rate drop
                {"dcost": factual_state.get("dcost", 0.0) + 2.0},  # Cost shock
            ]
            whatif_analysis = self.crca.multi_layer_whatif_analysis(
                scenarios=test_scenarios,
                depth=3,
            )
            results["whatif_analysis"] = whatif_analysis
            logger.info(f"What-if analysis: {whatif_analysis.get('summary', {}).get('total_scenarios', 0)} scenarios")
        except Exception as e:
            logger.debug(f"What-if analysis failed: {e}")
        
        # 9. Optimal Intervention Sequence: Bellman optimization
        try:
            optimal_intervention = self.crca.bellman_optimal_intervention(
                initial_state=factual_state,
                target="B",  # Minimize backlog (maximize service)
                intervention_vars=["L", "phi", "K", "dcost"],
                horizon=5,
                discount=0.9,
            )
            results["optimal_intervention"] = optimal_intervention
            if optimal_intervention.get("optimal_sequence"):
                logger.info(f"Optimal intervention sequence: {len(optimal_intervention['optimal_sequence'])} steps")
        except Exception as e:
            logger.debug(f"Optimal intervention failed: {e}")
        
        # 10. Explore Alternate Realities: Best intervention scenarios
        try:
            alternate_realities = self.crca.explore_alternate_realities(
                factual_state=factual_state,
                target_outcome="B",  # Minimize backlog
                target_value=0.0,  # Target zero backlog
                max_realities=30,
                max_interventions=3,
            )
            results["alternate_realities"] = alternate_realities
            if alternate_realities.get("best_reality"):
                improvement = factual_state.get("B", 0.0) - alternate_realities["best_reality"].get("target_value", 0.0)
                logger.info(f"Best alternate reality: {improvement:+.2f} backlog reduction")
        except Exception as e:
            logger.debug(f"Alternate realities exploration failed: {e}")
        
        # 11. Cascading Chain Reaction Analysis
        try:
            if "L" in factual_state:
                chain_reaction = self.crca.analyze_cascading_chain_reaction(
                    initial_intervention={"L": factual_state.get("L", 2.0) * 1.5},
                    target_outcomes=["B", "I", "O", "p", "D"],
                    max_hops=6,
                    include_feedback_loops=True,
                    num_iterations=4,
                )
                results["chain_reaction"] = chain_reaction
                logger.info(f"Chain reaction analysis: {chain_reaction.get('summary', {}).get('total_paths_found', 0)} paths")
        except Exception as e:
            logger.debug(f"Chain reaction analysis failed: {e}")
        
        # 12. Cross-validation for edge strength reliability
        try:
            cv_results = {}
            for parent, child in [("B", "p"), ("D", "O"), ("L", "R")]:
                if parent in panel.columns and child in panel.columns:
                    cv = self.crca.cross_validate_edge_strength(
                        df=panel,
                        parent=parent,
                        child=child,
                        n_folds=5,
                    )
                    if "error" not in cv:
                        cv_results[f"{parent}->{child}"] = {
                            "mean_cv_error": cv.get("mean_cv_error", 0.0),
                            "standard_error": cv.get("standard_error", 0.0),
                        }
            results["cross_validation"] = cv_results
            if cv_results:
                logger.info(f"Cross-validation for {len(cv_results)} edges")
        except Exception as e:
            logger.debug(f"Cross-validation failed: {e}")
        
        return results

    def intervene_and_compare(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        base = self.simulate()
        base_kpi = self.summarize(base)
        shock = self.simulate(interventions=scenario)
        shock_kpi = self.summarize(shock)
        delta = {k: float(shock_kpi.get(k, 0.0) - base_kpi.get(k, 0.0)) for k in base_kpi}
        return {"base": base_kpi, "shock": shock_kpi, "delta": delta}

    # ===== VSM overlay orchestration =====
    def control_cycle(
        self,
        telemetry_events: Optional[List[Dict[str, Any]]] = None,
        intel_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """One control cycle over S2–S5: ingest telemetry, monitor, propose, audit, gate."""
        # Ingest telemetry/intel (stubs)
        # Dynamic feeds first
        _ = self.poll_feeds()
        if telemetry_events:
            self.vsm.ingest_s1_flow(self.df, telemetry_events)
        if intel_events:
            self.vsm.ingest_s4_external(intel_events)

        # Simulate baseline for the cycle
        df = self.simulate()
        panel = df.groupby(level="t").mean(numeric_only=True)

        # S2 stability monitors
        s2 = self.vsm.s2_monitor(panel)

        # S3 propose optimizers only if S2 stable
        proposal = None
        if s2.get("stable", False):
            proposal = self.vsm.s3_optimize(panel, self.policy)
            # SPC/EWMA-driven mode switch: if EWMA breaches on backlog, bias to higher service target
            backlog_alerts = [a for a in s2.get("alerts", []) if a.get("type") == "ewma" and a.get("signal") == "B"]
            target_service = 0.95 + (0.02 if backlog_alerts else 0.0)
            z_cal = self.calibrate_z_to_service(target_service=target_service)
            proposal["z_alpha_new"] = float(z_cal)

        # S3★ audit
        audit = self.vsm.s3_star_audit(panel)

        # S5 gate with policy caps (LLM-led proposals are still safety-gated here)
        decision = self.vsm.s5_policy_gate(proposal, s2, audit, risk_cap_cvar=0.15)

        # Build causal strengths snapshot and LLM narratives
        strengths = self.causal_edges(df)
        # Build strength-aware explanation preface
        top_edges = sorted(strengths.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        explainer = {k: v for k, v in top_edges}
        # Compute weak edges to avoid claiming
        weak_edges = {k: v for k, v in strengths.items() if abs(v) <= 0.05}
        # Prepare concise narrative prompt for the agent
        kpis = self.summarize(df)
        alerts = s2.get("alerts", [])
        proposal_text = str(proposal) if proposal else "None"
        decision_text = str(decision)
        narrative_prompt = (
            f"Analyze supply chain state and propose safe actions.\n"
            f"KPIs: service={kpis.get('service_level', 0.0):.2f}, cost_proxy={kpis.get('cost_proxy', 0.0):.2f}, bullwhip={kpis.get('bullwhip', 0.0):.2f}.\n"
            f"Alerts: {alerts}.\n"
            f"Proposal: {proposal_text}. Decision: {decision_text}.\n"
            f"Causal edges (mirror these in explanation): {explainer}.\n"
            f"Do NOT claim effects via edges with near-zero strength: {list(weak_edges.keys())}.\n"
            f"Provide the 7-section output per instructions."
        )
        # Multi-round LLM-led decision loop
        ai_analysis, proposal = self._llm_decision_loop(
            panel=panel,
            s2=s2,
            proposal=proposal,
            decision=decision,
            strengths=strengths,
            kpis=kpis,
            rounds=3,
        )

        # Feasibility check and quantified impact
        feasibility_flag = False
        impact = None
        if isinstance(proposal, dict):
            # Gate logistics levers by learned DAG support
            allow_logistics = self._logistics_support_from_strengths(strengths, tol=0.05)
            allow_price = (abs(float(strengths.get("B->p", 0.0))) > 0.05) and (abs(float(strengths.get("p->D", 0.0))) > 0.05)
            if not allow_logistics:
                proposal["reroute_share"] = 0.0
                proposal["expedite_share"] = 0.0
            if not allow_price:
                proposal["price_adjust"] = 0.0
            else:
                proposal.setdefault("price_adjust", 0.0)
            feasibility_flag = self._feasible(proposal)
            z_new = float(proposal.get("z_alpha_new", self.policy.z_alpha))
            rr = float(proposal.get("reroute_share", 0.0))
            ex = float(proposal.get("expedite_share", 0.0))
            price_adj = float(proposal.get("price_adjust", 0.0))
            impact = self._quantified_impact(z_new, rr, ex, trials=50, price_adjust=price_adj)
            # auto de-risk actions on red tier or excessive cost
            if s2.get("tier") == "red" or (impact and impact.get("expected", {}).get("cost_proxy", 0.0) > 0):
                # freeze increases to expedite/reroute
                proposal["expedite_share"] = min(float(proposal.get("expedite_share", 0.0)), self.last_controls.get("expedite_share", 0.0))
                proposal["reroute_share"] = min(float(proposal.get("reroute_share", 0.0)), self.last_controls.get("reroute_share", 0.0))
            # Optional: refine via constrained MPC if feasible
            if feasibility_flag:
                mpc = constrained_mpc_policy(
                    last=self.last_controls,
                    limits=self.max_weekly_change,
                    lane_cap=self.lane_capacity,
                    weights=self.mpc_weights,
                    impact=impact,
                )
                if "error" not in mpc:
                    proposal.update({k: v for k, v in mpc.items() if k in ["z_alpha_new", "reroute_share", "expedite_share", "dual_source_share"]})
                # Final hard-feasibility polish via CP-SAT
                cps = cpsat_policy(self.last_controls, self.max_weekly_change, self.lane_capacity)
                if "error" not in cps:
                    proposal.update({k: v for k, v in cps.items() if k in ["z_alpha_new", "reroute_share", "expedite_share", "dual_source_share"]})
                # CVaR neighborhood selection
                cvar_pick = self.cvar_select_from_grid(proposal, alpha=0.9, eps=0.05, trials=30, weights=self.mpc_weights)
                if cvar_pick.get("choice"):
                    proposal.update(cvar_pick["choice"])
                
                # UPGRADED: Try gradient-based optimization for refinement
                try:
                    # Use current state as initial for gradient optimization
                    current_state = {
                        "z_alpha": z_new,
                        "reroute_share": rr,
                        "expedite_share": ex,
                        "B": float(panel.get("B", pd.Series([0.0])).iloc[-1]),
                        "I": float(panel.get("I", pd.Series([120.0])).iloc[-1]),
                        "p": float(panel.get("p", pd.Series([15.0])).iloc[-1]),
                    }
                    
                    # Optimize z_alpha to minimize backlog (via gradient)
                    if "B" in panel.columns:
                        opt_result = self.crca.gradient_based_intervention_optimization(
                            initial_state=current_state,
                            target="B",  # Minimize backlog
                            intervention_vars=["z_alpha"],
                            constraints={"z_alpha": (0.5, 3.0)},
                            method="L-BFGS-B",
                        )
                        if opt_result.get("success") and opt_result.get("optimal_intervention"):
                            # Refine z_alpha if gradient optimization suggests improvement
                            opt_z = opt_result["optimal_intervention"].get("z_alpha", z_new)
                            if abs(opt_z - z_new) < 0.3:  # Within rate limit
                                proposal["z_alpha_new"] = float(opt_z)
                                logger.info(f"Gradient optimization refined z_alpha: {z_new:.2f} → {opt_z:.2f}")
                except Exception as e:
                    logger.debug(f"Gradient optimization failed: {e}")
                
                # Provide minimal Pareto frontier for transparency
                pareto = self.pareto_front(base_z=z_new, allow_logistics=allow_logistics, trials=20, allow_price=allow_price)
            else:
                pareto = []
        else:
            # Infeasible: rollback immediately
            if self.action_history:
                last = self.action_history[-1]
                self.policy.z_alpha = float(last.get("z_alpha_prev", self.policy.z_alpha))
                self.last_controls["z_alpha_new"] = self.policy.z_alpha
                for k in ["reroute_share", "expedite_share", "dual_source_share"]:
                    self.last_controls[k] = max(0.0, self.last_controls.get(k, 0.0) * 0.5)
            pareto = []

        # Apply approved proposal to live policy (LLM primacy) after safety gate
        if isinstance(proposal, dict) and feasibility_flag:
            decision = self.vsm.s5_policy_gate(proposal, s2, audit, risk_cap_cvar=0.15)
            if decision.get("approved"):
                self.policy.z_alpha = float(proposal.get("z_alpha_new", self.policy.z_alpha))
                # persist last controls for rate-limit checks
                self.last_controls = {
                    "z_alpha_new": self.policy.z_alpha,
                    "reroute_share": float(proposal.get("reroute_share", 0.0)),
                    "expedite_share": float(proposal.get("expedite_share", 0.0)),
                }
                # record action for possible rollback and RL
                self.action_history.append({
                    "z_alpha_prev": float(self.last_controls["z_alpha_new"]),
                    "reroute_share": float(self.last_controls["reroute_share"]),
                    "expedite_share": float(self.last_controls["expedite_share"]),
                })
                if impact and impact.get("expected"):
                    self.reinforce_from_outcome(impact["expected"])

        # Advanced causal analysis using upgraded methods
        advanced_causal = {}
        try:
            advanced_causal = self.advanced_causal_analysis(df)
            logger.info(f"Advanced causal analysis completed: {len([k for k, v in advanced_causal.items() if v and 'error' not in str(v)])} methods succeeded")
        except Exception as e:
            logger.debug(f"Advanced causal analysis failed: {e}")
        
        # Human-readable causal summary via CR-CA agent (enhanced with upgraded insights)
        try:
            sensitivity_note = ""
            if advanced_causal.get("sensitivity_analysis"):
                sens_svc = advanced_causal["sensitivity_analysis"].get("service_drivers", {})
                sens_cost = advanced_causal["sensitivity_analysis"].get("cost_drivers", {})
                most_influential_svc = sens_svc.get("most_influential_variable", "N/A")
                most_influential_cost = sens_cost.get("most_influential_variable", "N/A")
                sensitivity_note = f" Sensitivity analysis shows {most_influential_svc} most drives service, {most_influential_cost} most drives cost."
            
            granger_note = ""
            if advanced_causal.get("granger_causality"):
                gc = advanced_causal["granger_causality"]
                if gc.get("demand_granger_causes_orders"):
                    granger_note = " Granger causality: D→O confirmed. "
                if gc.get("backlog_granger_causes_price"):
                    granger_note += "B→p temporally confirmed."
            
            summary_prompt = (
                f"Summarize key drivers (B→p, I→p, dcost→p, L→R) and their implications.{sensitivity_note}{granger_note} "
                f"Reference learned strengths: {dict(list(strengths.items())[:5])}."
            )
            causal_summary = self.crca.run(summary_prompt)
        except Exception as e:
            causal_summary = f"causal_unavailable: {e}"

        return {
            "s2": s2,
            "proposal": proposal,
            "audit": audit,
            "decision": decision,
            "kpis": kpis,
            "causal_strengths": strengths,
            "ai_analysis": ai_analysis,
            "causal_summary": causal_summary,
            "feasible": feasibility_flag,
            "impact": impact,
            "pareto_front": pareto,
            "z_service_curve": self.z_service_curve(),
            # Upgraded CR-CA analysis results
            "advanced_causal": advanced_causal,
            # Advanced hooks (stubs returning diagnostics)
            "estimation": {
                "ms_filter": ms_regime_filter(panel),
                "bsts_nowcast": bsts_nowcast(panel),
                "svar": svar_identification(panel),
            },
            "robust_optimization": {
                "dro_mpc": dro_mpc_plan(panel),
                "chance_mpc": chance_constrained_mpc(panel),
                "h_infinity": h_infinity_controller(panel),
                "sddp": sddp_policy_stub(panel),
            },
            "multi_echelon": {
                "clark_scarf": clark_scarf_baseline(panel),
                "risk_newsvendor": risk_averse_newsvendor_stub(panel),
            },
            "network_risk": {
                "percolation": percolation_stub(panel),
                "eisenberg_noe": eisenberg_noe_stub(panel),
                "k_cuts": k_cut_sets_stub(),
            },
            "pricing": {
                "logit": logit_share_stub(panel),
                "dp_pricing": dynamic_pricing_dp_stub(panel),
                "ramsey": ramsey_pricing_stub(panel),
            },
            "policy_eval": {
                "synth_control": synthetic_control_stub(panel),
                "did_iv": did_iv_stub(panel),
            },
            "security": {
                "secure_state": secure_state_l1_stub(panel),
                "spectral": spectral_shift_stub(panel),
            },
            "investment": {
                "real_options": real_options_stub(),
                "supplier_portfolio": supplier_portfolio_stub(),
            },
            "advanced2": {
                "transport_modes": transport_modes_sim_stub(),
                "online_em_ms": online_em_ms(panel),
                "hier_bsts": hierarchical_bsts_stub(panel),
                "multi_stage_mpc": multi_stage_mpc_stub(panel),
                "iot_fusion": iot_fusion_stub(),
                "nested_logit": nested_logit_stub(),
                "kcut_hardening": kcut_hardening_stub(),
                "linucb_policy": linucb_policy_stub(panel),
                "l1_observer": l1_observer_stub(panel),
                "fan_charts": fan_charts_stub(panel),
                "async_ingest": {"kafka": True, "duckdb": True}
            }
        }


# ===== Viable System Overlay (S2–S5) =====
class VSMOverlay:
    def __init__(self) -> None:
        self.state: Dict[str, Any] = {"R_posterior": np.array([1.0])}

    # Telemetry ingest (stubs)
    def ingest_s1_flow(self, df: pd.DataFrame, events: List[Dict[str, Any]]) -> None:
        # Example event schema documented; here we could append/merge into df
        # For now we no-op (upgrade path to Kafka/Timescale later)
        pass

    def ingest_s4_external(self, events: List[Dict[str, Any]]) -> None:
        # Store the latest intel snapshot
        self.state["intel_last"] = events[-1] if events else None

    # S2: Stability/coordination
    def s2_monitor(self, panel: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {"alerts": [], "stable": True, "tier": "green"}
        # EWMA residuals on key KPIs
        for col in [c for c in ["L", "I", "B"] if c in panel.columns]:
            z, breaches = ewma_monitor(panel[col].values, lam=0.2, k=3.0)
            if breaches > 0:
                out["alerts"].append({"type": "ewma", "signal": col, "breaches": breaches})
        # CUSUM on lead time
        if "L" in panel.columns:
            plus, minus, trips = cusum(panel["L"].values, mu0=float(panel["L"].head(5).mean()), k=0.1, h=5.0)
            if trips:
                out["alerts"].append({"type": "cusum", "signal": "L", "trips": len(trips)})
        # Page-Hinkley on demand
        if "D" in panel.columns and len(panel["D"]) > 20:
            ph = page_hinkley(panel["D"].astype(float).values)
            if ph["alarm"]:
                out["alerts"].append({"type": "page_hinkley", "signal": "D", "mT": ph["mT"]})
        # BOCPD change-point minimal
        if "D" in panel.columns:
            p_break = bocpd_break_prob(panel["D"].values, hazard=0.02)
            if p_break > 0.5:
                out["alerts"].append({"type": "bocpd", "p_break": p_break})
        # Queueing sanity
        if set(["D", "K"]).issubset(panel.columns):
            lam = float(max(1e-6, panel["D"].iloc[-1]))
            mu = float(max(1e-6, panel["K"].iloc[-1]))
            rho = lam / mu
            if rho > 0.85:
                out["alerts"].append({"type": "utilization", "rho": rho})
        # Stable if few alerts; tiering
        n_alerts = len(out["alerts"])
        out["stable"] = n_alerts == 0
        if n_alerts == 0:
            out["tier"] = "green"
        elif n_alerts < 3:
            out["tier"] = "yellow"
        else:
            out["tier"] = "red"
        return out

    # S3: Optimizers (stubs)
    def s3_optimize(self, panel: pd.DataFrame, policy: PolicyParams) -> Dict[str, Any]:
        # Heuristic constrained optimizer proxy; prefer stability
        z = policy.z_alpha
        rr = 0.0
        ex = 0.05
        ds = 0.0
        if "B" in panel.columns:
            b_trend = float(np.polyfit(np.arange(len(panel)), panel["B"].values, 1)[0]) if len(panel) >= 5 else 0.0
            z = float(np.clip(z + 0.05 * np.sign(b_trend), 0.5, 2.5))
        if self.state.get("intel_last"):
            rr = 0.1
        return {"z_alpha_new": z, "reroute_share": rr, "expedite_share": ex, "dual_source_share": ds}

    # S3★: Audit
    def s3_star_audit(self, panel: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if set(["O", "D"]).issubset(panel.columns):
            varO = float(panel["O"].var())
            varD = float(panel["D"].var())
            out["BWE"] = float(varO / max(varD, 1e-6))
        # Simple receipts cross-check stub (noisy)
        out["receipts_delta_sigma"] = 0.0
        return out

    # S5: Policy gate
    def s5_policy_gate(
        self,
        proposal: Optional[Dict[str, Any]],
        s2: Dict[str, Any],
        audit: Dict[str, Any],
        risk_cap_cvar: float = 0.2,
    ) -> Dict[str, Any]:
        if proposal is None:
            return {"approved": False, "reason": "no_proposal"}
        if not s2.get("stable", False):
            return {"approved": False, "reason": "unstable_S2"}
        if audit.get("BWE", 1.0) > 1.3:
            # tighten smoothing recommendation by halving expedite/reroute
            proposal = {**proposal, "expedite_share": proposal.get("expedite_share", 0.0) * 0.5, "reroute_share": proposal.get("reroute_share", 0.0) * 0.5}
        return {"approved": True, "proposal": proposal}


# ===== Detectors (S2) =====
def ewma_monitor(x: np.ndarray, lam: float = 0.2, k: float = 3.0) -> Tuple[np.ndarray, int]:
    z = np.zeros_like(x, dtype=float)
    z[0] = x[0]
    for t in range(1, len(x)):
        z[t] = lam * x[t] + (1 - lam) * z[t - 1]
    resid = x - z
    sigma = np.std(resid[max(1, len(resid)//10):]) or 1.0
    breaches = int(np.sum(np.abs(resid) > k * sigma))
    return z, breaches


def cusum(x: np.ndarray, mu0: float, k: float, h: float) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    s_plus = np.zeros_like(x, dtype=float)
    s_minus = np.zeros_like(x, dtype=float)
    trips: List[int] = []
    for t in range(1, len(x)):
        s_plus[t] = max(0.0, s_plus[t - 1] + (x[t] - mu0 - k))
        s_minus[t] = max(0.0, s_minus[t - 1] + (mu0 - x[t] - k))
        if s_plus[t] > h or s_minus[t] > h:
            trips.append(t)
            s_plus[t] = 0.0
            s_minus[t] = 0.0
    return s_plus, s_minus, trips


def bocpd_break_prob(x: np.ndarray, hazard: float = 0.02) -> float:
    # Minimal BOCPD: approximate break prob by normalized absolute diff of recent means with hazard weight
    if len(x) < 10:
        return 0.0
    n = len(x)
    m1 = float(np.mean(x[: n // 2]))
    m2 = float(np.mean(x[n // 2 :]))
    delta = abs(m2 - m1) / (np.std(x) or 1.0)
    p = 1.0 - math.exp(-hazard * delta)
    return float(np.clip(p, 0.0, 1.0))


def page_hinkley(x: np.ndarray, delta: float = 0.005, lamb: float = 50.0, alpha: float = 1.0) -> Dict[str, Any]:
    # Minimal Page-Hinkley drift detector
    mean = 0.0
    mT = 0.0
    MT = 0.0
    alarm = False
    for i, xi in enumerate(x):
        mean = mean + (xi - mean) / (i + 1)
        mT = mT + xi - mean - delta
        MT = min(MT, mT)
        if mT - MT > lamb * alpha:
            alarm = True
            break
    return {"alarm": alarm, "mT": float(mT)}


# ===== Advanced Estimation (stubs) =====
def ms_regime_filter(panel: pd.DataFrame) -> Dict[str, Any]:
    """Two-regime Markov-switching on demand using statsmodels MarkovRegression.

    Returns current regime probabilities and smoothed prediction for next step.
    """
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        if "D" not in panel.columns or len(panel) < 20:
            return {"regime": "insufficient", "p_regime": {}}
        y = panel["D"].astype(float).values
        # Fit a simple mean-switching model with 2 regimes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = MarkovRegression(y, k_regimes=2, trend='c', switching_variance=True)
            res = mod.fit(disp=False)
        p_t = res.smoothed_marginal_probabilities.values[:, -1] if hasattr(res.smoothed_marginal_probabilities, 'values') else res.smoothed_marginal_probabilities
        p_reg1 = float(p_t[-1])
        regime = "shock" if p_reg1 > 0.5 else "normal"
        return {"regime": regime, "p_regime": {"regime1": p_reg1, "regime0": 1 - p_reg1}}
    except Exception as e:
        return {"regime": "error", "error": str(e)}


def bsts_nowcast(panel: pd.DataFrame) -> Dict[str, Any]:
    """Dynamic factor nowcast for demand using statsmodels DynamicFactor (FAVAR-like)."""
    try:
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
        cols = [c for c in ["D", "R", "I", "B", "O"] if c in panel.columns]
        if len(cols) < 2 or len(panel) < 20:
            mu = float(panel.get("D", pd.Series([100.0])).iloc[-1])
            return {"D_nowcast": mu, "uncertainty": 0.2, "cols": cols}
        endog = panel[cols].astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = DynamicFactor(endog, k_factors=1, factor_order=1)
            res = mod.fit(disp=False)
        # One-step ahead forecast for demand
        fc = res.get_forecast(steps=1)
        d_idx = cols.index("D")
        d_mean = float(fc.predicted_mean.iloc[0, d_idx])
        d_var = float(fc.var_pred_mean.iloc[0, d_idx]) if hasattr(fc, 'var_pred_mean') else 0.2
        return {"D_nowcast": d_mean, "uncertainty": max(0.05, min(0.5, d_var ** 0.5)), "cols": cols}
    except Exception as e:
        mu = float(panel.get("D", pd.Series([100.0])).iloc[-1])
        return {"D_nowcast": mu, "uncertainty": 0.3, "error": str(e)}


def svar_identification(panel: pd.DataFrame) -> Dict[str, Any]:
    """Estimate a small VAR and report impulse responses as proxy for shocks."""
    try:
        from statsmodels.tsa.api import VAR
        cols = [c for c in ["D", "R", "I", "B", "p"] if c in panel.columns]
        if len(cols) < 3 or len(panel) < 30:
            return {"irf": {}, "cols": cols}
        endog = panel[cols].astype(float)
        model = VAR(endog)
        res = model.fit(maxlags=2, ic='aic')
        irf = res.irf(5)
        irf_dict = {f"{src}->{dst}": float(irf.irfs[1, cols.index(src), cols.index(dst)]) for src in cols for dst in cols}
        return {"irf_h1": irf_dict, "cols": cols}
    except Exception as e:
        return {"error": str(e)}


# ===== Robust Optimization & Control (stubs) =====
def dro_mpc_plan(panel: pd.DataFrame) -> Dict[str, Any]:
    """Single-variable DRO-style MPC for z_alpha using cvxpy.

    Minimize holding/backorder proxy under Wasserstein penalty.
    """
    try:
        import cvxpy as cp
        from scipy.stats import norm
        D = float(panel.get("D", pd.Series([100.0])).iloc[-1])
        muL = float(panel.get("L", pd.Series([2.0])).iloc[-1])
        sigL = max(0.2, 0.5 * muL)
        z = cp.Variable()
        # Proxy cost: holding ~ z, backlog ~ max(0, Phi^{-1}(0.95)-z)
        z_req = norm.ppf(0.95)
        backlog_short = cp.pos(z_req - z)
        cost = 0.1 * z + 0.5 * backlog_short
        # DRO penalty
        rho = 0.1
        objective = cp.Minimize(cost + rho * cp.abs(z))
        constraints = [z >= 0.5, z <= 3.0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, warm_start=True)
        return {"type": "DRO-MPC", "z_alpha": float(z.value), "status": prob.status, "rho": rho}
    except Exception as e:
        return {"type": "DRO-MPC", "error": str(e)}


def constrained_mpc_policy(
    last: Dict[str, float],
    limits: Dict[str, float],
    lane_cap: float,
    weights: Dict[str, float],
    impact: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Optimize z, r, e, ds under caps and rate limits to minimize weighted proxy cost
    try:
        import cvxpy as cp
        z = cp.Variable()
        r = cp.Variable()
        e = cp.Variable()
        ds = cp.Variable()
        # Objective: weights over expected deltas if available, else regularization
        if impact and "expected" in impact:
            exp = impact["expected"]
            svc = exp.get("service", 0.0)
            cst = exp.get("cost_proxy", 0.0)
            bwe = exp.get("bullwhip", 0.0)
            obj = weights.get("cost", 1.0) * cst + weights.get("service", 1.0) * (-svc) + weights.get("bullwhip", 1.0) * bwe
        else:
            obj = 0.1 * (z - last.get("z_alpha_new", 1.28)) ** 2 + 0.05 * (r - last.get("reroute_share", 0.0)) ** 2 + 0.05 * (e - last.get("expedite_share", 0.0)) ** 2 + 0.05 * (ds - last.get("dual_source_share", 0.0)) ** 2
        constraints = [
            z >= 0.5, z <= 3.0,
            r >= 0.0, e >= 0.0, ds >= 0.0,
            r + e + ds <= lane_cap,
            cp.abs(z - last.get("z_alpha_new", 1.28)) <= limits.get("z_alpha_new", 0.3),
            cp.abs(r - last.get("reroute_share", 0.0)) <= limits.get("reroute_share", 0.15),
            cp.abs(e - last.get("expedite_share", 0.0)) <= limits.get("expedite_share", 0.15),
            cp.abs(ds - last.get("dual_source_share", 0.0)) <= limits.get("dual_source_share", 0.2),
        ]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.ECOS_BB, mi_max_iterations=1000)
        return {"z_alpha_new": float(z.value), "reroute_share": float(r.value), "expedite_share": float(e.value), "dual_source_share": float(ds.value), "status": prob.status}
    except Exception as e:
        return {"error": str(e)}


def cpsat_policy(
    last: Dict[str, float],
    limits: Dict[str, float],
    lane_cap: float,
) -> Dict[str, Any]:
    """Hard-feasibility projection using cvxpy (no OR-Tools).

    Minimizes squared distance to last controls subject to rate and capacity limits.
    """
    try:
        import cvxpy as cp
        z = cp.Variable()
        r = cp.Variable()
        e = cp.Variable()
        ds = cp.Variable()
        # Objective: keep close to last (stability)
        obj = (z - last.get("z_alpha_new", 1.28)) ** 2 + (r - last.get("reroute_share", 0.0)) ** 2 + (e - last.get("expedite_share", 0.0)) ** 2 + (ds - last.get("dual_source_share", 0.0)) ** 2
        cons = [
            z >= 0.5, z <= 3.0,
            r >= 0.0, e >= 0.0, ds >= 0.0,
            r + e + ds <= lane_cap,
            cp.abs(z - last.get("z_alpha_new", 1.28)) <= limits.get("z_alpha_new", 0.3),
            cp.abs(r - last.get("reroute_share", 0.0)) <= limits.get("reroute_share", 0.15),
            cp.abs(e - last.get("expedite_share", 0.0)) <= limits.get("expedite_share", 0.15),
            cp.abs(ds - last.get("dual_source_share", 0.0)) <= limits.get("dual_source_share", 0.2),
        ]
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=cp.ECOS)
        if z.value is None:
            return {"error": "infeasible"}
        return {
            "z_alpha_new": float(z.value),
            "reroute_share": float(r.value),
            "expedite_share": float(e.value),
            "dual_source_share": float(ds.value),
            "status": prob.status,
        }
    except Exception as e:
        return {"error": str(e)}


def chance_constrained_mpc(panel: pd.DataFrame) -> Dict[str, Any]:
    """Closed-form chance constraint for service level: z >= Phi^{-1}(beta)."""
    try:
        from scipy.stats import norm
        beta = 0.95
        z_req = float(norm.ppf(beta))
        return {"type": "Chance-MPC", "beta": beta, "z_min": z_req, "feasible": True}
    except Exception as e:
        return {"type": "Chance-MPC", "error": str(e)}


def h_infinity_controller(panel: pd.DataFrame) -> Dict[str, Any]:
    """Approximate robust controller via discrete LQR (DARE) for inventory linearization."""
    try:
        from scipy.linalg import solve_discrete_are
        # Simple 1D: x_{t+1} = x_t - u_t + w_t
        A = np.array([[1.0]])
        B = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[0.5]])
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
        gamma = float(np.sqrt(np.max(np.linalg.eigvals(P))))
        return {"type": "LQR", "K": float(K), "gamma_proxy": gamma}
    except Exception as e:
        return {"type": "LQR", "error": str(e)}


def sddp_policy_stub(panel: pd.DataFrame) -> Dict[str, Any]:
    return {"type": "SDDP", "stages": 3, "status": "ok"}


# ===== Multi-Echelon Inventory (stubs) =====
def clark_scarf_baseline(panel: pd.DataFrame) -> Dict[str, Any]:
    """Compute a simple echelon base-stock as I + mean(P) across time (proxy)."""
    try:
        if set(["I"]).issubset(panel.columns):
            echelon = float(panel["I"].mean()) + float(panel.get("P", pd.Series([0.0])).mean())
            return {"echelon_base_stock": echelon}
        return {"echelon_base_stock": 0.0}
    except Exception as e:
        return {"error": str(e)}


def risk_averse_newsvendor_stub(panel: pd.DataFrame) -> Dict[str, Any]:
    """Entropic-risk newsvendor: grid search Q to minimize 1/theta log E[exp(theta*cost)]."""
    try:
        D = panel.get("D", pd.Series([100.0])).astype(float).values
        if len(D) < 10:
            return {"Q_star": float(np.percentile(D, 95)), "risk_measure": "entropic", "theta": 0.5}
        theta = 0.5
        c_h, c_b = 0.1, 0.5
        Q_grid = np.linspace(np.percentile(D, 10), np.percentile(D, 99), 30)
        best, bestQ = 1e9, Q_grid[0]
        for Q in Q_grid:
            cost = c_h * np.maximum(Q - D, 0.0) + c_b * np.maximum(D - Q, 0.0)
            risk = (1.0 / theta) * np.log(np.mean(np.exp(theta * cost)))
            if risk < best:
                best, bestQ = risk, Q
        return {"Q_star": float(bestQ), "risk_measure": "entropic", "theta": theta, "objective": float(best)}
    except Exception as e:
        return {"error": str(e)}


# ===== Digital twin gating scenarios =====
def run_gating_scenarios(agent: "SupplyShockCRCAgent") -> Dict[str, Any]:
    grid = [
        {"name": "baseline", "z": agent.policy.z_alpha, "r": 0.0, "e": 0.0},
        {"name": "z1.6_r0.20_e0.10", "z": 1.6, "r": 0.20, "e": 0.10},
        {"name": "z1.8_r0.15_e0.15", "z": 1.8, "r": 0.15, "e": 0.15},
        {"name": "z2.0_e0.20", "z": 2.0, "r": 0.00, "e": 0.20},
    ]
    results: Dict[str, Any] = {}
    base = agent.simulate()
    base_kpi = agent.summarize(base)
    for g in grid:
        impact = agent._quantified_impact(g["z"], g["r"], g["e"], trials=50)
        exp = impact.get("expected", {})
        svc_delta = float(exp.get("service", 0.0))
        cost_delta = float(exp.get("cost_proxy", 0.0))
        marginal_cost_per_pp = float(cost_delta / max(1e-6, svc_delta * 100.0)) if svc_delta > 0 else float("inf")
        results[g["name"]] = {
            "expected": exp,
            "ci80": impact.get("ci80", {}),
            "marginal_cost_per_1pp_service": marginal_cost_per_pp,
        }
    return {"baseline": base_kpi, "scenarios": results}


# ===== Network & Systemic Risk (stubs) =====
def percolation_stub(panel: pd.DataFrame) -> Dict[str, Any]:
    return {"gc_threshold": 0.3, "expected_shortfall": 0.12}


def eisenberg_noe_stub(panel: pd.DataFrame) -> Dict[str, Any]:
    return {"clearing_vector_norm": 0.97}


def k_cut_sets_stub() -> Dict[str, Any]:
    return {"min_k_cut": 2, "critical_arcs": ["lane_CN-EU", "lane_US-EU"]}


# ===== Pricing & Demand (stubs) =====
def logit_share_stub(panel: pd.DataFrame) -> Dict[str, Any]:
    return {"alpha_price": 0.8, "elasticity": -1.2}


def dynamic_pricing_dp_stub(panel: pd.DataFrame) -> Dict[str, Any]:
    return {"policy": "approxDP", "discount": 0.98}


def ramsey_pricing_stub(panel: pd.DataFrame) -> Dict[str, Any]:
    return {"lambda": 0.2, "implied_markups": {"A": 0.1, "B": 0.05}}


# ===== Policy Evaluation (stubs) =====
def synthetic_control_stub(panel: pd.DataFrame) -> Dict[str, Any]:
    return {"effect": -0.03, "weight_entropy": 0.9}


def did_iv_stub(panel: pd.DataFrame) -> Dict[str, Any]:
    return {"beta": -0.08, "se_robust": 0.04}


# ===== Security & Integrity (stubs) =====
def secure_state_l1_stub(panel: pd.DataFrame) -> Dict[str, Any]:
    return {"attacks_detected": 0}


def spectral_shift_stub(panel: pd.DataFrame) -> Dict[str, Any]:
    return {"eig_drift": 0.05}


# ===== Investment under Uncertainty (stubs) =====
def real_options_stub() -> Dict[str, Any]:
    return {"LSM_value": 1.2}


def supplier_portfolio_stub() -> Dict[str, Any]:
    return {"selected_suppliers": ["S1", "S3"], "service_prob": 0.97}


# ===== Further Advanced (realistic approximations or stubs) =====
def transport_modes_sim_stub() -> Dict[str, Any]:
    return {"modes": ["SEA", "AIR", "RAIL"], "schedules": {"SEA": 7, "AIR": 2, "RAIL": 5}}


def online_em_ms(panel: pd.DataFrame) -> Dict[str, Any]:
    # Lightweight online update: adapt mean/var of L based on EWMA
    if "L" not in panel.columns:
        return {"mu": 2.0, "sigma": 1.0}
    L = panel["L"].astype(float).values
    lam = 0.2
    mu = L[0]
    var = 1.0
    for x in L[1:]:
        mu = lam * x + (1 - lam) * mu
        var = lam * (x - mu) ** 2 + (1 - lam) * var
    return {"mu": float(mu), "sigma": float(max(0.2, var ** 0.5))}


def hierarchical_bsts_stub(panel: pd.DataFrame) -> Dict[str, Any]:
    # Placeholder summary for hierarchical pooling
    groups = {"facilities": int(panel.shape[1] > 0)}
    return {"pooled": True, "groups": groups}


def multi_stage_mpc_stub(panel: pd.DataFrame) -> Dict[str, Any]:
    return {"horizon": 3, "reroute_share": 0.1, "expedite_share": 0.05}


def iot_fusion_stub() -> Dict[str, Any]:
    return {"adjust_phi": -0.05, "adjust_L": +0.3}


def nested_logit_stub() -> Dict[str, Any]:
    try:
        from statsmodels.discrete.discrete_model import MNLogit  # noqa: F401
        return {"available": True, "note": "Use MNLogit with inventory and price features"}
    except Exception:
        return {"available": False}


def kcut_hardening_stub() -> Dict[str, Any]:
    return {"harden": ["lane_CN-EU"], "budget": 1}


def linucb_policy_stub(panel: pd.DataFrame) -> Dict[str, Any]:
    # Simple LinUCB placeholders for reroute/expedite arms
    alpha = 0.5
    arms = ["keep", "reroute", "expedite"]
    scores = {a: 0.5 + alpha * 0.1 for a in arms}
    choice = max(scores, key=scores.get)
    return {"choice": choice, "scores": scores}


def l1_observer_stub(panel: pd.DataFrame) -> Dict[str, Any]:
    try:
        import cvxpy as cp  # noqa: F401
        return {"feasible": True}
    except Exception:
        return {"feasible": False}


def fan_charts_stub(panel: pd.DataFrame) -> Dict[str, Any]:
    if "D" not in panel.columns:
        return {"quantiles": {}}
    y = panel["D"].astype(float).values
    q = {
        "p10": float(np.percentile(y, 10)),
        "p50": float(np.percentile(y, 50)),
        "p90": float(np.percentile(y, 90)),
    }
    return {"quantiles": q}


async def main() -> None:
    skus = [SKUKey("SKU1", "F1"), SKUKey("SKU2", "F1"), SKUKey("SKU3", "F2")]
    agent = SupplyShockCRCAgent(skus=skus, T=40)

    # Baseline simulate
    df = agent.simulate()
    kpis = agent.summarize(df)
    strengths = agent.causal_edges(df)
    print("Baseline KPIs:", kpis)
    print("Causal strengths:", strengths)

    # Example interventions
    scenario_port = {"disruption_route": {"F1": 1}}
    scenario_outage = {"outage_facility": {"F2": 1}}
    cmp_port = agent.intervene_and_compare(scenario_port)
    cmp_outage = agent.intervene_and_compare(scenario_outage)
    print("Port disruption delta:", cmp_port["delta"])
    print("Facility outage delta:", cmp_outage["delta"])

    # Full control cycle with AI narrative and upgraded CR-CA analysis
    result = agent.control_cycle()
    print("-" * 80)
    print("AI Narrative Analysis:\n", result.get("ai_analysis"))
    print("-" * 80)
    print("Causal Summary:\n", result.get("causal_summary"))
    
    # Display upgraded analysis results
    advanced = result.get("advanced_causal", {})
    if advanced and not advanced.get("error"):
        print("\n" + "=" * 80)
        print("UPGRADED CR-CA ANALYSIS RESULTS")
        print("=" * 80)
        
        if advanced.get("sensitivity_analysis"):
            sens = advanced["sensitivity_analysis"]
            print("\n--- Sensitivity Analysis ---")
            if sens.get("service_drivers"):
                svc = sens["service_drivers"]
                print(f"Service (backlog) drivers:")
                print(f"  Most influential: {svc.get('most_influential_variable', 'N/A')} (sensitivity: {svc.get('most_influential_sensitivity', 0.0):.4f})")
            if sens.get("cost_drivers"):
                cost = sens["cost_drivers"]
                print(f"Cost (price) drivers:")
                print(f"  Most influential: {cost.get('most_influential_variable', 'N/A')} (sensitivity: {cost.get('most_influential_sensitivity', 0.0):.4f})")
        
        if advanced.get("granger_causality"):
            gc = advanced["granger_causality"]
            print("\n--- Granger Causality Tests ---")
            print(f"D → O: {gc.get('demand_granger_causes_orders', False)} (F={gc.get('d_o_f_stat', 0.0):.2f})")
            print(f"B → p: {gc.get('backlog_granger_causes_price', False)} (F={gc.get('b_p_f_stat', 0.0):.2f})")
        
        if advanced.get("information_theory") and advanced["information_theory"].get("entropies"):
            it = advanced["information_theory"]
            print("\n--- Information Theory ---")
            print("Variable entropies:")
            for var, entropy in list(it["entropies"].items())[:5]:
                print(f"  H({var}) = {entropy:.3f} bits")
            if it.get("mutual_information"):
                print("Top mutual information:")
                mi_items = sorted(it["mutual_information"].items(), key=lambda x: x[1], reverse=True)[:3]
                for pair, mi_val in mi_items:
                    print(f"  I({pair}) = {mi_val:.3f} bits")
        
        if advanced.get("shapley_attribution"):
            shap = advanced["shapley_attribution"]
            if shap.get("backlog_drivers") and shap["backlog_drivers"].get("shapley_values"):
                print("\n--- Shapley Value Attribution (Backlog Drivers) ---")
                shap_b = shap["backlog_drivers"]["shapley_values"]
                for var, value in sorted(shap_b.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                    print(f"  {var}: {value:+.4f}")
        
        if advanced.get("root_causes") and advanced["root_causes"].get("service_issues"):
            rc = advanced["root_causes"]["service_issues"]
            if rc.get("ultimate_root_causes"):
                print("\n--- Deep Root Cause Analysis (Service Issues) ---")
                print("Ultimate root causes of backlog:")
                for root in rc["ultimate_root_causes"][:5]:
                    print(f"  - {root.get('root_cause', 'N/A')} "
                          f"(path strength: {root.get('path_strength', 0.0):.3f}, depth: {root.get('depth', 0)})")
        
        if advanced.get("optimal_intervention") and advanced["optimal_intervention"].get("optimal_sequence"):
            opt = advanced["optimal_intervention"]
            print("\n--- Optimal Intervention Sequence (Bellman) ---")
            print(f"Optimal {len(opt['optimal_sequence'])}-step sequence to minimize backlog:")
            for i, step in enumerate(opt["optimal_sequence"][:3], 1):
                print(f"  Step {i}: {step}")
            print(f"Expected total value: {opt.get('total_value', 0.0):.2f}")
        
        if advanced.get("alternate_realities") and advanced["alternate_realities"].get("best_reality"):
            ar = advanced["alternate_realities"]
            best = ar["best_reality"]
            print("\n--- Alternate Realities Exploration ---")
            print(f"Best alternate reality (minimize backlog):")
            print(f"  Interventions: {best.get('interventions', {})}")
            print(f"  Expected backlog: {best.get('target_value', 0.0):.2f}")
            print(f"  Improvement: {ar.get('improvement_potential', 0.0):+.2f}")
        
        if advanced.get("chain_reaction"):
            cr = advanced["chain_reaction"]
            print("\n--- Cascading Chain Reaction Analysis ---")
            summary = cr.get("summary", {})
            print(f"Total paths found: {summary.get('total_paths_found', 0)}")
            print(f"Feedback loops: {summary.get('feedback_loops_detected', 0)}")
            if cr.get("final_predictions"):
                print(f"Final predictions: {dict(list(cr['final_predictions'].items())[:5])}")
        
        if advanced.get("cross_validation"):
            cv = advanced["cross_validation"]
            print("\n--- Cross-Validation for Edge Reliability ---")
            for edge, cv_data in list(cv.items())[:5]:
                print(f"  {edge}: CV error={cv_data.get('mean_cv_error', 0.0):.4f} ± {cv_data.get('standard_error', 0.0):.4f}")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    import asyncio

    logger.remove()
    logger.add("crca_supply_shock.log", rotation="50 MB", retention="7 days", level="INFO")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")


