"""DefaultWatch dashboard (Streamlit).

Run with `defaultwatch dashboard` (reads the outputs of `defaultwatch train`) or
`defaultwatch dashboard --demo` (builds synthetic outputs under demo/ first).
The config file comes from the DEFAULTWATCH_CONFIG environment variable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from defaultwatch.config import load_config
from defaultwatch.features import FEATURE_LABELS

# --------------------------------------------------------------------------- palette
# Reference data-viz palette: light and dark steps of the same hues.
PALETTE = {
    "light": {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100", "s5": "#e87ba4",
              "surface": "#ffffff", "grid": "#e1e0d9", "axis": "#c3c2b7", "muted": "#898781", "ink2": "#52514e",
              "risk_up": "#e34948", "risk_down": "#2a78d6",
              "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]},
    "dark": {"s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "s4": "#c98500", "s5": "#d55181",
             "surface": "#0e1117", "grid": "#2c2c2a", "axis": "#383835", "muted": "#898781", "ink2": "#c3c2b7",
             "risk_up": "#e66767", "risk_down": "#3987e5",
             "seq": ["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#6da7ec", "#b7d3f6"]},
}
STATUS = {"stable": ("✅", "Stable"), "watch": ("⚠️", "Watch"), "alert": ("🔴", "Alert")}
MODEL_LABEL = {"scorecard": "Logistic scorecard", "gbm": "Gradient boosting"}
MODEL_SLOT = {"scorecard": "s1", "gbm": "s2"}   # colour follows the model everywhere


def theme() -> dict:
    try:
        kind = st.context.theme.type or "light"
    except Exception:
        kind = "light"
    return PALETTE["dark" if kind == "dark" else "light"]


def style(fig: go.Figure, height: int = 380, xtitle: str | None = None, ytitle: str | None = None,
          hovermode: str = "closest", xgrid: bool = True) -> go.Figure:
    c = theme()
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        hovermode=hovermode,
        bargap=0.25,
    )
    fig.update_xaxes(title_text=xtitle, gridcolor=c["grid"], linecolor=c["axis"], zeroline=False, showline=True)
    fig.update_yaxes(title_text=ytitle, gridcolor=c["grid"], linecolor=c["axis"], zeroline=False, showline=True)
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
    if not xgrid:
        fig.update_xaxes(showgrid=False)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def pct(x, digits: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{digits}f}%"


def num(x, digits: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def status_badge(status: str | None) -> str:
    icon, label = STATUS.get(status or "", ("", "n/a"))
    return f"{icon} {label}".strip()


def flabel(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


# --------------------------------------------------------------------------- data
@st.cache_resource
def get_config(path: str | None) -> dict:
    return load_config(path)


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def read_json(path: str, mtime: float) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def read_csv(path: str, mtime: float) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def get_bundle(path: str, mtime: float):
    from defaultwatch.bundle import ModelBundle

    return ModelBundle.load(path) if Path(path).exists() else None


class Reports:
    def __init__(self, cfg: dict):
        self.dir = Path(cfg["paths"]["reports_dir"])

    def json(self, name: str):
        p = self.dir / name
        return read_json(str(p), _mtime(p))

    def csv(self, name: str):
        p = self.dir / name
        return read_csv(str(p), _mtime(p))


# --------------------------------------------------------------------------- tabs
def overview(m: dict) -> None:
    h, d = m["headline"], m["data"]
    champ = m["champion"]
    psi = (m.get("stress") or {}).get("psi_vs_training", {})
    target = m["target_bad_rate"]

    cols = st.columns(4)
    cols[0].metric(f"Out-of-time AUC · {d['oot_vintage']}", num(h["oot_auc"]),
                   delta=f"{h['oot_auc_gbm'] - h['oot_auc_scorecard']:+.3f} GBM vs scorecard", delta_color="off",
                   help="Area under the ROC curve on the out-of-time vintage, champion model.")
    cols[1].metric("Calibration error by decile", pct(h["calibration_error_by_decile"]),
                   help="Ten equal groups by predicted PD; size-weighted mean of |observed − predicted|.")
    cols[2].metric(f"PSI · {d.get('stress_vintage') or 'stress'} vs. training", num(h.get("psi_stress_vs_training")),
                   delta=status_badge(psi.get("score_status")), delta_color="off",
                   help="Population stability of the score. <0.10 stable, 0.10–0.25 watch, ≥0.25 alert.")
    cols[3].metric(f"Approval rate at {pct(target)} default rate", pct(h["approval_rate_at_target"], 1),
                   help="Most loans approved (lowest PD first) while the default rate among approved stays at target.")

    st.caption(
        f"Champion: **{MODEL_LABEL[champ]}** · development {', '.join(map(str, d['train_vintages']))} "
        f"({d['n_dev']:,} loans, {pct(d['dev_default_rate'])} default) · out-of-time {d['oot_vintage']} "
        f"({d['n_oot']:,} loans, {pct(d['oot_default_rate'])}) · stress {d.get('stress_vintage') or 'n/a'} "
        f"({d.get('n_stress', 0):,} loans) · last OOT window closes {d.get('oot_last_window_end')}, "
        f"{d.get('oot_windows_reaching_covid', 0)} reach COVID relief")

    st.subheader("Scorecard vs. gradient boosting")
    rows = []
    for label, src, key, fmt in [
        ("AUC, in-time holdout", "holdout", "auc", num),
        ("AUC, out-of-time", "oot", "auc", num),
        ("KS, out-of-time", "oot", "ks", num),
        ("Calibration error by decile", "oot", "ece_decile", pct),
        ("Largest decile gap", "oot", "max_decile_error", pct),
        ("Observed ÷ predicted", "oot", "calibration_in_the_large", lambda v: num(v, 2)),
        ("Brier score", "oot", "brier", lambda v: num(v, 5)),
    ]:
        rows.append({"Metric": label, **{MODEL_LABEL[k]: fmt(m["models"][k][src].get(key)) for k in MODEL_LABEL}})
    rows.append({"Metric": f"Approval rate at {pct(target)} default",
                 **{MODEL_LABEL[k]: pct(m["models"][k]["approval_oot"]["approval_rate"], 1) for k in MODEL_LABEL}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Champion rule: gradient boosting must beat the scorecard's out-of-time AUC by ≥ "
               f"{m.get('champion_rule_gain', 0.005)}. Observed gain: {m['auc_gain_gbm_vs_scorecard']:+.4f}. "
               f"Decision cutoff PD {pct(m['decision_cutoff']['pd_cutoff'])}, set on the in-time holdout.")


def validation(m: dict, r: Reports) -> None:
    c = theme()
    oot = m["data"]["oot_vintage"]
    left, right = st.columns(2)

    roc = r.csv("roc_oot.csv")
    with left:
        st.markdown(f"**ROC curve · {oot}**")
        if roc is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=c["axis"], width=1),
                                     hoverinfo="skip", showlegend=False))
            for name in MODEL_LABEL:
                d = roc[roc["model"] == name]
                auc = m["models"][name]["oot"]["auc"]
                fig.add_trace(go.Scatter(
                    x=d["fpr"], y=d["tpr"], mode="lines", name=f"{MODEL_LABEL[name]} · AUC {auc:.3f}",
                    line=dict(color=c[MODEL_SLOT[name]], width=2),
                    hovertemplate="False positive %{x:.1%}<br>Defaults caught %{y:.1%}<extra>" + MODEL_LABEL[name] + "</extra>"))
            show(style(fig, xtitle="False positive rate", ytitle="True positive rate"))

    with right:
        st.markdown(f"**Calibration by decile · {oot}**")
        tables = {n: r.csv(f"calibration_oot_{n}.csv") for n in MODEL_LABEL}
        if all(t is not None for t in tables.values()):
            allv = pd.concat(tables.values())
            pos = pd.concat([allv["mean_pd"], allv["observed_rate"][allv["observed_rate"] > 0]])
            lo, hi = pos.min() * 0.7, pos.max() * 1.4
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=c["axis"], width=1),
                                     hoverinfo="skip", name="Perfect calibration", showlegend=False))
            for name, t in tables.items():
                fig.add_trace(go.Scatter(
                    x=t["mean_pd"], y=t["observed_rate"].where(t["observed_rate"] > 0), mode="lines+markers",
                    name=MODEL_LABEL[name], line=dict(color=c[MODEL_SLOT[name]], width=2),
                    marker=dict(size=9, line=dict(width=2, color=c["surface"])),
                    customdata=np.c_[t["decile"], t["n"], t["defaults"]],
                    hovertemplate=("Decile %{customdata[0]}<br>Predicted %{x:.2%}<br>Observed %{y:.2%}"
                                   "<br>%{customdata[1]:,} loans, %{customdata[2]:,} defaults<extra>"
                                   + MODEL_LABEL[name] + "</extra>")))
            fig.update_xaxes(type="log", tickformat=".1%", range=[np.log10(lo), np.log10(hi)])
            fig.update_yaxes(type="log", tickformat=".1%", range=[np.log10(lo), np.log10(hi)])
            show(style(fig, xtitle="Mean predicted default rate", ytitle="Observed default rate"))
            st.caption("Points on the grey diagonal are perfectly calibrated. Log scales.")

    st.markdown(f"**Approval rate vs. default rate among approved · {oot}**")
    curve = r.csv("approval_curve_oot.csv")
    if curve is not None:
        target = m["target_bad_rate"]
        fig = go.Figure()
        for name in MODEL_LABEL:
            d = curve[curve["model"] == name]
            fig.add_trace(go.Scatter(
                x=d["approval_rate"], y=d["bad_rate_approved"], mode="lines", name=MODEL_LABEL[name],
                line=dict(color=c[MODEL_SLOT[name]], width=2), customdata=d["cutoff_pd"],
                hovertemplate="Approve %{x:.1%}<br>Default rate %{y:.2%}<br>Cutoff PD %{customdata:.2%}<extra>"
                              + MODEL_LABEL[name] + "</extra>"))
            op = m["models"][name]["approval_oot"]
            fig.add_trace(go.Scatter(x=[op["approval_rate"]], y=[op["bad_rate_approved"]], mode="markers",
                                     marker=dict(size=11, color=c[MODEL_SLOT[name]], line=dict(width=2, color=c["surface"])),
                                     showlegend=False, hoverinfo="skip"))
        fig.add_hline(y=target, line=dict(color=c["muted"], width=1),
                      annotation_text=f"target {pct(target)}", annotation_position="top left")
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        fig.update_yaxes(tickformat=".1%", rangemode="tozero")
        show(style(fig, height=360, xtitle="Approval rate", ytitle="Default rate among approved"))
        st.caption("Dots mark each model's approval rate at the target default rate (the headline measure).")

    with st.expander("Table view: calibration by decile"):
        for name in MODEL_LABEL:
            t = r.csv(f"calibration_oot_{name}.csv")
            if t is not None:
                st.markdown(f"*{MODEL_LABEL[name]}*")
                st.dataframe(t[["decile", "n", "defaults", "mean_pd", "observed_rate", "abs_error"]],
                             hide_index=True, width="stretch",
                             column_config={"mean_pd": st.column_config.NumberColumn("Mean PD", format="percent"),
                                            "observed_rate": st.column_config.NumberColumn("Observed", format="percent"),
                                            "abs_error": st.column_config.NumberColumn("Abs. gap", format="percent")})


def stress(m: dict, r: Reports) -> None:
    c = theme()
    st_res = m.get("stress") or {}
    champ = m["champion"]
    s = st_res.get(champ)
    if not s:
        st.info("The stress vintage was not available in this run.")
        return
    vint = m["data"]["stress_vintage"]
    psi = st_res.get("psi_vs_training", {})
    cols = st.columns(4)
    cols[0].metric(f"Observed 90+ rate · {vint}", pct(s["observed_rate"]))
    cols[1].metric("Excluding relief (real default)", pct(s["observed_real_rate"]))
    cols[2].metric("Share of 90+ explained by relief", pct(s["relief_share_of_defaults"], 1))
    cols[3].metric("Score PSI vs. training", num(psi.get("score_psi")), delta=status_badge(psi.get("score_status")),
                   delta_color="off")

    breakdown = r.csv("outcome_breakdown.csv")
    if breakdown is not None:
        measures = [("mean_pd", "Mean predicted PD", "s3"), ("observed_rate", "Observed 90+ (all causes)", "s4"),
                    ("observed_real_rate", "Observed 90+ excluding relief", "s5")]
        fig = go.Figure()
        for col, label, slot in measures:
            fig.add_trace(go.Bar(x=breakdown["label"], y=breakdown[col], name=label, marker_color=c[slot],
                                 marker_line=dict(width=0), text=[pct(v) for v in breakdown[col]],
                                 textposition="outside", hovertemplate="%{x}<br>" + label + ": %{y:.2%}<extra></extra>"))
        fig.update_yaxes(tickformat=".1%", rangemode="tozero")
        fig.update_layout(barmode="group")
        show(style(fig, height=380, ytitle="Rate within 24 months", xgrid=False))
        st.caption("Relief = a payment-deferral, disaster-delinquency or forbearance/repayment/trial flag in the 90+ "
                   "month or the three months before it. The model is meant to match the real-default bar.")

    rows = []
    for name in MODEL_LABEL:
        x = st_res.get(name, {})
        rows.append({"Model": MODEL_LABEL[name], "AUC vs. all 90+": num(x.get("auc_all")),
                     "AUC vs. real default": num(x.get("auc_real")),
                     "Observed ÷ predicted (all)": num(x.get("calibration_in_the_large_all"), 2),
                     "Observed ÷ predicted (real)": num(x.get("calibration_in_the_large_real"), 2),
                     "Calibration error by decile (real)": pct(x.get("ece_decile_real"))})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if breakdown is not None:
        with st.expander("Table view: outcome breakdown"):
            st.dataframe(breakdown, hide_index=True, width="stretch")


def monitoring(m: dict, r: Reports, bundle) -> None:
    c = theme()
    summary = r.csv("monitoring/psi_summary.csv")
    if summary is None or summary.empty:
        st.info("No post-training vintages have been monitored yet. Add later vintages to data/raw and run "
                "`defaultwatch prepare` then `defaultwatch monitor`.")
        return
    warn = bundle.baseline.warn if bundle else 0.10
    alert = bundle.baseline.alert if bundle else 0.25
    top = m.get("top_features", [])

    alerts = summary[summary["status"] == "alert"]
    if len(alerts):
        st.error(f"🔴 Alert on vintage(s) {', '.join(map(str, alerts['vintage']))}: "
                 + "; ".join(f"{v}: {a}" for v, a in zip(alerts["vintage"], alerts["alerts"].fillna(""))))
    else:
        st.success("✅ No PSI alerts on the score or the watched features.")

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Score PSI by vintage**")
        score_status = ["alert" if v >= alert else "watch" if v >= warn else "stable" for v in summary["score_psi"]]
        labels = [f"{v:.3f} {STATUS[s][0]}" for v, s in zip(summary["score_psi"], score_status)]
        fig = go.Figure(go.Bar(
            x=summary["vintage"].astype(str), y=summary["score_psi"], marker_color=c["s1"], text=labels,
            textposition="outside",
            customdata=np.c_[summary["n"], [STATUS[s][1] for s in score_status], summary["status"]],
            hovertemplate="Vintage %{x}<br>Score PSI %{y:.3f} (%{customdata[1]})<br>%{customdata[0]:,} loans<br>"
                          "Vintage status incl. features: %{customdata[2]}<extra></extra>"))
        for level, name in ((warn, "watch"), (alert, "alert")):
            fig.add_hline(y=level, line=dict(color=c["muted"], width=1), annotation_text=f"{name} {level:.2f}",
                          annotation_position="top left")
        fig.update_yaxes(range=[0, max(alert * 1.3, float(summary["score_psi"].max()) * 1.25)])
        fig.update_xaxes(type="category")
        show(style(fig, height=380, xtitle="Vintage", ytitle="PSI vs. training", xgrid=False))
        st.caption("Icons show the score's own PSI status; feature alerts are in the heatmap and table.")

    with right:
        st.markdown("**Feature PSI by vintage** (★ = watched for alerts)")
        feats = [col[4:] for col in summary.columns if col.startswith("psi_")]
        z = summary[[f"psi_{f}" for f in feats]].to_numpy().T
        ylabels = [f"{'★ ' if f in top else ''}{flabel(f)}" for f in feats]
        zmax = max(alert, float(np.nanmax(z)))
        fig = go.Figure(go.Heatmap(
            z=z, x=summary["vintage"].astype(str), y=ylabels, zmin=0, zmax=zmax,
            colorscale=[[i / (len(c["seq"]) - 1), col] for i, col in enumerate(c["seq"])],
            text=[[f"{v:.3f}" for v in row] for row in z], texttemplate="%{text}", xgap=2, ygap=2,
            colorbar=dict(title=dict(text="PSI"), thickness=10),
            hovertemplate="%{y}<br>Vintage %{x}<br>PSI %{z:.3f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(type="category", showgrid=False)
        show(style(fig, height=380))

    table = summary[["vintage", "n", "mean_pd", "score_psi", "max_watched_psi", "status", "alerts", "warnings"]].copy()
    table["status"] = table["status"].map(status_badge)
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"mean_pd": st.column_config.NumberColumn("Mean PD", format="percent"),
                                "score_psi": st.column_config.NumberColumn("Score PSI", format="%.3f"),
                                "max_watched_psi": st.column_config.NumberColumn("Worst watched PSI", format="%.3f"),
                                "n": st.column_config.NumberColumn("Loans", format="%d")})
    outcome_cols = [col for col in summary.columns if col.startswith("outcome_")]
    if outcome_cols:
        st.markdown("**Outcomes where the 24-month windows have closed**")
        pct_cols = {"outcome_observed_rate": "Observed 90+", "outcome_observed_real_rate": "Observed excl. relief",
                    "outcome_mean_pd_complete": "Mean PD"}
        ratio_cols = {"outcome_observed_to_predicted": "Observed ÷ predicted",
                      "outcome_observed_real_to_predicted": "Excl. relief ÷ predicted",
                      "outcome_auc": "AUC", "outcome_auc_real": "AUC excl. relief"}
        config = {"outcome_n_complete": st.column_config.NumberColumn("Loans", format="%d")}
        config.update({c: st.column_config.NumberColumn(label, format="percent") for c, label in pct_cols.items()})
        config.update({c: st.column_config.NumberColumn(label, format="%.2f") for c, label in ratio_cols.items()})
        st.dataframe(summary[["vintage"] + outcome_cols].dropna(subset=["outcome_auc"]), hide_index=True,
                     width="stretch", column_config=config)
        if "complete_share" in summary.columns:
            pending = summary[summary["outcome_auc"].isna()]
            if len(pending):
                st.caption("Outcomes pending (windows still open): " + ", ".join(
                    f"{v} ({s:.0%} closed)" for v, s in zip(pending["vintage"], pending["complete_share"])) + ".")
    st.caption(f"Thresholds: < {warn} stable, {warn}–{alert} watch, ≥ {alert} alert. "
               "`defaultwatch monitor --fail-on-alert` exits with code 2 on an alert for schedulers.")


def reasons_tab(m: dict, r: Reports) -> None:
    c = theme()
    rs = m.get("reasons", {})
    cols = st.columns(4)
    cols[0].metric("Out-of-time declines", f"{rs.get('declined', 0):,}", delta=f"{pct(rs.get('decline_rate'), 1)} of loans",
                   delta_color="off")
    cols[1].metric("Declines with ≥1 reason", pct(rs.get("with_at_least_one_reason"), 1))
    cols[2].metric("Reasons per decline", num(rs.get("mean_reasons"), 1))
    cols[3].metric("Faithfulness of #1 reason", pct(rs.get("faithfulness_top_reason"), 1),
                   help="Share of declines whose PD falls when the #1 reason's factor is set to the typical approved value.")

    freq = r.csv("reason_frequency_oot.csv")
    if freq is not None:
        freq = freq.sort_values("share_of_declines_citing")
        names = [flabel(f) for f in freq["feature"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=names, x=freq["share_of_declines_citing"], orientation="h", name="Cited at all",
                             marker_color=c["s3"], text=[pct(v, 0) for v in freq["share_of_declines_citing"]],
                             textposition="outside", hovertemplate="%{y}: cited in %{x:.1%} of declines<extra></extra>"))
        fig.add_trace(go.Bar(y=names, x=freq["share_as_top_reason"], orientation="h", name="#1 reason",
                             marker_color=c["s4"], text=[pct(v, 0) for v in freq["share_as_top_reason"]],
                             textposition="outside", hovertemplate="%{y}: #1 reason in %{x:.1%} of declines<extra></extra>"))
        fig.update_layout(barmode="group")
        fig.update_xaxes(tickformat=".0%", range=[0, 1.15])
        show(style(fig, height=420, xtitle="Share of out-of-time declines"))

    points = r.csv("scorecard_points.csv")
    if points is not None:
        st.subheader("Scorecard points")
        in_model = points[points["in_model"]]
        feature = st.selectbox("Characteristic", in_model["feature"].unique(), format_func=flabel)
        t = points[points["feature"] == feature]
        fig = go.Figure(go.Bar(
            x=t["bin"].astype(str), y=t["points"], marker_color=c["s1"], text=[f"{p:.0f}" for p in t["points"]],
            textposition="outside", customdata=np.c_[t["count"], t["bad_rate"]],
            hovertemplate="%{x}<br>%{y:.1f} points<br>%{customdata[0]:,} loans, bad rate %{customdata[1]:.2%}<extra></extra>"))
        fig.update_xaxes(type="category")
        show(style(fig, height=320, xtitle=flabel(feature), ytitle="Points (higher = lower risk)", xgrid=False))
        st.dataframe(t[["bin", "count", "bads", "bad_rate", "woe", "points"]], hide_index=True, width="stretch",
                     column_config={"bad_rate": st.column_config.NumberColumn("Bad rate", format="percent"),
                                    "woe": st.column_config.NumberColumn("WOE", format="%.3f"),
                                    "points": st.column_config.NumberColumn("Points", format="%.1f")})


PURPOSES = {"P": "Purchase", "C": "Cash-out refinance", "N": "No-cash-out refinance"}
OCCUPANCIES = {"P": "Primary residence", "S": "Second home", "I": "Investment property"}
PROPERTIES = {"SF": "Single-family", "PU": "Planned unit development", "CO": "Condominium",
              "MH": "Manufactured home", "CP": "Co-op"}


def score_tab(bundle) -> None:
    c = theme()
    if bundle is None:
        st.info("No fitted model here. Models trained on Freddie Mac data stay out of the public repo under the "
                "dataset's terms, so live scoring runs locally: clone the repo, then `defaultwatch train` and "
                "`defaultwatch dashboard` (or `defaultwatch dashboard --demo` for synthetic data).")
        return
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("**Application**")
        no_score = st.checkbox("Credit score not available")
        cs = None if no_score else st.slider("Credit score", 300, 850, 690)
        no_dti = st.checkbox("DTI not available")
        dti = None if no_dti else st.slider("Debt-to-income ratio (%)", 0, 65, 38)
        ltv = st.slider("Loan-to-value ratio (%)", 5, 105, 85)
        cltv = st.slider("Combined LTV incl. other liens (%)", ltv, 125, ltv)
        term = st.selectbox("Loan term", [360, 240, 180, 120], format_func=lambda v: f"{v // 12} years")
        units = st.selectbox("Units", [1, 2, 3, 4])
        purpose = st.selectbox("Loan purpose", list(PURPOSES), format_func=PURPOSES.get)
        occ = st.selectbox("Occupancy", list(OCCUPANCIES), format_func=OCCUPANCIES.get)
        ptype = st.selectbox("Property type", list(PROPERTIES), format_func=PROPERTIES.get)
    app = {"credit_score": cs, "orig_dti": dti, "orig_ltv": ltv, "orig_cltv": cltv, "orig_loan_term": term,
           "num_units": units, "loan_purpose": purpose, "occupancy_status": occ, "property_type": ptype}
    raw = pd.DataFrame([app])
    result = bundle.score_frame(raw).iloc[0]

    with right:
        decision = result["decision"]
        k = st.columns(3)
        k[0].metric("Probability of default", pct(result["pd"]),
                    help="90+ days delinquent within 24 months of first payment.")
        k[1].metric("Score", f"{int(result['score'])}", help="Higher = lower risk.")
        k[2].metric("Decision", "✅ Approve" if decision == "approve" else "🔴 Decline",
                    delta=f"cutoff PD {pct(bundle.pd_cutoff)}", delta_color="off")

        st.markdown("**Principal reasons**" + (" (adverse-action notice)" if decision == "decline" else
                                                " (largest risk drivers)"))
        if result["reasons"]:
            for rsn in result["reasons"]:
                st.markdown(f"{rsn['rank']}. {rsn['reason']}  \n"
                            f"<span style='opacity:.7;font-size:.85em'>{rsn['code']} · {rsn['reg_b_category']}</span>",
                            unsafe_allow_html=True)
        else:
            st.write("No factor raised this applicant's risk above the reference.")

        X = bundle.features(raw)
        contrib = bundle.model.contributions(X).iloc[0].sort_values()
        units_label = ("Contribution to log-odds of default" if bundle.model_type == "gbm"
                       else "Points below the characteristic's maximum")
        fig = go.Figure(go.Bar(
            y=[flabel(f) for f in contrib.index], x=contrib.values, orientation="h",
            marker_color=[c["risk_up"] if v > 0 else c["risk_down"] for v in contrib.values],
            hovertemplate="%{y}<br>%{x:+.3f}<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=c["axis"], width=1))
        show(style(fig, height=330, xtitle=units_label))
        st.caption("Red bars raise the risk estimate; blue bars lower it.")

    st.markdown("**What-if: how the PD moves with one factor, holding the rest**")
    grids = {"credit_score": np.arange(580, 851, 5), "orig_dti": np.arange(5, 66, 1),
             "orig_ltv": np.arange(20, 106, 1), "orig_cltv": np.arange(20, 126, 1)}
    feature = st.selectbox("Factor", list(grids), format_func=flabel)
    grid = grids[feature]
    sweep = pd.DataFrame([app] * len(grid))
    sweep[feature] = grid
    if feature == "orig_ltv":
        sweep["orig_cltv"] = np.maximum(sweep["orig_cltv"].astype(float), grid)
    curve = bundle.predict_pd(sweep)
    color = c[MODEL_SLOT[bundle.model_type]]
    fig = go.Figure(go.Scatter(x=grid, y=curve, mode="lines", line=dict(color=color, width=2),
                               name=MODEL_LABEL[bundle.model_type], showlegend=False,
                               hovertemplate=flabel(feature) + " %{x}<br>PD %{y:.2%}<extra></extra>"))
    current = app[feature]
    if current is not None:
        fig.add_trace(go.Scatter(x=[current], y=[result["pd"]], mode="markers", showlegend=False,
                                 marker=dict(size=11, color=color, line=dict(width=2, color=c["surface"])),
                                 hovertemplate="This application<br>PD %{y:.2%}<extra></extra>"))
    fig.add_hline(y=bundle.pd_cutoff, line=dict(color=c["muted"], width=1),
                  annotation_text=f"decline above {pct(bundle.pd_cutoff)}", annotation_position="top left")
    fig.update_yaxes(tickformat=".1%", rangemode="tozero")
    show(style(fig, height=320, xtitle=flabel(feature), ytitle="Probability of default", hovermode="x unified"))


# --------------------------------------------------------------------------- app
def main() -> None:
    st.set_page_config(page_title="DefaultWatch", page_icon="🏠", layout="wide")
    cfg_path = os.getenv("DEFAULTWATCH_CONFIG")
    cfg = get_config(cfg_path)
    reports = Reports(cfg)
    metrics = reports.json("metrics.json")
    bpath = Path(cfg["paths"]["model_dir"]) / "defaultwatch_bundle.joblib"
    bundle = get_bundle(str(bpath), _mtime(bpath))

    st.title("DefaultWatch")
    st.caption("Mortgage default model · 90+ days delinquent within 24 months · Freddie Mac loan-level data")

    synthetic = bool((metrics or {}).get("synthetic_data") or (bundle and bundle.metadata.get("synthetic")))
    if synthetic:
        st.warning("**Synthetic data.** These outputs come from the built-in generator and only exercise the "
                   "pipeline. They are not results on Freddie Mac data.", icon="⚠️")

    with st.sidebar:
        st.header("Run")
        if metrics:
            st.markdown(f"**Champion:** {MODEL_LABEL[metrics['champion']]}  \n"
                        f"**Trained:** {metrics.get('trained_at', 'n/a')}  \n"
                        f"**Data:** {'synthetic' if synthetic else 'Freddie Mac sample files'}")
            if metrics.get("mlflow"):
                mf = metrics["mlflow"]
                st.markdown(f"**MLflow:** `{mf.get('registered_model')}` v{mf.get('registered_version')} "
                            f"@{mf.get('alias')}")
        st.markdown(f"**Reports:** `{reports.dir}`")
        if st.button("Reload outputs"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    if metrics is None:
        st.info(f"No `metrics.json` in `{reports.dir}`. Run `defaultwatch train` on the Freddie Mac files, or "
                "`defaultwatch dashboard --demo` to explore synthetic outputs.")
        if bundle is not None:
            score_tab(bundle)
        return

    tabs = st.tabs(["Overview", "Validation", "Stress test", "Monitoring", "Reason codes", "Score a loan"])
    with tabs[0]:
        overview(metrics)
    with tabs[1]:
        validation(metrics, reports)
    with tabs[2]:
        stress(metrics, reports)
    with tabs[3]:
        monitoring(metrics, reports, bundle)
    with tabs[4]:
        reasons_tab(metrics, reports)
    with tabs[5]:
        score_tab(bundle)


if __name__ == "__main__":   # `streamlit run dashboard.py`; app/streamlit_app.py imports main() instead
    main()
