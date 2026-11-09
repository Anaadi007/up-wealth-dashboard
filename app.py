"""
app.py – Streamlit dashboard: descriptive and diagnostic analytics for the UP wealth-management start-up.
Run:  pip install -r requirements.txt   then   streamlit run app.py
Reads the cleaned data written by 02_clean_transform.py. All data is synthetic.
"""
import numpy as np, pandas as pd, streamlit as st, plotly.express as px, plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="UP Wealth Management – EDA", page_icon="📊", layout="wide")
TIERS = ["Tier 1", "Tier 2", "Tier 3"]
SEGS = ["Affluent professionals", "Young digital earners", "Conservative govt & retirees", "Traditional business families", "Rural & low-literacy savers"]
STAGES = ["Lead", "Contacted", "Discovery meeting", "Plan presented", "KYC & onboarding", "Client"]
LIK = {"risk_tolerance": "Risk tolerance", "digital_comfort": "Digital comfort", "trust_in_advisors": "Trust in advisors", "importance_face_to_face": "Face-to-face importance", "awareness_wealth_mgmt": "Awareness of wealth mgmt"}
COLORS = px.colors.qualitative.Safe

@st.cache_data
def load():
    s = pd.read_csv("data/clean/survey_clean.csv")
    p = pd.read_csv("data/clean/pipeline_clean.csv", parse_dates=["created_date", "closed_date"])
    log = pd.read_csv("data/clean/cleaning_log.csv")
    raw_s = pd.read_csv("data/raw/survey_raw.csv"); raw_p = pd.read_csv("data/raw/pipeline_raw.csv")
    return s, p, log, raw_s, raw_p
s_all, p_all, LOG, RAW_S, RAW_P = load()
pct = lambda x: f"{x:.1%}"; inr = lambda x: f"₹{x:,.0f}"

# ---------------- sidebar filters ----------------
st.sidebar.header("Filters")
tiers = st.sidebar.multiselect("City tier", TIERS, default=TIERS)
segs = st.sidebar.multiselect("Customer segment", SEGS, default=SEGS)
sources = st.sidebar.multiselect("Lead source (pipeline only)", sorted(p_all.lead_source.unique()), default=sorted(p_all.lead_source.unique()))
st.sidebar.caption("Filters apply to every chart. All data is synthetic, generated to demonstrate the method.")
s = s_all[s_all.city_tier.isin(tiers) & s_all.segment.isin(segs)]
p = p_all[p_all.city_tier.isin(tiers) & p_all.segment.isin(segs) & p_all.lead_source.isin(sources)]
if len(s) < 30 or len(p) < 30:
    st.warning("Too few rows for these filters. Widen the selection."); st.stop()

st.title("Validating a wealth-management start-up in Uttar Pradesh")
st.caption(f"Descriptive and diagnostic analytics · survey: {len(s):,} of {len(s_all):,} respondents · pipeline: {len(p):,} of {len(p_all):,} leads in the current filter")
k = st.columns(6)
k[0].metric("Respondents", f"{len(s):,}"); k[1].metric("Likely to sign up", pct(s.engage_flag.mean()))
k[2].metric("Median stated SIP", inr(s.willing_monthly_investment_inr.median())); k[3].metric("Leads", f"{len(p):,}")
k[4].metric("Lead → client", pct(p.won_flag.mean())); k[5].metric("Median days to close", f"{p.loc[p.won_flag==1,'days_to_close'].median():.0f}")

tab = st.tabs(["Data & cleaning", "Descriptive", "Correlation", "Diagnostic", "Sales pipeline", "Insights"])

# ================= DATA & CLEANING =================
with tab[0]:
    c1, c2 = st.columns(2)
    c1.subheader("Survey"); c1.write(f"Raw **{len(RAW_S):,}** rows → clean **{len(s_all):,}** rows ({len(s_all)/len(RAW_S):.1%} kept), {s_all.shape[1]} columns after feature engineering.")
    c2.subheader("Sales pipeline"); c2.write(f"Raw **{len(RAW_P):,}** rows → clean **{len(p_all):,}** rows ({len(p_all)/len(RAW_P):.1%} kept), {p_all.shape[1]} columns after feature engineering.")
    st.subheader("Cleaning log"); st.dataframe(LOG, width="stretch", hide_index=True)
    mv = pd.DataFrame({"Raw survey": RAW_S.isna().mean(), "Clean survey": s_all.reindex(columns=RAW_S.columns).isna().mean()}).dropna().query("`Raw survey` > 0").sort_values("Raw survey")
    fig = px.bar(mv.reset_index().melt(id_vars="index"), y="index", x="value", color="variable", barmode="group", orientation="h", labels={"index": "", "value": "Share missing", "variable": ""}, title="Missing values before and after cleaning (survey)")
    fig.update_xaxes(tickformat=".0%"); st.plotly_chart(fig, width="stretch")
    st.markdown(f"**Explanation:** before cleaning, up to {mv['Raw survey'].max():.1%} of a column was missing (savings had the most item non-response). After imputation by group median or mode, only the outcome *stated monthly investment* keeps its {s_all.willing_monthly_investment_inr.isna().sum()} gaps on purpose, because imputing an outcome would invent results.")
    st.subheader("Is non-response concentrated in some groups?")
    MC = pd.read_csv("data/clean/missingness_check.csv"); st.dataframe(MC, width="stretch", hide_index=True)
    st.markdown(f"**Explanation:** chi-square tests check whether a column's missing values depend on city tier or occupation. {int((MC.chi2_p<0.05).sum())} of {len(MC)} tests have p < 0.05, and that one is the un-imputed outcome column. Gaps in the imputed columns are spread evenly, so filling them with the group median is defensible.")
    c1, c2 = st.columns(2)
    ob = pd.DataFrame({"value": np.r_[np.log10(RAW_S.annual_income_lakh.dropna().clip(lower=1)), np.log10(s_all.annual_income_lakh)], "data": ["Raw"]*RAW_S.annual_income_lakh.notna().sum() + ["Clean"]*len(s_all)})
    fig = px.box(ob, x="data", y="value", points="outliers", labels={"value": "Income, log10(₹ lakh)", "data": ""}, title="Survey income before and after cleaning"); c1.plotly_chart(fig, width="stretch")
    c1.markdown(f"**Explanation:** raw income reached ₹{RAW_S.annual_income_lakh.max():,.0f} L because of unit errors. After correction the maximum is ₹{s_all.annual_income_lakh.max():,.0f} L, a genuine high earner kept and flagged.")
    oc = pd.DataFrame({"value": np.r_[RAW_P.acquisition_cost_inr.dropna(), p_all.acquisition_cost_inr], "data": ["Raw"]*RAW_P.acquisition_cost_inr.notna().sum() + ["Clean"]*len(p_all)})
    fig = px.box(oc, x="data", y="value", points="outliers", labels={"value": "Acquisition cost per lead, ₹", "data": ""}, title="Pipeline acquisition cost before and after capping"); c2.plotly_chart(fig, width="stretch")
    c2.markdown(f"**Explanation:** raw costs reached ₹{RAW_P.acquisition_cost_inr.max():,.0f} per lead; values above Q3 + 3×IQR were capped at ₹{p_all.acquisition_cost_inr.max():,.0f} and flagged.")
    with st.expander("Preview the cleaned data"):
        st.dataframe(s.head(50), width="stretch"); st.dataframe(p.head(50), width="stretch")

# ================= DESCRIPTIVE =================
with tab[1]:
    c1, c2 = st.columns(2)
    fig = px.histogram(s, x="annual_income_lakh", color="city_tier", log_x=True, nbins=60, category_orders={"city_tier": TIERS}, opacity=.7, barmode="overlay", labels={"annual_income_lakh": "Annual income, ₹ lakh (log scale)"}, title="Income distribution by city tier")
    c1.plotly_chart(fig, width="stretch")
    med = s.groupby("city_tier").annual_income_lakh.median()
    c1.markdown(f"**Explanation:** income is right-skewed (skew {s.annual_income_lakh.skew():.1f}): the mean ₹{s.annual_income_lakh.mean():.1f} L sits well above the median ₹{s.annual_income_lakh.median():.1f} L, so a log scale is used. Median income by tier: " + ", ".join(f"{t} ₹{v:.1f} L" for t, v in med.items()) + ". Lower-tier households have less to invest.")
    d = s.dropna(subset=["willing_monthly_investment_inr"])
    order = d.groupby("segment").willing_monthly_investment_inr.median().sort_values(ascending=False).index.tolist()
    fig = px.box(d, y="segment", x="willing_monthly_investment_inr", log_x=True, points=False, category_orders={"segment": order}, labels={"willing_monthly_investment_inr": "Stated monthly investment, ₹ (log)", "segment": ""}, title="Stated monthly investment by segment")
    c2.plotly_chart(fig, width="stretch")
    m = d.groupby("segment").willing_monthly_investment_inr.median()
    c2.markdown(f"**Explanation:** the median respondent would invest {inr(d.willing_monthly_investment_inr.median())} a month if guided. {m.idxmax()} state the most ({inr(m.max())}) and {m.idxmin()} the least ({inr(m.min())}), a {m.max()/m.min():.0f}× gap. Trail revenue scales with assets, so ticket size largely sets client value.")
    c1, c2 = st.columns(2)
    r = s.groupby(["segment", "city_tier"]).engage_flag.mean().reset_index()
    fig = px.density_heatmap(r, x="city_tier", y="segment", z="engage_flag", text_auto=".0%", category_orders={"city_tier": TIERS, "segment": SEGS}, color_continuous_scale="Blues", labels={"engage_flag": "Likely to sign up", "city_tier": "", "segment": ""}, title="Sign-up intent by segment × tier")
    fig.update_coloraxes(colorbar_tickformat=".0%"); c1.plotly_chart(fig, width="stretch")
    bt = s.groupby("city_tier").engage_flag.mean()
    c1.markdown(f"**Explanation:** each cell is the share likely to sign up within 6 months. Overall intent is {pct(s.engage_flag.mean())}; by tier it is " + ", ".join(f"{t} {pct(v)}" for t, v in bt.items()) + ". Segment differences are larger than tier differences; the Diagnostic tab's multivariate check shows whether tier still matters once income, trust and awareness are controlled for.")
    hold = s.filter(like="holds_").mean().rename(lambda c: c.replace("holds_", "").replace("_", " ")).sort_values()
    want = s.filter(like="int_").mean().rename(lambda c: c.replace("int_", "").replace("_", " ")).sort_values()
    fig = go.Figure([go.Bar(y=hold.index, x=hold.values, orientation="h", name="Hold today", marker_color=COLORS[0]), go.Bar(y=want.index, x=want.values, orientation="h", name="Want help with", marker_color=COLORS[1])])
    fig.update_layout(title="Products held vs services wanted", xaxis_tickformat=".0%", barmode="group", height=480); c2.plotly_chart(fig, width="stretch")
    c2.markdown(f"**Explanation:** traditional assets dominate: FDs {pct(s.holds_FD.mean())}, gold {pct(s.holds_Gold.mean())} and traditional LIC {pct(s.holds_LIC_Traditional.mean())}, while only {pct(s.holds_Mutual_Funds.mean())} hold mutual funds. {pct(s.traditional_only.mean())} hold no market-linked product at all. The most-wanted service is {want.idxmax().lower()} ({pct(want.max())}).")
    dist = pd.DataFrame({LIK[c]: s[c].value_counts(normalize=True).reindex([1, 2, 3, 4, 5]).fillna(0) for c in LIK}).T
    fig = go.Figure([go.Bar(y=dist.index, x=dist[k_], name=str(k_), orientation="h", marker_color=c) for k_, c in zip([1, 2, 3, 4, 5], ["#c0504d", "#e6a19f", "#d9d9d9", "#9dc3e6", "#1f4e79"])])
    fig.update_layout(barmode="stack", title="Attitude ratings (1 = low … 5 = high)", xaxis_tickformat=".0%", legend_title="Rating", height=330); st.plotly_chart(fig, width="stretch")
    st.markdown(f"**Explanation:** {pct((s.trust_in_advisors<=2).mean())} rate trust in advisors 1–2 and {pct((s.awareness_wealth_mgmt<=2).mean())} rate awareness 1–2, yet {pct((s.importance_face_to_face>=4).mean())} rate meeting in person as important. The market is unaware and wary but wants a human relationship, which is the gap a local, face-to-face firm can fill.")

# ================= CORRELATION =================
with tab[2]:
    cc = {"age": "Age", "log_income": "Income (log)", "log_savings": "Savings (log)", "expense_pct_income": "Expense %", "financial_literacy_score": "Literacy", "risk_tolerance": "Risk tol.", "digital_comfort": "Digital", "trust_in_advisors": "Trust", "importance_face_to_face": "Face-to-face", "awareness_wealth_mgmt": "Awareness", "num_products_held": "# products", "num_goals": "# goals", "past_bad_experience": "Past mis-sold", "uses_investing_app": "App user", "log_monthly_investment": "Invest (log)", "engage_flag": "Sign-up"}
    method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True, format_func=str.title)
    C = s[list(cc)].corr(method=method).rename(index=cc, columns=cc)
    fig = px.imshow(C.where(np.tril(np.ones_like(C, dtype=bool))), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title=f"Correlation matrix ({method.title()})")
    fig.update_layout(height=640); st.plotly_chart(fig, width="stretch")
    pr = C.where(np.tril(np.ones_like(C, dtype=bool), -1)).stack().sort_values(key=abs, ascending=False)
    st.markdown("**Explanation:** the strongest relationships in the current filter are " + "; ".join(f"{a} ↔ {b} ({v:+.2f})" for (a, b), v in pr.head(5).items()) + ". A *capacity* group (income, savings, expenses) goes with how much people invest, while an *attitude* group (age, digital comfort, risk, face-to-face) describes how they want to be served.")
    c1, c2 = st.columns(2)
    t = s[list(cc)].corr()["engage_flag"].drop(["engage_flag", "log_monthly_investment"]).rename(cc).sort_values()
    fig = px.bar(x=t.values, y=t.index, orientation="h", color=t.values > 0, color_discrete_map={True: COLORS[0], False: "#c0504d"}, labels={"x": "Point-biserial r with sign-up", "y": ""}, title="What correlates with sign-up intent"); fig.update_layout(showlegend=False)
    c1.plotly_chart(fig, width="stretch")
    c1.markdown(f"**Explanation:** the largest positive correlations are with {t.index[-1].lower()} (r = {t.iloc[-1]:.2f}) and {t.index[-2].lower()} ({t.iloc[-2]:.2f}); the strongest negative is {t.index[0].lower()} ({t.iloc[0]:.2f}). These are associations, not proven causes, but trust and awareness are things the firm can try to influence through marketing and reputation. Effect sizes: below 0.1 negligible, 0.1–0.3 weak, 0.3–0.5 moderate, above 0.5 strong.")
    d = s.dropna(subset=["willing_monthly_investment_inr"]); d = d[d.willing_monthly_investment_inr > 0]
    smp = d.sample(min(1500, len(d)), random_state=1)
    fig = px.scatter(smp, x="monthly_surplus_inr", y="willing_monthly_investment_inr", color="segment", log_x=True, log_y=True, opacity=.5, category_orders={"segment": SEGS}, trendline="ols", trendline_scope="overall", trendline_options=dict(log_x=True, log_y=True), labels={"monthly_surplus_inr": "Monthly surplus, ₹", "willing_monthly_investment_inr": "Stated investment, ₹"}, title="Monthly surplus vs stated investment (log–log)")
    c2.plotly_chart(fig, width="stretch")
    r_ = stats.pearsonr(np.log(d.monthly_surplus_inr.clip(lower=500)), np.log(d.willing_monthly_investment_inr))[0]
    c2.markdown(f"**Explanation:** a very strong log–log relationship (r = {r_:.2f}). The median respondent would invest {d.invest_to_surplus_ratio.median():.1%} of their monthly surplus, so two quick questions (income, expenses) predict ticket size before the first meeting.")
    H = s.filter(like="holds_").rename(columns=lambda c: c.replace("holds_", "").replace("_", " ")).corr()
    fig = px.imshow(H.where(np.tril(np.ones_like(H, dtype=bool), -1)), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-.6, zmax=.6, title="Product co-holding (φ correlation)"); st.plotly_chart(fig, width="stretch")
    hp = H.where(np.tril(np.ones_like(H, dtype=bool), -1)).stack().sort_values(key=abs, ascending=False)
    st.markdown(f"**Explanation:** the strongest co-holding is {hp.index[0][0]} with {hp.index[0][1]} (φ = {hp.iloc[0]:.2f}); traditional LIC is negatively linked to mutual funds (φ = {H.loc['LIC Traditional','Mutual Funds']:.2f}). Households split into market-linked and traditional portfolio styles, and the traditional group is the conversion opportunity.")

# ================= DIAGNOSTIC =================
with tab[3]:
    c1, c2 = st.columns(2)
    g = s.pivot_table(index="trust_in_advisors", columns="awareness_wealth_mgmt", values="engage_flag", aggfunc="mean")
    n_ = s.pivot_table(index="trust_in_advisors", columns="awareness_wealth_mgmt", values="engage_flag", aggfunc="count")
    fig = px.imshow(g.where(n_ >= 15), text_auto=".0%", origin="lower", color_continuous_scale="Blues", labels={"x": "Awareness (1–5)", "y": "Trust (1–5)", "color": "Sign-up"}, title="Sign-up by trust × awareness (cells n ≥ 15)")
    c1.plotly_chart(fig, width="stretch")
    gv = g.where(n_ >= 15)
    c1.markdown(f"**Explanation:** intent rises from {pct(gv.min().min())} in the weakest cell to {pct(gv.max().max())} in the strongest. Trust and awareness reinforce each other, so pairing education with trust signals (SEBI registration, local references) beats either alone.")
    rows = []
    for lbl, col in (("Past mis-selling", "past_bad_experience"), ("Invests via app", "uses_investing_app")):
        for v, nm in ((0, "No"), (1, "Yes")):
            q = s.loc[s[col] == v, "engage_flag"]; m = q.mean(); rows.append(dict(factor=lbl, answer=nm, rate=m, ci=1.96 * np.sqrt(m * (1 - m) / max(len(q), 1)), n=len(q)))
    R = pd.DataFrame(rows)
    fig = px.bar(R, x="factor", y="rate", color="answer", barmode="group", error_y="ci", text=R.rate.map(lambda x: f"{x:.0%}"), labels={"rate": "Likely to sign up", "factor": ""}, title="Barriers to sign-up (95% CI)"); fig.update_yaxes(tickformat=".0%")
    c2.plotly_chart(fig, width="stretch")
    pv1 = stats.chi2_contingency(pd.crosstab(s.past_bad_experience, s.engage_flag))[1]; pv2 = stats.chi2_contingency(pd.crosstab(s.uses_investing_app, s.engage_flag))[1]
    a_, b_ = R.iloc[0].rate, R.iloc[1].rate
    c2.markdown(f"**Explanation:** respondents previously mis-sold sign up at {pct(b_)} against {pct(a_)} (χ² p = {pv1:.1e}). App users are {'not ' if pv2 > .05 else ''}significantly different (p = {pv2:.2f}). Damaged trust from past agents, not app competition, is the barrier to address.")
    c1, c2 = st.columns(2)
    d = s.dropna(subset=["willing_monthly_investment_inr"])
    fig = px.box(d, x="risk_tolerance", y="willing_monthly_investment_inr", log_y=True, points=False, labels={"risk_tolerance": "Risk tolerance (1–5)", "willing_monthly_investment_inr": "Stated investment, ₹ (log)"}, title="Investment by risk tolerance")
    c1.plotly_chart(fig, width="stretch")
    kw = stats.kruskal(*[g_.willing_monthly_investment_inr for _, g_ in d.groupby("risk_tolerance") if len(g_) > 2]); rr = d.groupby("risk_tolerance").invest_to_surplus_ratio.median()
    c1.markdown(f"**Explanation:** median investment rises with risk tolerance (Kruskal–Wallis H = {kw.statistic:.0f}, p = {kw.pvalue:.1e}). Even after allowing for income, the share of surplus committed rises from {rr.iloc[0]:.1%} at the lowest rating to {rr.iloc[-1]:.1%} at the highest. Income sets what people *can* invest; risk tolerance goes with how much they *will* commit.")
    lb = pd.cut(s.financial_literacy_score, [-1, 4, 6, 10], labels=["Low (0–4)", "Medium (6)", "High (8–10)"])
    ft = pd.crosstab(lb, s.preferred_fee_model, normalize="index").reset_index().melt(id_vars="financial_literacy_score")
    fig = px.bar(ft, y="financial_literacy_score", x="value", color="preferred_fee_model", orientation="h", labels={"value": "Share", "financial_literacy_score": "Literacy", "preferred_fee_model": "Fee model"}, title="Preferred fee model by literacy"); fig.update_xaxes(tickformat=".0%")
    c2.plotly_chart(fig, width="stretch")
    lo = (s[lb == "Low (0–4)"].preferred_fee_model == "Commission-based").mean(); hi = (s[lb == "High (8–10)"].preferred_fee_model == "Commission-based").mean()
    c2.markdown(f"**Explanation:** {pct(lo)} of low-literacy respondents prefer commission-based pricing against {pct(hi)} of high-literacy respondents, who are more open to flat or asset-based fees. That supports a two-track model: distributor for the mass market, fee-based advisory for affluent clients.")
    st.subheader("All factors together: logistic regression of sign-up intent")
    try:
        import statsmodels.formula.api as smf
        dm = s.assign(tier2=(s.city_tier == "Tier 2").astype(int), tier3=(s.city_tier == "Tier 3").astype(int))
        terms = ["trust_in_advisors", "awareness_wealth_mgmt", "financial_literacy_score", "log_income", "past_bad_experience", "uses_investing_app", "num_goals", "risk_tolerance", "importance_face_to_face"] + [t for t in ("tier2", "tier3") if dm[t].nunique() > 1]
        lg = smf.logit("engage_flag ~ " + " + ".join(terms), dm).fit(disp=0)
        nm = {"trust_in_advisors": "Trust (+1)", "awareness_wealth_mgmt": "Awareness (+1)", "financial_literacy_score": "Literacy (+1)", "log_income": "Income (×2.7)", "past_bad_experience": "Past mis-selling", "uses_investing_app": "App user", "num_goals": "Goals (+1)", "risk_tolerance": "Risk tolerance (+1)", "importance_face_to_face": "Face-to-face (+1)", "tier2": "Tier 2 vs Tier 1", "tier3": "Tier 3 vs Tier 1"}
        OR = pd.DataFrame({"factor": [nm[t] for t in terms], "OR": np.exp(lg.params[terms]).values, "lo": np.exp(lg.conf_int().loc[terms, 0]).values, "hi": np.exp(lg.conf_int().loc[terms, 1]).values, "p": lg.pvalues[terms].values}).sort_values("OR")
        fig = go.Figure(go.Scatter(x=OR.OR, y=OR.factor, mode="markers", error_x=dict(type="data", symmetric=False, array=OR.hi - OR.OR, arrayminus=OR.OR - OR.lo), marker_color=COLORS[0]))
        fig.add_vline(x=1, line_dash="dash"); fig.update_xaxes(type="log", title="Odds ratio (95% CI), log scale"); fig.update_layout(title=f"Odds ratios for sign-up (n = {int(lg.nobs):,}, pseudo-R² = {lg.prsquared:.2f})", height=420)
        st.plotly_chart(fig, width="stretch")
        sig = OR[OR.p < .05]; ns = OR[OR.p >= .05]
        st.markdown(f"**Explanation:** with all factors in one model, the largest positive odds ratio is **{sig.iloc[-1].factor}** (×{sig.iloc[-1].OR:.2f}) and the strongest negative is **{sig.iloc[0].factor}** (×{sig.iloc[0].OR:.2f}). Not significant after controls: {', '.join(ns.factor) if len(ns) else 'none'}. If city tier appears here, its raw gap is explained by income, awareness and trust rather than by geography.")
    except Exception as e:
        st.info("Too few respondents in this filter to fit the multivariate model.")
    G = s.groupby("num_goals").engage_flag.agg(["mean", "count"]).reset_index(); G = G[G["count"] >= 20]
    fig = px.line(G, x="num_goals", y="mean", markers=True, labels={"num_goals": "Number of financial goals", "mean": "Likely to sign up"}, title="Sign-up intent by number of goals"); fig.update_yaxes(tickformat=".0%"); st.plotly_chart(fig, width="stretch")
    st.markdown(f"**Explanation:** intent rises from {pct(G['mean'].iloc[0])} with {G.num_goals.iloc[0]} goals to a peak of {pct(G['mean'].max())}. People with concrete goals see the point of planning, so sales conversations should open with goals, not products.")

# ================= PIPELINE =================
with tab[4]:
    cnt = pd.Series({st_: (p.stage_index >= i).sum() for i, st_ in enumerate(STAGES)})
    fig = go.Figure(go.Funnel(y=STAGES, x=cnt.values, textinfo="value+percent previous+percent initial", marker_color=[COLORS[0]] * 5 + [COLORS[2]]))
    fig.update_layout(title="Sales funnel (pilot, Jul 2025 – Jun 2026)", height=420); st.plotly_chart(fig, width="stretch")
    step = (cnt / cnt.shift()).iloc[1:]; drop = (cnt.shift() - cnt).iloc[1:]
    lost_tot = int(cnt.iloc[0] - cnt.iloc[-1]); lost_before = int(cnt.iloc[0] - cnt.iloc[2])
    st.markdown(f"**Explanation:** {cnt.iloc[-1]:,} of {cnt.iloc[0]:,} leads became clients ({cnt.iloc[-1]/cnt.iloc[0]:.1%}). The weakest step is *{step.idxmin()}* ({step.min():.0%} of the previous stage move on), and the largest absolute loss happens entering *{drop.idxmax()}* ({drop.max():,.0f} leads). Of the leads not won, {lost_before:,} ({lost_before/max(lost_tot,1):.0%}) dropped out before meeting an advisor and {lost_tot-lost_before:,} after, so both booking meetings and converting after them need work.")
    c1, c2 = st.columns(2)
    src = p.groupby("lead_source").agg(leads=("lead_id", "count"), won=("won_flag", "sum"), cost=("acquisition_cost_inr", "sum")).reset_index()
    src["win_rate"] = src.won / src.leads; src["cost_per_client"] = src.cost / src.won.replace(0, np.nan); src = src.sort_values("win_rate", ascending=False)
    fig = px.bar(src, x="lead_source", y="win_rate", text=src.win_rate.map(lambda x: f"{x:.0%}"), labels={"lead_source": "", "win_rate": "Win rate"}, title="Win rate by lead source"); fig.update_yaxes(tickformat=".0%"); c1.plotly_chart(fig, width="stretch")
    q = src.dropna(subset=["cost_per_client"])
    fig = px.scatter(q, x="win_rate", y="cost_per_client", size="leads", text="lead_source", labels={"win_rate": "Win rate", "cost_per_client": "Acquisition cost per client, ₹"}, title="Cost per client vs win rate (bubble = leads)"); fig.update_traces(textposition="top center"); fig.update_xaxes(tickformat=".0%")
    c2.plotly_chart(fig, width="stretch")
    if len(q):
        b, w = q.loc[q.cost_per_client.idxmin()], q.loc[q.cost_per_client.idxmax()]
        st.markdown(f"**Explanation:** the best source is **{src.iloc[0].lead_source}** ({pct(src.iloc[0].win_rate)} win rate). The cheapest client comes from **{b.lead_source}** ({inr(b.cost_per_client)}), the most expensive from **{w.lead_source}** ({inr(w.cost_per_client)}, {w.cost_per_client/b.cost_per_client:.1f}× more). Warm, trust-based introductions beat paid and cold channels, so shift budget towards referral and partner programmes.")
    ue = p.groupby("lead_source").agg(won=("won_flag", "sum"), cost=("acquisition_cost_inr", "sum"), rev=("first_year_trail_revenue_inr", "sum")).reset_index()
    ue = ue[(ue.lead_source != "Unknown") & (ue.won > 0)]; ue["payback_years"] = (ue.cost / ue.won) / (ue.rev / ue.won); ue = ue.sort_values("payback_years")
    fig = px.bar(ue, y="lead_source", x="payback_years", orientation="h", text=ue.apply(lambda r: f"{r.payback_years:.1f} yrs (n={int(r.won)})", axis=1), labels={"lead_source": "", "payback_years": "Years of first-year trail revenue to repay acquisition cost"}, title="Payback on acquisition cost, by channel")
    fig.add_vline(x=1, line_dash="dash"); st.plotly_chart(fig, width="stretch")
    if len(ue):
        bl = p.acquisition_cost_inr.sum() / max(p.won_flag.sum(), 1); br = p.first_year_trail_revenue_inr.sum() / max(p.won_flag.sum(), 1)
        st.markdown(f"**Explanation:** acquisition cost per client divided by first-year trail revenue per client (0.8% assumed trail). Fastest payback: **{ue.iloc[0].lead_source}** ({ue.iloc[0].payback_years:.1f} years); slowest: **{ue.iloc[-1].lead_source}** ({ue.iloc[-1].payback_years:.1f} years). Blended: ₹{bl:,.0f} to win a client against ₹{br:,.0f} earned in year 1. Channels with few clients (small n) are unreliable.")
    c1, c2 = st.columns(2)
    order_r = ["≤2 h", "2–24 h", "1–3 days", ">3 days"]
    rr = p[p.response_bucket.isin(order_r)].groupby("response_bucket").won_flag.agg(["mean", "count"]).reindex(order_r).reset_index()
    fig = px.bar(rr, x="response_bucket", y="mean", text=rr["mean"].map(lambda x: f"{x:.0%}"), labels={"response_bucket": "Time to first contact", "mean": "Win rate"}, title="Win rate by speed of first response"); fig.update_yaxes(tickformat=".0%"); c1.plotly_chart(fig, width="stretch")
    c1.markdown(f"**Explanation:** leads contacted within 2 hours win {pct(rr['mean'].iloc[0])}, against {pct(rr['mean'].iloc[-1])} after 3 days. Interest decays fast, so a same-day response standard pays off.")
    fu = p[p.stage_index >= 1].groupby("followup_bucket").won_flag.mean().reindex(["0", "1–2", "3–5", "6+"]).reset_index()
    fig = px.bar(fu, x="followup_bucket", y="won_flag", text=fu.won_flag.map(lambda x: f"{x:.0%}"), labels={"followup_bucket": "Follow-ups", "won_flag": "Win rate"}, title="Win rate by number of follow-ups"); fig.update_yaxes(tickformat=".0%"); c2.plotly_chart(fig, width="stretch")
    c2.markdown(f"**Explanation:** win rates peak at **{fu.loc[fu.won_flag.idxmax(),'followup_bucket']}** follow-ups ({pct(fu.won_flag.max())}) and fall after that: diminishing returns. Beyond about five, effort is better spent on new leads.")
    L = p[p.outcome == "Lost"]
    lt = pd.crosstab(L.segment, L.lost_reason, normalize="index")
    fig = px.imshow(lt, text_auto=".0%", color_continuous_scale="Oranges", aspect="auto", labels={"x": "", "y": "", "color": "Share"}, title="Why leads were lost, by segment"); st.plotly_chart(fig, width="stretch")
    top = lt.idxmax(axis=1)
    st.markdown("**Explanation:** the top lost reason differs by segment: " + "; ".join(f"{sg}: *{rsn.lower()}* ({lt.loc[sg, rsn]:.0%})" for sg, rsn in top.items()) + ". Each segment needs its own objection-handling playbook.")
    c1, c2 = st.columns(2)
    W = p[(p.won_flag == 1) & p.days_to_close.notna()]
    if len(W) > 5:
        fig = px.box(W, y="segment", x="days_to_close", points=False, labels={"segment": "", "days_to_close": "Days from lead to client"}, title="Sales cycle by segment"); c1.plotly_chart(fig, width="stretch")
        mm = W.groupby("segment").days_to_close.median()
        c1.markdown(f"**Explanation:** the median sale takes {W.days_to_close.median():.0f} days; {mm.idxmin()} close fastest ({mm.min():.0f} days) and {mm.idxmax()} slowest ({mm.max():.0f} days). Slow segments need multi-meeting relationship selling and longer forecast horizons.")
    M = p.groupby("created_month").agg(leads=("lead_id", "count"), won=("won_flag", "mean")).reset_index()
    fig = go.Figure([go.Bar(x=M.created_month, y=M.leads, name="Leads", marker_color="#bdd7ee"), go.Scatter(x=M.created_month, y=M.won, name="Win rate", yaxis="y2", mode="lines+markers", line_color=COLORS[1])])
    fig.update_layout(title="Leads per month and their win rate", yaxis=dict(title="Leads"), yaxis2=dict(title="Win rate", overlaying="y", side="right", tickformat=".0%"), legend=dict(orientation="h")); c2.plotly_chart(fig, width="stretch")
    pk = M.loc[M.leads.idxmax()]
    c2.markdown(f"**Explanation:** lead volume peaks in {pk.created_month} ({pk.leads:,} leads, tax-saving season), but that month's win rate ({pct(pk.won)}) is no better than average ({pct(p.won_flag.mean())}). The last months look weaker partly because some leads are still open.")
    c1, c2 = st.columns(2)
    rm = p.groupby("assigned_rm").won_flag.agg(["mean", "count"]).reset_index(); rm["ci"] = 1.96 * np.sqrt(rm["mean"] * (1 - rm["mean"]) / rm["count"]); rm = rm.sort_values("mean")
    fig = px.bar(rm, y="assigned_rm", x="mean", error_x="ci", orientation="h", labels={"assigned_rm": "", "mean": "Win rate"}, title="Win rate by relationship manager (95% CI)"); fig.update_xaxes(tickformat=".0%"); c1.plotly_chart(fig, width="stretch")
    c1.markdown(f"**Explanation:** win rates range from {pct(rm['mean'].min())} ({rm.iloc[0].assigned_rm}) to {pct(rm['mean'].max())} ({rm.iloc[-1].assigned_rm}). Where the intervals don't overlap, the gap is unlikely to be chance. Lead-source mix by RM is similar (χ² p = {stats.chi2_contingency(pd.crosstab(p.assigned_rm, p.lead_source))[1]:.2f}), so the gap isn't explained by lead quality; pairing the strongest and weakest RMs for coaching is worth testing.")
    mt = p[p.stage_index >= 2].groupby(["meeting_mode", "city_tier"]).won_flag.mean().reset_index()
    fig = px.bar(mt, x="meeting_mode", y="won_flag", color="city_tier", barmode="group", category_orders={"city_tier": TIERS}, labels={"meeting_mode": "", "won_flag": "Win rate after meeting", "city_tier": "Tier"}, title="Win rate after a meeting, by mode and tier"); fig.update_yaxes(tickformat=".0%"); c2.plotly_chart(fig, width="stretch")
    mo = p[p.stage_index >= 2].groupby("meeting_mode").won_flag.mean()
    c2.markdown(f"**Explanation:** in-person meetings win {pct(mo.get('In person', np.nan))} after a meeting, against {pct(mo.get('Video call', np.nan))} for video and {pct(mo.get('Phone', np.nan))} for phone, consistent with the survey's strong preference for face-to-face advice.")

    W = p[p.won_flag == 1]
    if len(W) > 10:
        rt = W.groupby("segment").churned_flag.agg(["mean", "count"]).reset_index()
        fig = px.bar(rt, y="segment", x="mean", orientation="h", text=rt.apply(lambda r: f"{r['mean']:.0%} (n={int(r['count'])})", axis=1), labels={"segment": "", "mean": "Share of won clients who stopped their SIP"}, title="After the sale: churn by 30 Jun 2026"); fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        lo_ = W[W.sip_realisation < .7].churned_flag.mean(); hi_ = W[W.sip_realisation >= .7].churned_flag.mean()
        st.markdown(f"**Explanation:** {pct(W.churned_flag.mean())} of won clients had stopped after a median {W.months_since_onboarding.median():.0f} months. Clients who started below 70% of the proposed SIP churned at {pct(lo_)}, against {pct(hi_)} for the rest, so a low starting commitment is an early warning sign. The window is short, so these are early indicators, not lifetime churn.")

# ================= INSIGHTS =================
with tab[5]:
    st.subheader("End-to-end validation: the same people's survey answers vs what they did")
    Lk = p_all[p_all.survey_link_valid == 1].merge(s_all[["respondent_id", "likely_to_engage", "willing_monthly_investment_inr"]], left_on="survey_respondent_id", right_on="respondent_id")
    Lk = Lk[Lk.city_tier.isin(tiers) & Lk.segment.isin(segs)]; Lc = Lk[Lk.outcome != "Open"]
    if len(Lc) > 20:
        c1, c2 = st.columns(2)
        wr = Lc.groupby("likely_to_engage").won_flag.agg(["mean", "count"]).reset_index()
        fig = px.bar(wr, x="likely_to_engage", y="mean", text=wr.apply(lambda r: f"{r['mean']:.0%} (n={int(r['count'])})", axis=1), labels={"likely_to_engage": "Said in the survey they were likely to sign up", "mean": "Became a client"}, title="Linked survey opt-in leads: stated intent vs outcome"); fig.update_yaxes(tickformat=".0%"); c1.plotly_chart(fig, width="stretch")
        Lw = Lk[(Lk.won_flag == 1) & (Lk.willing_monthly_investment_inr > 0)]
        fig = px.scatter(Lw, x="willing_monthly_investment_inr", y="first_sip_inr", log_x=True, log_y=True, labels={"willing_monthly_investment_inr": "Stated in survey, ₹/month", "first_sip_inr": "Actual first SIP, ₹/month"}, title="Same people: stated vs actual SIP"); c2.plotly_chart(fig, width="stretch")
        y_ = wr.set_index("likely_to_engage")["mean"]
        rho = stats.spearmanr(Lw.willing_monthly_investment_inr, Lw.first_sip_inr)[0] if len(Lw) > 5 else np.nan
        st.markdown(f"**Explanation:** respondents who said 'likely' became clients {pct(y_.get('Yes', np.nan))} of the time, against {pct(y_.get('No', np.nan))} of the rest. Stated and actual SIPs rank closely (Spearman ρ = {rho:.2f}), and actual SIPs were a median {(Lw.first_sip_inr / Lw.willing_monthly_investment_inr).median():.2f}× the stated amount. The survey predicts real behaviour, but forecasts need both discounts.")
    else:
        st.info("Too few linked leads in this filter.")
    sv = s.groupby("segment").engage_flag.mean(); pv = p[p.lead_source != "Survey opt-in"].groupby("segment").won_flag.mean()
    B = pd.DataFrame({"Survey: likely to sign up": sv, "Pipeline: leads won": pv}).reset_index().melt(id_vars="segment")
    fig = px.bar(B, y="segment", x="value", color="variable", barmode="group", orientation="h", labels={"value": "", "segment": "", "variable": ""}, title="Segment level: stated intent vs win rate (non-survey channels)"); fig.update_xaxes(tickformat=".0%"); st.plotly_chart(fig, width="stretch")
    st.markdown("**Explanation:** the segment ranking holds: segments with higher stated intent also win more leads through other channels. Survey respondents and other leads are different groups, so the record-level comparison above is the more reliable test.")
    st.subheader("Key insights")
    st.markdown("""
1. **Demand is real but concentrated.** It is strongest among affluent professionals and conservative salaried households. Lower Tier 3 intent reflects lower income, awareness and trust, not geography itself.
2. **Trust and awareness go with *whether* people sign up; income and surplus go with *how much*.** Past mis-selling is the largest barrier.
3. **Savings sit in FDs, gold and LIC.** Retirement, goal planning and insurance review are the entry services.
4. **The pipeline leaks both before and after the first meeting, in roughly equal measure.** Survey opt-ins, referrals and CA/doctor partners convert best and repay their acquisition cost within a year. A same-day response, 3–5 follow-ups and in-person meetings help. Seminars, ads and RWA camps take 2–4 years to pay back.
5. **The survey predicts behaviour.** For the same people, 'likely' respondents became clients about 3.6× as often, and stated and actual SIPs rank almost identically.
6. **Lost reasons differ by segment.** Business families need trust-building, young earners a hybrid app-plus-advisor offer, conservative savers FD alternatives, and affluent clients a clear fee justification.
""")
