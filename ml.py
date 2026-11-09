"""
ml.py – predictive and pattern-mining models used by the dashboard's
Classification, Clustering, Association rules and Regression tabs.
All functions are pure and cached by app.py. Models learn only from profile,
finance, attitude, goal and holdings answers, never from the outcome they predict.
"""
import itertools
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (roc_auc_score, accuracy_score, roc_curve, r2_score, median_absolute_error,
                             adjusted_rand_score, silhouette_score)

LIK = ["risk_tolerance", "digital_comfort", "trust_in_advisors", "importance_face_to_face", "awareness_wealth_mgmt"]
NICE = {"age": "Age", "dependents": "Dependents", "log_income": "Income (log)", "log_savings": "Savings (log)",
        "expense_pct_income": "Expenses % of income", "financial_literacy_score": "Financial literacy", "num_goals": "Number of goals",
        "risk_tolerance": "Risk tolerance", "digital_comfort": "Digital comfort", "trust_in_advisors": "Trust in advisors",
        "importance_face_to_face": "Face-to-face importance", "awareness_wealth_mgmt": "Awareness of wealth mgmt",
        "past_bad_experience": "Past mis-selling", "num_products_held": "Products held", "uses_investing_app": "Uses investing app", "has_emi": "Has an EMI", "married": "Married"}


def _nice(c):
    if c in NICE: return NICE[c]
    if c.startswith("holds_"): return "Holds " + c[6:].replace("_", " ")
    if c.startswith("int_"): return "Wants " + c[4:].replace("_", " ")
    for p in ("city_tier_", "occupation_", "education_", "gender_"):
        if c.startswith(p): return c[len(p):]
    return c


def features(s):
    hold = [c for c in s if c.startswith("holds_")]
    num = ["age", "dependents", "log_income", "log_savings", "expense_pct_income", "financial_literacy_score", "num_goals"] + LIK + \
          ["past_bad_experience", "uses_investing_app", "has_emi", "married"] + hold
    cat = ["city_tier", "occupation", "education", "gender"]
    return pd.get_dummies(s[num + cat], columns=cat, drop_first=True).astype(float)


# ---------------- classification: will they sign up? ----------------
def classification(s):
    X, y = features(s), s.engage_flag.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    models = {"Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
              "Random forest": RandomForestClassifier(300, min_samples_leaf=5, random_state=0, n_jobs=-1),
              "Gradient boosting": GradientBoostingClassifier(random_state=0)}
    rows, curves, gains = [], {}, {}
    for n, m in models.items():
        m.fit(Xtr, ytr); pr = m.predict_proba(Xte)[:, 1]
        o = np.argsort(-pr); g = np.r_[0, yte.values[o].cumsum() / yte.sum()]
        rows.append(dict(model=n, auc=roc_auc_score(yte, pr), accuracy=accuracy_score(yte, pr >= .5),
                         top30=g[int(round(.3 * len(yte)))]))
        fpr, tpr, _ = roc_curve(yte, pr); curves[n] = (fpr, tpr); gains[n] = g
    res = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    lr = models["Logistic regression"]
    coef = pd.Series(lr[-1].coef_[0], index=X.columns)
    coef = coef.reindex(coef.abs().sort_values(ascending=False).index)[:12]
    imp = pd.DataFrame({"feature": [_nice(c) for c in coef.index], "coef": coef.values})
    return dict(results=res, roc=curves, gains=gains, baseline=1 - y.mean(), n_train=len(Xtr), n_test=len(Xte),
                importance=imp, base_rate=y.mean())


# ---------------- clustering: who are the customers? ----------------
CLU = ["age", "log_income", "log_savings", "financial_literacy_score"] + LIK + ["num_products_held", "holds_Real_Estate", "holds_Mutual_Funds", "holds_Chit_Committee"]


def clustering(s):
    Z = StandardScaler().fit_transform(s[CLU])
    km = KMeans(5, n_init=20, random_state=0).fit(Z)
    sil = [dict(k=k, silhouette=silhouette_score(Z, KMeans(k, n_init=10, random_state=0).fit_predict(Z), sample_size=2000, random_state=0))
           for k in range(2, 9)]
    rng = np.random.default_rng(0); aris = []
    for i in range(10):
        idx = rng.choice(len(Z), len(Z), replace=True)
        aris.append(adjusted_rand_score(km.labels_, KMeans(5, n_init=10, random_state=i).fit(Z[idx]).predict(Z)))
    zs = pd.DataFrame(Z, columns=[_nice(c) for c in CLU]); zs["segment"] = s.segment.values
    finger = zs.groupby("segment").mean()
    prof = s.groupby("segment").agg(people=("respondent_id", "size"), sign_up=("engage_flag", "mean"),
                                    median_income_lakh=("annual_income_lakh", "median"),
                                    median_monthly_invest=("willing_monthly_investment_inr", "median"))
    prof["share"] = prof.people / prof.people.sum()
    return dict(sil=pd.DataFrame(sil), ari=float(np.mean(aris)), ari_min=float(np.min(aris)),
                sil5=[r["silhouette"] for r in sil if r["k"] == 5][0], finger=finger, profile=prof.sort_values("sign_up", ascending=False))


# ---------------- association rules: what goes together? ----------------
def rules(s, min_support=.05, min_conf=.3):
    items = [c for c in s if c.startswith("holds_")] + [c for c in s if c.startswith("int_")]
    B = s[items].astype(bool); targets = [c for c in s if c.startswith("int_")]; out = []
    for k in (1, 2):
        for A in itertools.combinations(items, k):
            mA = B[list(A)].all(axis=1); sA = mA.mean()
            if sA < min_support: continue
            for C in targets:
                if C in A: continue
                sAC = (mA & B[C]).mean()
                if sAC < min_support: continue
                conf = sAC / sA
                if conf < min_conf: continue
                out.append(dict(rule=" + ".join(_nice(a) for a in A) + " → " + _nice(C), support=sAC, confidence=conf,
                                lift=conf / B[C].mean(), size=k))
    R = pd.DataFrame(out)
    return R.sort_values("lift", ascending=False).reset_index(drop=True) if len(R) else R


# ---------------- regression: how much will they invest? ----------------
def regression(s):
    r = s[s.willing_monthly_investment_inr > 0]
    X, y = features(r), np.log(r.willing_monthly_investment_inr)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42)
    base = np.exp(ytr.median())
    rows = [dict(model="Median guess (baseline)", r2=0.0, med_err=median_absolute_error(np.exp(yte), np.full(len(yte), base)))]
    models = {"Linear regression": LinearRegression(),
              "Random forest": RandomForestRegressor(300, min_samples_leaf=5, random_state=0, n_jobs=-1),
              "Gradient boosting": GradientBoostingRegressor(random_state=0)}
    preds = {}
    for n, m in models.items():
        m.fit(Xtr, ytr); p = m.predict(Xte); preds[n] = p
        rows.append(dict(model=n, r2=r2_score(yte, p), med_err=median_absolute_error(np.exp(yte), np.exp(p))))
    res = pd.DataFrame(rows)
    best = res.iloc[1:].sort_values("r2", ascending=False).iloc[0].model
    gb = models["Gradient boosting"]
    imp = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
    imp = pd.DataFrame({"feature": [_nice(c) for c in imp.index], "importance": imp.values})
    scatter = pd.DataFrame({"actual": np.exp(yte.values), "predicted": np.exp(preds[best])})
    return dict(results=res, best=best, importance=imp, scatter=scatter, n=len(r), median=r.willing_monthly_investment_inr.median(),
                mean=r.willing_monthly_investment_inr.mean())
