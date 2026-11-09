"""
02_clean_transform.py
Cleans and transforms the two raw datasets.
Inputs : data/raw/survey_raw.csv, data/raw/pipeline_raw.csv
Outputs: data/clean/survey_clean.csv        cleaned survey + engineered features
         data/clean/survey_model_ready.csv  one-hot encoded + standardised numeric features
         data/clean/pipeline_clean.csv      cleaned pipeline + engineered features
         data/clean/cleaning_log.csv        every issue found, rows affected, action taken
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
LOG = []
def log(ds, issue, n, action): LOG.append(dict(dataset=ds, issue=issue, rows_affected=int(n), action=action))

# =============================== SURVEY ===============================
s = pd.read_csv("data/raw/survey_raw.csv"); n0 = len(s); log("survey", "Raw rows loaded", n0, "—")
LIK = ["risk_tolerance","digital_comfort","trust_in_advisors","importance_face_to_face","awareness_wealth_mgmt"]
d = s.duplicated().sum(); s = s.drop_duplicates(); log("survey","Exact duplicate submissions",d,"Removed")
NEAR = ["city","age","gender","occupation","annual_income_lakh","total_savings_lakh","risk_tolerance","digital_comfort","trust_in_advisors","importance_face_to_face","awareness_wealth_mgmt","willing_monthly_investment_inr"]
nd = s.assign(city=s.city.str.strip().str.lower()).duplicated(subset=NEAR, keep=False).sum(); log("survey","Near-duplicates (same answers, different respondent id)",nd,"Checked; none found" if nd==0 else "Flagged for review")
# missingness check: is non-response concentrated in some groups? (chi-square, missing vs city tier)
from scipy.stats import chi2_contingency
MISS=[]
for c in ["annual_income_lakh","total_savings_lakh","expense_pct_income","max_annual_fee_inr","willing_monthly_investment_inr"]:
    for g in ["city_tier","occupation"]:
        t = pd.crosstab(s[g], s[c].isna()); pv = chi2_contingency(t)[1] if t.shape[1]==2 else 1.0
        MISS.append(dict(column=c, by=g, overall_missing=round(s[c].isna().mean(),4), min_group=round(s.groupby(g)[c].apply(lambda x:x.isna().mean()).min(),4), max_group=round(s.groupby(g)[c].apply(lambda x:x.isna().mean()).max(),4), chi2_p=round(pv,4)))
MISS=pd.DataFrame(MISS); MISS.to_csv("data/clean/missingness_check.csv",index=False)
log("survey","Missingness test: missing vs city tier and occupation (10 chi-square tests)",int((MISS.chi2_p<0.05).sum()),"Number of tests with p < 0.05; non-response is spread evenly, so group-median imputation is reasonable" if (MISS.chi2_p<0.05).sum()<=1 else "Some non-response is concentrated; imputed within groups to respect this")
CITY_CANON = {c.lower(): c for c in ["Lucknow","Noida","Ghaziabad","Kanpur","Agra","Varanasi","Prayagraj","Meerut","Bareilly","Gorakhpur","Aligarh","Small town","Village / Rural"]}
fixed_city = s.city.str.strip().str.lower().map(CITY_CANON)
bad_city = (fixed_city != s.city).sum(); s["city"] = fixed_city; log("survey","City names in wrong case or with trailing spaces",bad_city,"Mapped to the 13 canonical city names")
lab = s.likely_to_engage.str.strip().str.lower().map({"yes":"Yes","y":"Yes","no":"No","n":"No"})
log("survey","Inconsistent Yes/No labels in target",(lab != s.likely_to_engage).sum(),"Mapped to Yes/No"); s["likely_to_engage"] = lab
a = s.attention_check != "Agree"; log("survey","Failed attention check",a.sum(),"Removed (careless respondents)"); s = s[~a]
sl = s[LIK].nunique(axis=1).eq(1) & s[LIK].notna().all(axis=1); log("survey","Same answer on all 5 attitude items",sl.sum(),"Flagged (straightliner_flag), kept"); s["straightliner_flag"] = sl.astype(int)
ba = (s.age < 18) | (s.age > 100); log("survey","Impossible ages (0, 4, 150, 199, 250)",ba.sum(),"Set to missing, then imputed"); s.loc[ba,"age"] = np.nan
ir = (s.occupation == "Retired / Pensioner") & (s.age < 40); log("survey","Retired but under 40 (contradiction)",ir.sum(),"Removed"); s = s[~ir]
ue = ((s.annual_income_lakh > 300) & (s.willing_monthly_investment_inr.abs() < 60000)) | (s.annual_income_lakh > 2000); log("survey","Income typed in ₹ thousands (100× too big): above ₹300 L with a small investment, or above ₹2,000 L",ue.sum(),"Divided by 100"); s.loc[ue,"annual_income_lakh"] /= 100
ng = s.willing_monthly_investment_inr < 0; log("survey","Negative investment amount",ng.sum(),"Sign corrected"); s.loc[ng,"willing_monthly_investment_inr"] = s.loc[ng,"willing_monthly_investment_inr"].abs()
ex = s.expense_pct_income > 100; log("survey","Expenses above 100% of income",ex.sum(),"Set to missing, then imputed"); s.loc[ex,"expense_pct_income"] = np.nan
NUM = ["age","annual_income_lakh","total_savings_lakh","expense_pct_income","max_annual_fee_inr"]
miss = s[NUM+LIK].isna().sum(); 
for c in NUM:
    if miss[c]: log("survey",f"Missing values: {c}",miss[c],"Median by city tier × occupation")
    s[c] = s[c].fillna(s.groupby(["city_tier","occupation"])[c].transform("median")).fillna(s[c].median())
for c in LIK:
    if miss[c]: log("survey",f"Missing values: {c}",miss[c],"Mode by city tier")
    s[c] = s[c].fillna(s.groupby("city_tier")[c].transform(lambda x: x.mode()[0])).astype(int)
mt = s.willing_monthly_investment_inr.isna(); log("survey","Missing monthly investment (outcome)",mt.sum(),"Kept as missing; excluded only where this column is analysed")
q1,q3 = s.annual_income_lakh.quantile([.25,.75]); hni = s.annual_income_lakh > q3 + 3*(q3-q1)
log("survey","Extreme incomes beyond Q3 + 3×IQR (genuine HNIs)",hni.sum(),"Kept, flagged (hni_flag); log scale used in analysis"); s["hni_flag"] = hni.astype(int)
s["age"] = s.age.round().astype(int)
# ---- transformations / feature engineering ----
hold=[c for c in s if c.startswith("holds_")]; ints=[c for c in s if c.startswith("int_")]; goals=[c for c in s if c.startswith("goal_")]
s["financial_literacy_score"] = 2*s.filter(like="lit_q").sum(axis=1)
s["num_products_held"] = s[hold].sum(axis=1); s["num_services_wanted"] = s[ints].sum(axis=1); s["num_goals"] = s[goals].sum(axis=1)
s["engage_flag"] = (s.likely_to_engage == "Yes").astype(int)
s["log_income"] = np.log(s.annual_income_lakh); s["log_savings"] = np.log1p(s.total_savings_lakh)
s["log_monthly_investment"] = np.log1p(s.willing_monthly_investment_inr)
s["monthly_surplus_inr"] = (s.annual_income_lakh*1e5/12*(1-s.expense_pct_income/100)).round(0)
s["invest_to_surplus_ratio"] = (s.willing_monthly_investment_inr / s.monthly_surplus_inr.clip(lower=500)).round(3)
s["age_band"] = pd.cut(s.age,[17,30,40,50,60,101],labels=["18–30","31–40","41–50","51–60","60+"]).astype(str)
s["income_band"] = pd.cut(s.annual_income_lakh,[0,5,10,20,50,1e9],labels=["<₹5L","₹5–10L","₹10–20L","₹20–50L","₹50L+"]).astype(str)
s["traditional_only"] = ((s.holds_Mutual_Funds==0)&(s.holds_Direct_Stocks==0)&(s.holds_NPS==0)).astype(int)
# customer segment via K-Means on standardised attitude/wealth/behaviour features (k=5)
cf=["age","log_income","log_savings","financial_literacy_score"]+LIK+["num_products_held","holds_Real_Estate","holds_Mutual_Funds","holds_Chit_Committee"]
Z=StandardScaler().fit_transform(s[cf]); km=KMeans(5,n_init=20,random_state=0).fit(Z); s["cluster"]=km.labels_
from sklearn.metrics import silhouette_score; SIL=silhouette_score(Z, km.labels_, sample_size=2000, random_state=0)
g=s.groupby("cluster"); prof=pd.DataFrame(dict(inc=g.annual_income_lakh.median(),dig=g.digital_comfort.mean(),age=g.age.mean(),risk=g.risk_tolerance.mean()))
names={}; left=set(prof.index)
def take(c,n): names[c]=n; left.discard(c)
take(prof.loc[list(left)].inc.idxmax(),"Affluent professionals"); take(prof.loc[list(left)].dig.idxmin(),"Rural & low-literacy savers")
take(prof.loc[list(left)].age.idxmin(),"Young digital earners"); take(prof.loc[list(left)].risk.idxmin(),"Conservative govt & retirees"); take(list(left)[0],"Traditional business families")
s["segment"]=s.cluster.map(names); s=s.drop(columns=["cluster","attention_check"])
log("survey",f"K-Means segment quality (silhouette, k = 5)",0,f"Silhouette = {SIL:.2f}: weak separation, so segments are broad profiles, not sharp groups")
log("survey","Derived features added",14,"literacy score, product/service/goal counts, logs, surplus, ratio, age & income bands, traditional_only, engage_flag, K-Means segment")
log("survey","Clean rows written",len(s),"data/clean/survey_clean.csv")
s.to_csv("data/clean/survey_clean.csv",index=False)
mr_num=["age","dependents","log_income","log_savings","expense_pct_income","financial_literacy_score"]+LIK+["num_products_held","num_services_wanted","num_goals"]
mr=pd.get_dummies(s[["respondent_id"]+mr_num+["city_tier","occupation","education","gender","preferred_language","segment"]+hold+ints+goals+["past_bad_experience","uses_investing_app","engage_flag","log_monthly_investment"]],
                  columns=["city_tier","occupation","education","gender","preferred_language","segment"],drop_first=True)
mr[mr_num]=StandardScaler().fit_transform(mr[mr_num]); mr=mr.astype({c:int for c in mr.columns if mr[c].dtype==bool})
mr.to_csv("data/clean/survey_model_ready.csv",index=False)

# =============================== PIPELINE ===============================
p = pd.read_csv("data/raw/pipeline_raw.csv"); log("pipeline","Raw rows loaded",len(p),"—")
d = p.duplicated().sum(); p = p.drop_duplicates(); log("pipeline","Duplicate lead records",d,"Removed")
ndp = p.duplicated(subset=["created_date","city","age","annual_income_lakh","assigned_rm"], keep=False).sum(); log("pipeline","Near-duplicates (same date, city, age, income, RM; different lead id)",ndp,"Checked; none found" if ndp==0 else "Flagged for review")
STG = {"lead":"Lead","contacted":"Contacted","discovery meeting":"Discovery meeting","discovery mtg":"Discovery meeting","plan presented":"Plan presented","proposal sent":"Plan presented","kyc & onboarding":"KYC & onboarding","kyc":"KYC & onboarding","client":"Client","won":"Client"}
raw_stage=p.stage_reached.copy(); p["stage_reached"]=p.stage_reached.str.strip().str.lower().map(STG)
log("pipeline","Inconsistent stage labels (e.g. 'CLIENT ', 'Proposal sent', 'Won')",(raw_stage!=p.stage_reached).sum(),"Mapped to 6 standard stages")
rmfix=(p.assigned_rm!=p.assigned_rm.str.title()).sum(); p["assigned_rm"]=p.assigned_rm.str.title(); log("pipeline","RM names in inconsistent case",rmfix,"Title-cased")
ms=p.lead_source.isna().sum(); p["lead_source"]=p.lead_source.fillna("Unknown"); log("pipeline","Missing lead source",ms,"Labelled 'Unknown'")
for c in ["created_date","contacted_date","meeting_date","plan_date","kyc_date","closed_date"]: p[c]=pd.to_datetime(p[c])
cb=(p.closed_date<p.created_date); log("pipeline","Closed date before created date",cb.sum(),"Closed date set to missing (cycle time unknown)"); p.loc[cb,"closed_date"]=pd.NaT
ns=p.first_sip_inr<0; log("pipeline","Negative first SIP amount",ns.sum(),"Sign corrected"); p.loc[ns,"first_sip_inr"]=p.loc[ns,"first_sip_inr"].abs()
nr=p.first_response_hours<0; log("pipeline","Negative response time",nr.sum(),"Sign corrected"); p.loc[nr,"first_response_hours"]=p.loc[nr,"first_response_hours"].abs()
ui=p.annual_income_lakh>300; log("pipeline","Income typed 100× too big",ui.sum(),"Divided by 100"); p.loc[ui,"annual_income_lakh"]/=100
mi=p.annual_income_lakh.isna().sum(); p["annual_income_lakh"]=p.annual_income_lakh.fillna(p.groupby("segment").annual_income_lakh.transform("median")); log("pipeline","Missing income",mi,"Median by segment")
mr_=p.first_response_hours.isna()&p.contacted_date.notna(); p.loc[mr_,"first_response_hours"]=p[mr_].lead_source.map(p.groupby("lead_source").first_response_hours.median()); log("pipeline","Missing response time for contacted leads",mr_.sum(),"Median by lead source")
q1,q3=p.acquisition_cost_inr.quantile([.25,.75]); cap=q3+3*(q3-q1); co=p.acquisition_cost_inr>cap
log("pipeline",f"Acquisition-cost outliers above ₹{cap:,.0f} (Q3 + 3×IQR)",co.sum(),"Capped at the threshold (winsorised), flagged"); p["cost_outlier_flag"]=co.astype(int); p.loc[co,"acquisition_cost_inr"]=cap
# ---- integration with the survey (record-level link for survey opt-in leads) ----
lk = p.survey_respondent_id.notna(); valid = p.survey_respondent_id.isin(s.respondent_id)
log("pipeline","Leads linked to a survey respondent (survey opt-in)",lk.sum(),"Joined on survey_respondent_id")
log("pipeline","Linked leads whose respondent was removed in survey cleaning",(lk & ~valid).sum(),"Link kept but flagged (survey_link_valid = 0)")
p["survey_link_valid"] = np.where(lk, valid.astype(int), np.nan)
segmap = s.set_index("respondent_id").segment; newseg = p.loc[lk & valid,"survey_respondent_id"].map(segmap)
log("pipeline","Linked leads whose segment label differed from the survey's K-Means segment",(newseg != p.loc[lk & valid,"segment"]).sum(),"Segment harmonised to the survey's K-Means segment")
p.loc[lk & valid,"segment"] = newseg
mcs = ((p.outcome=="Won") & p.current_monthly_sip_inr.isna()).sum(); log("pipeline","Missing current SIP for won clients",mcs,"Left missing (outcome variable)")
# ---- transformations ----
ORDER=["Lead","Contacted","Discovery meeting","Plan presented","KYC & onboarding","Client"]
p["stage_index"]=p.stage_reached.map({s:i for i,s in enumerate(ORDER)})
for i,sname in enumerate(ORDER[1:],1): p["reached_"+sname.split()[0].lower().replace("&","")]=(p.stage_index>=i).astype(int)
p["won_flag"]=(p.outcome=="Won").astype(int)
p["days_to_contact"]=(p.first_response_hours/24).round(2)
p["days_lead_to_meeting"]=(p.meeting_date-p.created_date).dt.days
p["days_to_close"]=(p.closed_date-p.created_date).dt.days
p["created_month"]=p.created_date.dt.to_period("M").astype(str)
p["response_bucket"]=pd.cut(p.first_response_hours,[-0.01,2,24,72,1e6],labels=["≤2 h","2–24 h","1–3 days",">3 days"]).astype(str).replace("nan","Not contacted")
p["followup_bucket"]=pd.cut(p.num_followups,[-1,0,2,5,100],labels=["0","1–2","3–5","6+"]).astype(str)
p["income_band"]=pd.cut(p.annual_income_lakh,[0,5,10,20,50,1e9],labels=["<₹5L","₹5–10L","₹10–20L","₹20–50L","₹50L+"]).astype(str)
p["first_year_aum_inr"]=(p.first_sip_inr.fillna(0)*12+p.lump_sum_inr.fillna(0)).where(p.won_flag==1)
p["first_year_trail_revenue_inr"]=(p.first_year_aum_inr*0.008).round(0)   # 0.8% blended trail (assumption)
p["sip_realisation"]=(p.first_sip_inr/p.proposed_sip_inr).round(3)
log("pipeline","Derived features added",15,"stage index + reached-stage flags, won flag, response/cycle times, month, buckets, income band, first-year AUM & trail revenue, SIP realisation")
log("pipeline","Clean rows written",len(p),"data/clean/pipeline_clean.csv")
p.to_csv("data/clean/pipeline_clean.csv",index=False)
pd.DataFrame(LOG).to_csv("data/clean/cleaning_log.csv",index=False)
print(pd.DataFrame(LOG).to_string()); print(s.shape, p.shape)
