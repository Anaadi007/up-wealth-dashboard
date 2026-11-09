"""
01_generate_data.py
Generates the two synthetic datasets used in this project:
  data/raw/survey_raw.csv    - market-validation survey of UP households (4,110 rows incl. noise)
  data/raw/pipeline_raw.csv  - 12-month pilot sales pipeline, 3,600 leads (+75 duplicates); 600 leads
                               come from survey respondents who consented to contact (survey_respondent_id)
All data is synthetic. Fixed random seeds make it reproducible.
"""
# ---------------- SURVEY ----------------
import numpy as np, pandas as pd
rng = np.random.default_rng(2026)
N = 4000
T1,T2,T3="Tier 1","Tier 2","Tier 3"
P = {
 "Young Digital Salaried":   dict(p=.25, age=(23,35), occ={"Salaried - Private":.72,"Professional":.16,"Self-employed":.12}, inc=(8.5,.5), tier={T1:.62,T2:.34,T3:.04}, lit=.72, risk=3.8, dig=4.4, trust=2.7, f2f=2.3, aware=3.1),
 "Traditional Business Family": dict(p=.22, age=(33,60), occ={"Business owner / Trader":.85,"Self-employed":.15}, inc=(13,.65), tier={T1:.25,T2:.55,T3:.20}, lit=.45, risk=3.0, dig=2.8, trust=2.4, f2f=4.5, aware=2.0),
 "Conservative Govt / Retiree": dict(p=.24, age=(40,72), occ={"Salaried - Govt / PSU":.6,"Retired / Pensioner":.4}, inc=(10,.35), tier={T1:.35,T2:.45,T3:.20}, lit=.5, risk=1.8, dig=2.5, trust=3.4, f2f=4.2, aware=2.4),
 "Affluent Professional":     dict(p=.12, age=(31,56), occ={"Professional":.7,"Business owner / Trader":.3}, inc=(32,.55), tier={T1:.6,T2:.38,T3:.02}, lit=.78, risk=3.6, dig=4.0, trust=3.1, f2f=3.6, aware=3.6),
 "Rural / Agri Saver":        dict(p=.17, age=(25,66), occ={"Farmer / Agriculture":.7,"Business owner / Trader":.15,"Salaried - Private":.15}, inc=(4.2,.55), tier={T2:.15,T3:.85}, lit=.3, risk=1.9, dig=1.9, trust=2.2, f2f=4.6, aware=1.4),
}
CITY={T1:["Lucknow","Noida","Ghaziabad"],T2:["Kanpur","Agra","Varanasi","Prayagraj","Meerut","Bareilly","Gorakhpur","Aligarh"],T3:["Small town","Village / Rural"]}
names=list(P); pr=np.array([P[k]["p"] for k in names]); pr/=pr.sum()
def pick(d): k=list(d); p=np.array(list(d.values()),float); return rng.choice(k,p=p/p.sum())
def lik(mu,sd=.95): return int(np.clip(np.round(rng.normal(mu,sd)),1,5))
def b(p): return int(rng.random()<np.clip(p,.01,.99))
rows=[]
for i in range(N):
    pn=rng.choice(names,p=pr); Q=P[pn]; tier=pick(Q["tier"]); occ=pick(Q["occ"])
    age=int(rng.integers(*Q["age"])); 
    if occ.startswith("Retired"): age=max(age,58)
    gender=rng.choice(["Male","Female"],p=[.6,.4] if pn=="Young Digital Salaried" else [.7,.3])
    edu=pick({"Up to 12th":.6,"Graduate":.35,"Post-graduate / Professional":.05} if pn=="Rural / Agri Saver" else {"Up to 12th":.02,"Graduate":.28,"Post-graduate / Professional":.7} if pn=="Affluent Professional" else {"Up to 12th":.15,"Graduate":.5,"Post-graduate / Professional":.35})
    married = int(age>27 and rng.random()<.85)
    dep=int(np.clip(rng.poisson(2.2 if married else .5),0,7))
    inc=float(np.clip(rng.lognormal(np.log(Q["inc"][0]*(1.15 if tier==T1 else .85 if tier==T3 else 1)),Q["inc"][1]),1.2,300))
    sav=float(np.clip(inc*rng.uniform(.3,2.5)*(1+(age-25)/30),.1,3000))
    emi=b(.45 if age<45 else .2)
    expct=float(np.clip(rng.normal(.58+.035*dep-.003*min(inc,60)+.08*emi,.08),.25,.97))
    pl=Q["lit"]+(.08 if edu.startswith("Post") else 0)
    quiz=[b(pl+d) for d in (.1,.05,0,-.05,-.12)]  # inflation, compounding, diversification, SIP, expense ratio
    lit=2*sum(quiz)
    risk=lik(Q["risk"]-(.5 if age>58 else 0)); dig=lik(Q["dig"]-(.6 if age>55 else 0))
    trust=lik(Q["trust"]); f2f=lik(Q["f2f"]); aware=lik(Q["aware"]+.12*(lit-5))
    bad=b(.18+(.12 if pn in("Traditional Business Family","Rural / Agri Saver") else 0)); 
    if bad: trust=max(1,trust-1)
    app=b(.12+.17*(dig-1))
    lang=pick({"Hindi":.3,"English":.2,"Both":.5} if tier==T1 else {"Hindi":.75,"English":.05,"Both":.2})
    g=dict(goal_retirement=b(.2+.015*(age-25)),goal_child_education=b(.75 if dep and age<50 else .1),goal_house=b(.55 if age<40 else .15),goal_wealth_creation=b(.3+.01*min(inc,50)),goal_emergency_fund=b(.35))
    h=dict(holds_FD=b(.8 if pn!="Young Digital Salaried" else .5),holds_Gold=b(.85 if pn in("Traditional Business Family","Rural / Agri Saver") else .6),
      holds_LIC_Traditional=b(.7 if pn in("Conservative Govt / Retiree","Traditional Business Family","Rural / Agri Saver") else .35),
      holds_Real_Estate=b(.75 if pn=="Traditional Business Family" else .6 if pn=="Affluent Professional" else .08 if pn=="Young Digital Salaried" else .35),
      holds_PPF_EPF=b(.85 if occ.startswith(("Salaried - Govt","Retired")) else .6 if occ.startswith("Salaried") else .25),
      holds_NPS=b(.7 if occ.startswith("Salaried - Govt") else .07),holds_Chit_Committee=b(.35 if pn in("Traditional Business Family","Rural / Agri Saver") else .05),
      holds_Health_Insurance=b(.2+.016*min(inc,40)+(.25 if occ.startswith("Salaried") else 0)))
    h["holds_Mutual_Funds"]=b(.08+.12*(dig-1)+.25*(lit-4)/10+.1*app); h["holds_Direct_Stocks"]=b(.04+.5*h["holds_Mutual_Funds"]*risk/5+.15*app)
    it=dict(int_Goal_Planning=b(.25+.15*(g["goal_child_education"] or g["goal_house"])+.05*(aware-2)),
      int_Tax_Planning=b(.12+(.45 if occ.startswith(("Salaried","Professional")) and inc>10 else .08)),
      int_Retirement_Planning=b(.12+.012*max(age-30,0)+(.2 if not h["holds_NPS"] and not h["holds_PPF_EPF"] else 0)),
      int_Insurance_Review=b(.18+.38*h["holds_LIC_Traditional"]-.1*h["holds_Health_Insurance"]),
      int_SIP_Mutual_Funds=b(.18+.38*(h["holds_FD"] and not h["holds_Mutual_Funds"] and age<45)+.08*(risk>=3)),
      int_Estate_Will_Planning=b(.04+(.48 if age>55 and h["holds_Real_Estate"] else .04)),
      int_PMS_AIF=b(.02+(.5 if inc>30 and h["holds_Direct_Stocks"] else .02)),
      int_Loan_Debt_Advisory=b(.1+(.3 if pn=="Traditional Business Family" else 0)+(.18 if emi else 0)))
    surplus=inc*1e5/12*(1-expct)
    # non-linear: saturating effect of risk, threshold of awareness
    mult=.08+.035*np.tanh(risk-2.5)+.03*(aware>=3)+.012*lit/2+.02*(trust>=4)
    will=max(0,surplus*mult*rng.lognormal(0,.45)+sav*1e5*.0004*rng.uniform(0,1))
    fee=pick({"Commission-based":.6,"Flat annual fee":.2,"% of assets":.1,"Not willing to pay":.1} if lit<6 else {"Commission-based":.35,"Flat annual fee":.3,"% of assets":.25,"Not willing to pay":.1})
    maxfee=0 if fee=="Not willing to pay" else max(0,rng.normal(1500+110*min(inc,80)+400*(trust-3),1300))
    z=(-1.15+.55*(aware-3)+.6*(trust-3)+.35*(f2f-3)+.022*min(inc,80)+.12*(lit-5)-.95*bad-.4*app*(dig>=4)
       +.2*sum(g.values())*.8+.15*(dep>0)+.45*(sum(it.values())>=3)+.3*(lang!="English" and tier!=T1)*(-1)*0+rng.normal(0,.7))
    eng="Yes" if rng.random()<1/(1+np.exp(-z)) else "No"
    rows.append(dict(respondent_id=f"R{i+1:05d}",city=rng.choice(CITY[tier]),city_tier=tier,age=age,gender=gender,married=married,education=edu,occupation=occ,dependents=dep,
      annual_income_lakh=round(inc,2),total_savings_lakh=round(sav,1),has_emi=emi,expense_pct_income=round(expct*100,1),
      lit_q1_inflation=quiz[0],lit_q2_compounding=quiz[1],lit_q3_diversification=quiz[2],lit_q4_sip=quiz[3],lit_q5_expense_ratio=quiz[4],
      risk_tolerance=risk,digital_comfort=dig,trust_in_advisors=trust,importance_face_to_face=f2f,awareness_wealth_mgmt=aware,attention_check="Agree",
      past_bad_experience=bad,uses_investing_app=app,preferred_language=lang,**g,**h,**it,preferred_fee_model=fee,max_annual_fee_inr=int(round(maxfee/100)*100),
      willing_monthly_investment_inr=int(round(will/100)*100),likely_to_engage=eng,_persona=pn))
df=pd.DataFrame(rows)
# Q55 consent to be contacted (separate RNG so all other survey values stay identical)
rng_c = np.random.default_rng(99)
pc = np.clip(.22 + .33*(df.likely_to_engage=="Yes") + .05*(df.trust_in_advisors-3), .02, .95)
df["consent_to_contact"] = np.where(rng_c.random(len(df)) < pc, "Yes", "No")
clean_truth=df.copy()
# ---------------- inject real-world mess ----------------
log={}
lik_cols=["risk_tolerance","digital_comfort","trust_in_advisors","importance_face_to_face","awareness_wealth_mgmt"]
# 1 missing values (item non-response, higher for money questions)
for c,p in [("annual_income_lakh",.07),("total_savings_lakh",.10),("willing_monthly_investment_inr",.03),("expense_pct_income",.05),("max_annual_fee_inr",.06)]+[(c,.02) for c in lik_cols]:
    m=rng.random(len(df))<p; df.loc[m,c]=np.nan
# 2 genuine HNI outliers (real but extreme)
idx=rng.choice(df.index,40,replace=False); df.loc[idx,"annual_income_lakh"]=np.round(rng.uniform(150,900,40),1); df.loc[idx,"willing_monthly_investment_inr"]=np.round(rng.uniform(1.5e5,1.2e6,40),-3); log["genuine_hni"]=40
# 3 data-entry errors
e=rng.choice(df.index,25,replace=False); df.loc[e,"age"]=rng.choice([0,4,150,199,250],25); log["age_typos"]=25
e=rng.choice(df.index,30,replace=False); df.loc[e,"annual_income_lakh"]=df.loc[e,"annual_income_lakh"]*100; log["income_unit_errors"]=30  # entered in thousands
e=rng.choice(df.index,15,replace=False); df.loc[e,"willing_monthly_investment_inr"]=-df.loc[e,"willing_monthly_investment_inr"].abs(); log["negative_invest"]=15
e=rng.choice(df.index,20,replace=False); df.loc[e,"expense_pct_income"]=rng.uniform(120,400,20).round(0); log["expense_over_100"]=20
# 4 straight-liners / failed attention check
s=rng.choice(df.index,90,replace=False); v=rng.integers(1,6,90)
for c in lik_cols: df.loc[s,c]=v
df.loc[s[:70],"attention_check"]=rng.choice(["Disagree","Neutral","Strongly disagree"],70); log["straightliners"]=90
# 5 inconsistent (retired aged under 30)
e=df[df.occupation=="Retired / Pensioner"].sample(12,random_state=1).index; df.loc[e,"age"]=rng.integers(19,28,12); log["inconsistent_retired_young"]=12
# 6 messy text labels
e=rng.choice(df.index,120,replace=False); df.loc[e,"city"]=df.loc[e,"city"].str.upper().str.strip()+" "
e=rng.choice(df.index,60,replace=False); df.loc[e,"likely_to_engage"]=rng.choice(["yes","Y","no","N"," Yes"],60)
# 7 duplicate submissions
d=df.sample(110,random_state=7).copy(); df=pd.concat([df,d]).sample(frac=1,random_state=3).reset_index(drop=True); log["duplicates"]=110
df.drop(columns="_persona").to_csv("data/raw/survey_raw.csv",index=False)
print("survey raw",df.shape, log)

# ---------------- SALES PIPELINE (12-month pilot, Jul 2025 – Jun 2026) ----------------
import numpy as np, pandas as pd
rng = np.random.default_rng(77)
NL = 3000
SRC = {  # share of leads, base win prob, acquisition cost per lead (INR)
 "Referral":            (.14, .24,  300),
 "CA / doctor partner": (.10, .19,  900),
 "Investor seminar":    (.20, .12,  650),
 "RWA / society camp":  (.14, .09,  400),
 "Website / SEO":       (.12, .06,  250),
 "Social media ads":    (.20, .035, 180),
 "Cold calling":        (.10, .02,  120),
}
SEG = {"Affluent professionals":(.15,1.55,30),"Young digital earners":(.27,.80,8.5),"Conservative govt & retirees":(.24,1.10,10),
       "Traditional business families":(.22,.90,13),"Rural & low-literacy savers":(.12,.45,4.2)}
TIERP={"Affluent professionals":[.6,.38,.02],"Young digital earners":[.62,.34,.04],"Conservative govt & retirees":[.35,.45,.20],"Traditional business families":[.25,.55,.20],"Rural & low-literacy savers":[0,.15,.85]}
CITY={"Tier 1":["Lucknow","Noida","Ghaziabad"],"Tier 2":["Kanpur","Agra","Varanasi","Prayagraj","Meerut","Bareilly","Gorakhpur","Aligarh"],"Tier 3":["Small town","Village / Rural"]}
RMS=["Aarti Verma","Rohit Singh","Imran Khan","Neha Gupta","Vikas Yadav"]; RMSKILL={"Aarti Verma":1.25,"Rohit Singh":1.0,"Imran Khan":1.1,"Neha Gupta":.95,"Vikas Yadav":.75}
STAGES=["Lead","Contacted","Discovery meeting","Plan presented","KYC & onboarding","Client"]
src_names=list(SRC); src_p=np.array([v[0] for v in SRC.values()]); src_p/=src_p.sum()
seg_names=list(SEG); seg_p=np.array([v[0] for v in SEG.values()]); seg_p/=seg_p.sum()
start=pd.Timestamp("2025-07-01"); rows=[]
for i in range(NL):
    src=rng.choice(src_names,p=src_p); seg=rng.choice(seg_names,p=seg_p)
    if src=="CA / doctor partner" and rng.random()<.5: seg="Affluent professionals"
    if src=="Social media ads" and rng.random()<.45: seg="Young digital earners"
    if src=="RWA / society camp" and rng.random()<.35: seg="Conservative govt & retirees"
    tier=["Tier 1","Tier 2","Tier 3"][rng.choice(3,p=TIERP[seg])]
    # seasonality: more leads Jan-Mar (tax season)
    m=rng.choice(12,p=np.array([.07,.07,.08,.08,.08,.08,.10,.11,.12,.07,.07,.07])/1.0)
    created=start+pd.DateOffset(months=int(m))+pd.Timedelta(days=int(rng.integers(0,28)))
    age=int(np.clip(rng.normal({"Young digital earners":29,"Affluent professionals":42,"Conservative govt & retirees":55,"Traditional business families":45,"Rural & low-literacy savers":44}[seg],7),21,75))
    inc=float(np.clip(rng.lognormal(np.log(SEG[seg][2]),.5),1.5,300))
    rm=rng.choice(RMS); mode=rng.choice(["In person","Video call","Phone"],p=[.5,.25,.25] if seg!="Young digital earners" else [.25,.5,.25])
    lang=rng.choice(["Hindi","English","Both"],p=[.55,.15,.30] if tier!="Tier 1" else [.25,.30,.45])
    resp_h=float(np.round(np.clip(rng.lognormal(np.log(10),1.0),.2,240),1))
    fu=int(np.clip(rng.poisson(3.2),0,12))
    # win probability
    p=SRC[src][1]*SEG[seg][1]*RMSKILL[rm]
    p*=1.35 if resp_h<=2 else 1.1 if resp_h<=24 else .7 if resp_h<=72 else .45
    p*= {0:.35,1:.6,2:.85,3:1.1,4:1.2,5:1.15,6:1.0}.get(fu,.8)
    p*=1.25 if mode=="In person" else .95 if mode=="Video call" else .75
    p*=1.0 if tier!="Tier 3" else .7
    p=min(p,.85)
    # stage progression: convert overall p into per-stage pass probs
    won = rng.random()<p
    if won: reached=5
    else:
        # distribute losses across stages, earlier drop more common for weak sources
        w=np.array([.30,.30,.22,.13,.05]) if SRC[src][1]<.08 else np.array([.18,.27,.28,.20,.07])
        reached=int(rng.choice(5,p=w/w.sum()))
    d_contact=created+pd.Timedelta(hours=resp_h) if reached>=1 else pd.NaT
    gap=lambda mu: pd.Timedelta(days=float(np.clip(rng.gamma(2,mu/2),1,120)))
    slow=1.5 if seg=="Traditional business families" else .7 if src=="Referral" else 1.0
    d_meet=d_contact+gap(6*slow) if reached>=2 else pd.NaT
    d_plan=d_meet+gap(8*slow) if reached>=3 else pd.NaT
    d_kyc=d_plan+gap(7*slow) if reached>=4 else pd.NaT
    d_close=d_kyc+gap(5*slow) if reached>=5 else pd.NaT
    snapshot=pd.Timestamp("2026-06-30")
    outcome="Won" if reached==5 else ("Open" if (created>snapshot-pd.Timedelta(days=45) and rng.random()<.6) else "Lost")
    lost_reason=None
    if outcome=="Lost":
        opts={"Not reachable":.2 if reached<=1 else .02,"Prefers FD / gold / LIC":.25 if seg in("Rural & low-literacy savers","Conservative govt & retirees") else .1,
              "Chose app / DIY investing":.35 if seg=="Young digital earners" else .06,"Trust concern":.3 if seg=="Traditional business families" else .12,
              "Fees too high":.18 if seg=="Affluent professionals" else .08,"Delayed decision":.15}
        k=list(opts); pr=np.array(list(opts.values())); lost_reason=rng.choice(k,p=pr/pr.sum())
    prop_sip=float(np.round(max(1000,inc*1e5/12*rng.uniform(.06,.14))/500)*500) if reached>=3 else np.nan
    act_sip=float(np.round(prop_sip*rng.uniform(.5,1.0)/500)*500) if outcome=="Won" else np.nan
    lump=float(np.round(inc*1e5*rng.uniform(0,.6)/1000)*1000) if outcome=="Won" and rng.random()<.45 else (0.0 if outcome=="Won" else np.nan)
    rows.append(dict(lead_id=f"L{i+1:05d}",created_date=created.date(),lead_source=src,city=rng.choice(CITY[tier]),city_tier=tier,segment=seg,age=age,annual_income_lakh=round(inc,1),
       assigned_rm=rm,preferred_language=lang,meeting_mode=mode if reached>=2 else None,first_response_hours=resp_h if reached>=1 else np.nan,num_followups=fu if reached>=1 else 0,
       stage_reached=STAGES[reached] if outcome!="Won" else "Client",contacted_date=d_contact.date() if pd.notna(d_contact) else None,meeting_date=d_meet.date() if pd.notna(d_meet) else None,
       plan_date=d_plan.date() if pd.notna(d_plan) else None,kyc_date=d_kyc.date() if pd.notna(d_kyc) else None,closed_date=d_close.date() if pd.notna(d_close) else None,
       outcome=outcome,lost_reason=lost_reason,proposed_sip_inr=prop_sip,first_sip_inr=act_sip,lump_sum_inr=lump,acquisition_cost_inr=float(round(SRC[src][2]*rng.uniform(.7,1.3)))))
# ---- 600 leads from survey respondents who consented to be contacted (record-level link) ----
PERSONA2SEG={"Young Digital Salaried":"Young digital earners","Traditional Business Family":"Traditional business families","Conservative Govt / Retiree":"Conservative govt & retirees","Affluent Professional":"Affluent professionals","Rural / Agri Saver":"Rural & low-literacy savers"}
rng_l = np.random.default_rng(78)
pool = clean_truth[clean_truth.consent_to_contact=="Yes"]
pick_ = pool.sample(600, random_state=5)
for j,(_,r0) in enumerate(pick_.iterrows()):
    seg=PERSONA2SEG[r0._persona]; tier=r0.city_tier; inc=float(r0.annual_income_lakh); age=int(r0.age)
    m=rng_l.choice(12,p=np.array([.07,.07,.08,.08,.08,.08,.10,.11,.12,.07,.07,.07]))
    created=start+pd.DateOffset(months=int(m))+pd.Timedelta(days=int(rng_l.integers(0,28)))
    rm=rng_l.choice(RMS); mode=rng_l.choice(["In person","Video call","Phone"],p=[.5,.25,.25]); lang=r0.preferred_language
    resp_h=float(np.round(np.clip(rng_l.lognormal(np.log(10),1.0),.2,240),1)); fu=int(np.clip(rng_l.poisson(3.2),0,12))
    p=(.07+.22*(r0.likely_to_engage=="Yes")+.02*(r0.trust_in_advisors-3)-.04*r0.past_bad_experience)*RMSKILL[rm]
    p*=1.35 if resp_h<=2 else 1.1 if resp_h<=24 else .7 if resp_h<=72 else .45
    p*={0:.35,1:.6,2:.85,3:1.1,4:1.2,5:1.15,6:1.0}.get(fu,.8); p*=1.25 if mode=="In person" else .95 if mode=="Video call" else .75
    p=float(np.clip(p,.005,.85)); won=rng_l.random()<p
    reached=5 if won else int(rng_l.choice(5,p=np.array([.15,.27,.30,.21,.07])))
    gap=lambda mu: pd.Timedelta(days=float(np.clip(rng_l.gamma(2,mu/2),1,120)))
    slow=1.5 if seg=="Traditional business families" else 1.0
    d_contact=created+pd.Timedelta(hours=resp_h) if reached>=1 else pd.NaT
    d_meet=d_contact+gap(6*slow) if reached>=2 else pd.NaT; d_plan=d_meet+gap(8*slow) if reached>=3 else pd.NaT
    d_kyc=d_plan+gap(7*slow) if reached>=4 else pd.NaT; d_close=d_kyc+gap(5*slow) if reached>=5 else pd.NaT
    outcome="Won" if reached==5 else ("Open" if (created>pd.Timestamp("2026-06-30")-pd.Timedelta(days=45) and rng_l.random()<.6) else "Lost")
    lost_reason=None
    if outcome=="Lost":
        opts={"Not reachable":.2 if reached<=1 else .02,"Prefers FD / gold / LIC":.25 if seg in("Rural & low-literacy savers","Conservative govt & retirees") else .1,"Chose app / DIY investing":.35 if seg=="Young digital earners" else .06,"Trust concern":.3 if seg=="Traditional business families" else .12,"Fees too high":.18 if seg=="Affluent professionals" else .08,"Delayed decision":.15}
        k=list(opts); pr=np.array(list(opts.values())); lost_reason=rng_l.choice(k,p=pr/pr.sum())
    stated=r0.willing_monthly_investment_inr
    prop_sip=float(np.round(max(1000,(stated if stated>0 else inc*1e5/12*.08)*rng_l.uniform(.7,1.1))/500)*500) if reached>=3 else np.nan
    act_sip=float(np.round(prop_sip*rng_l.uniform(.5,1.0)/500)*500) if outcome=="Won" else np.nan
    lump=float(np.round(inc*1e5*rng_l.uniform(0,.6)/1000)*1000) if outcome=="Won" and rng_l.random()<.45 else (0.0 if outcome=="Won" else np.nan)
    rows.append(dict(lead_id=f"L{NL+j+1:05d}",created_date=created.date(),lead_source="Survey opt-in",city=r0.city,city_tier=tier,segment=seg,age=age,annual_income_lakh=round(inc,1),
       assigned_rm=rm,preferred_language=lang,meeting_mode=mode if reached>=2 else None,first_response_hours=resp_h if reached>=1 else np.nan,num_followups=fu if reached>=1 else 0,
       stage_reached=STAGES[reached] if outcome!="Won" else "Client",contacted_date=d_contact.date() if pd.notna(d_contact) else None,meeting_date=d_meet.date() if pd.notna(d_meet) else None,
       plan_date=d_plan.date() if pd.notna(d_plan) else None,kyc_date=d_kyc.date() if pd.notna(d_kyc) else None,closed_date=d_close.date() if pd.notna(d_close) else None,
       outcome=outcome,lost_reason=lost_reason,proposed_sip_inr=prop_sip,first_sip_inr=act_sip,lump_sum_inr=lump,acquisition_cost_inr=float(round(150*rng_l.uniform(.8,1.2))),survey_respondent_id=r0.respondent_id))
P=pd.DataFrame(rows)
# ---- post-sale retention for won clients (snapshot 30 Jun 2026) ----
rng_r=np.random.default_rng(79); snap=pd.Timestamp("2026-06-30")
HZ={"Young digital earners":.015,"Affluent professionals":.008,"Conservative govt & retirees":.006,"Traditional business families":.012,"Rural & low-literacy savers":.02}
mo=((snap-pd.to_datetime(P.closed_date)).dt.days/30.44).round(1)
P["months_since_onboarding"]=mo.where(P.outcome=="Won")
h=P.segment.map(HZ)+.01*((P.first_sip_inr/P.proposed_sip_inr)<.7)
ch=(rng_r.random(len(P))<1-(1-h)**P.months_since_onboarding.fillna(0))&(P.outcome=="Won")
P["churned_flag"]=np.where(P.outcome=="Won",ch.astype(int),np.nan)
P["current_monthly_sip_inr"]=np.where(P.outcome!="Won",np.nan,np.where(ch,0.0,np.round(P.first_sip_inr*(1+rng_r.uniform(0,.15,len(P)))/500)*500))
# ---- inject mess ----
plog={}
m=P[P.lead_source!="Survey opt-in"].sample(frac=.04,random_state=1).index; P.loc[m,"lead_source"]=None; plog["missing_source"]=len(m)
m=P[P.outcome=="Won"].sample(frac=.03,random_state=13).index; P.loc[m,"current_monthly_sip_inr"]=np.nan; plog["missing_current_sip"]=len(m)
m=P.sample(frac=.05,random_state=2).index; P.loc[m,"annual_income_lakh"]=np.nan
m=P[P.first_response_hours.notna()].sample(frac=.03,random_state=3).index; P.loc[m,"first_response_hours"]=np.nan
m=P.sample(140,random_state=4).index; P.loc[m,"stage_reached"]=P.loc[m,"stage_reached"].map(lambda s:{"Client":rng.choice(["client","CLIENT ","Won"]),"Lead":"lead","Contacted":"contacted ","Discovery meeting":"Discovery Mtg","Plan presented":"Proposal sent","KYC & onboarding":"KYC"}[s])
m=P.sample(80,random_state=5).index; P.loc[m,"assigned_rm"]=P.loc[m,"assigned_rm"].str.upper()
m=P[P.closed_date.notna()].sample(18,random_state=6).index; P.loc[m,"closed_date"]=[pd.Timestamp(d)-pd.Timedelta(days=400) for d in P.loc[m,"created_date"]]; P.loc[m,"closed_date"]=P.loc[m,"closed_date"].map(lambda d:d.date()); plog["close_before_create"]=18
m=P[P.first_sip_inr.notna()].sample(10,random_state=7).index; P.loc[m,"first_sip_inr"]*=-1; plog["negative_sip"]=10
m=P.sample(12,random_state=8).index; P.loc[m,"acquisition_cost_inr"]=P.loc[m,"acquisition_cost_inr"]*rng.choice([25,40,60],12); plog["cost_outliers"]=12
m=P.sample(20,random_state=9).index; P.loc[m,"annual_income_lakh"]=P.loc[m,"annual_income_lakh"]*100; plog["income_unit_errors"]=20
m=P.sample(15,random_state=10).index; P.loc[m,"first_response_hours"]=-P.loc[m,"first_response_hours"].abs(); plog["negative_response_time"]=15
dup=P.sample(75,random_state=11); P=pd.concat([P,dup]).sample(frac=1,random_state=12).reset_index(drop=True); plog["duplicates"]=75
P.to_csv("data/raw/pipeline_raw.csv",index=False)
print("pipeline raw",P.shape,plog); print(P.outcome.value_counts())
