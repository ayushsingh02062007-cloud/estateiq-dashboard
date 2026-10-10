"""EstateIQ API - synthetic IoT feeds, anomaly detection, forecasting, recommendations, scenarios.
All outputs are decision-support insights from SYNTHETIC data, not official measurements."""
import zlib
from functools import lru_cache
import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.linear_model import Ridge

app = FastAPI(title="EstateIQ API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# zone = (name, kind, x, z, w, d, h)
FAC = {
 "hospital": dict(name="Government Medical College Hospital", kind="Hospital", weekend=0.95, zones=[
   ("OPD Block","clinical",-6,-4,5,4,3),("Inpatient Wards","clinical",1,-5,6,4,4),("Emergency Wing","clinical",8,-3,4,3,2),
   ("Biomedical Waste Yard","waste",-7,4,3,3,1),("Main Parking","parking",1,5,7,4,.4),("Power & Water Plant","utility",8,5,3,3,2)]),
 "college": dict(name="Engineering College Campus", kind="College", weekend=0.45, zones=[
   ("Academic Block","academic",-6,-4,6,4,3),("Hostels","residential",2,-5,5,4,4),("Central Library","academic",8,-3,3,3,2),
   ("Canteen & Waste Point","waste",-7,4,3,3,1),("Student Parking","parking",1,5,7,4,.4),("Workshops & Labs","workshop",8,5,4,3,2)]),
 "industrial": dict(name="Public-Sector Industrial Estate", kind="Industrial estate", weekend=0.7, zones=[
   ("Production Shed A","industry",-6,-4,6,4,3),("Production Shed B","industry",2,-5,6,4,3),("Admin Block","office",9,-3,3,3,2),
   ("Effluent & Scrap Yard","waste",-7,4,3,3,1),("Truck Bay & Gate","parking",1,5,7,4,.4),("Utility Block","utility",9,5,3,3,2)]),
 "corporate": dict(name="Company Tech Campus", kind="Corporate campus", weekend=0.2, zones=[
   ("Tower A","office",-6,-4,5,4,5),("Tower B","office",1,-5,5,4,4),("Cafeteria","food",8,-3,4,3,2),
   ("Waste Collection Hub","waste",-7,4,3,3,1),("Basement Parking","parking",1,5,7,4,.4),("Chiller Plant","utility",8,5,3,3,2)]),
}
KE = dict(clinical=1.3, academic=1.0, residential=.9, workshop=1.4, waste=.2, parking=.15, utility=.8, industry=2.0, office=1.1, food=.7)
HOURS = 24 * 14
GRID_KG_PER_KWH, TARIFF = 0.82, 8.0   # assumptions (India grid avg ~0.8 kgCO2/kWh; Rs/kWh illustrative)

def slug(s): return "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-")

@lru_cache(maxsize=8)
def generate(fid: str) -> pd.DataFrame:
    f = FAC[fid]; rng = np.random.default_rng(zlib.crc32(fid.encode()))
    idx = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=HOURS, freq="h")
    h, dow = idx.hour.values, idx.dayofweek.values
    day = np.clip(np.sin((h - 6) / 12 * np.pi), 0, None)
    rush = np.exp(-((h - 9) ** 2) / 6) + np.exp(-((h - 18) ** 2) / 6)
    wk = np.where(dow >= 5, f["weekend"], 1.0); rows = []
    for i, (name, kind, *_ ) in enumerate(f["zones"]):
        n = HOURS
        energy = KE[kind] * (18 + 62 * day * wk) + rng.normal(0, 2.5, n)
        water = 4 + 14 * (np.exp(-((h - 7) ** 2) / 5) + .7 * np.exp(-((h - 20) ** 2) / 8)) * wk * (1.3 if kind in ("residential", "clinical") else .8) + rng.normal(0, .8, n)
        pm25 = 38 + 22 * rush + 4 * i + rng.normal(0, 3, n) + (10 if kind in ("parking", "industry") else 0)
        park = np.clip(100 * day * wk * (0.95 if kind == "parking" else .35) + rng.normal(0, 4, n), 0, 100)
        rate = 4.6 if i == 3 else rng.uniform(1.6, 2.6); fill = np.zeros(n); cur = 10.0
        for k in range(n):
            if h[k] == 6: cur = 10.0
            cur += rate * (0.4 + 1.2 * day[k]) + rng.normal(0, .3); fill[k] = min(cur, 100)
        if i == 1: energy[-20:-10] += 38 * KE[kind]          # injected: HVAC/equipment left on overnight
        if i == 2: water[-30:] += 11                            # injected: possible leak (flat night flow)
        if i == 4: pm25[-5:] += 70                              # injected: pollution spike (idling / burning)
        rows.append(pd.DataFrame(dict(ts=idx, zone=name, energy=energy, water=water, pm25=pm25, waste=fill, parking=park)))
    return pd.concat(rows, ignore_index=True)

def robust_z(s: pd.Series) -> pd.Series:
    res = s - s.groupby(s.index.hour).transform("median")
    mad = max(res.abs().median(), 1e-6)
    return 0.6745 * res / mad

def forecast(s: pd.Series, steps=24):
    t = np.arange(len(s)); fut = pd.date_range(s.index[-1] + pd.Timedelta(hours=1), periods=steps, freq="h")
    def X(tt, ix):
        hh = ix.hour.values; return np.c_[tt / len(s), np.sin(2*np.pi*hh/24), np.cos(2*np.pi*hh/24), np.sin(4*np.pi*hh/24), np.cos(4*np.pi*hh/24), (ix.dayofweek.values >= 5)]
    m = Ridge(alpha=1.0).fit(X(t, s.index), s.values); sd = float(np.std(s.values - m.predict(X(t, s.index))))
    yh = m.predict(X(np.arange(len(s), len(s) + steps), fut))
    return [dict(ts=str(a), v=round(float(b), 1), lo=round(float(b - 1.28 * sd), 1), hi=round(float(b + 1.28 * sd), 1)) for a, b in zip(fut, yh)]

def analyse(fid: str):
    f, df = FAC[fid], generate(fid); zones, anomalies, risks = [], [], {}
    excess = dict(energy=0., water=0.)
    for (name, kind, x, z, w, d, hgt) in f["zones"]:
        zd = df[df.zone == name].set_index("ts"); latest = zd.iloc[-1]; flags = []
        for m, unit in (("energy", "kWh"), ("water", "KL"), ("pm25", "ug/m3")):
            zs = robust_z(zd[m]); recent = zs.iloc[-48:]; bad = recent[recent > 4.0]
            if len(bad):
                since = bad.index[0]
                exp = float(zd[m].groupby(zd.index.hour).median().loc[zd.index[-1].hour])
                anomalies.append(dict(zone=name, zid=slug(name), metric=m, unit=unit, since=str(since), hours=len(bad), value=round(float(latest[m]), 1),
                                      expected=round(exp, 1), z=round(float(bad.max()), 1), peak=round(float(zd[m].loc[bad.index].max()), 1)))
                flags.append(m)
            if m in excess:
                med = zd[m].groupby(zd.index.hour).transform("median"); excess[m] += float((zd[m] - med).iloc[-48:].clip(lower=0).sum()) / float(zd[m].iloc[-48:].sum() + 1e-6) / len(f["zones"])
        fill = float(latest.waste); slope = float(np.mean(np.diff(zd.waste.values[-8:])))
        eta = (95 - fill) / slope if slope > .2 and fill < 95 else (0. if fill >= 95 else None)
        if kind == "waste": risks[name] = dict(fill=round(fill), eta=None if eta is None else round(eta, 1))
        status = "alert" if flags or (kind == "waste" and eta is not None and eta < 4) else ("watch" if latest.pm25 > 60 or latest.parking > 90 else "ok")
        zones.append(dict(id=slug(name), name=name, kind=kind, x=x, z=z, w=w, d=d, h=hgt, status=status, flags=flags,
                          latest={k: round(float(latest[k]), 1) for k in ("pm25", "energy", "water", "waste", "parking")}))
    return f, df, zones, anomalies, risks, excess

@app.get("/health")
def health(): return {"ok": True}

@app.get("/api/facilities")
def facilities(): return [dict(id=k, name=v["name"], kind=v["kind"]) for k, v in FAC.items()]

@app.get("/api/dashboard")
def dashboard(facility: str = "hospital"):
    if facility not in FAC: raise HTTPException(404, "Unknown facility")
    f, df, zones, anomalies, risks, excess = analyse(facility)
    agg = df.groupby("ts").agg(energy=("energy", "sum"), pm25=("pm25", "mean"), water=("water", "sum"))
    pm_mean = float(agg.pm25.iloc[-24:].mean()); atrisk = sum(1 for r in risks.values() if r["eta"] is not None and r["eta"] < 4)
    peak_park = float(df[df.ts == df.ts.max()].parking.max())
    items = [dict(key="air", label="Air quality", score=100 - 1.5 * max(0, pm_mean - 40), note=f"24h mean PM2.5 {pm_mean:.0f} ug/m3 (India 24h standard: 60)"),
             dict(key="energy", label="Energy discipline", score=100 - 4 * min(25, 100 * excess["energy"]), note=f"{100*excess['energy']:.0f}% of 48h use above normal hourly pattern"),
             dict(key="water", label="Water stewardship", score=100 - 4 * min(25, 100 * excess["water"]), note=f"{100*excess['water']:.0f}% of 48h use above normal hourly pattern"),
             dict(key="waste", label="Waste handling", score=100 - 35 * atrisk, note=f"{atrisk} collection point(s) forecast to overflow within 4h"),
             dict(key="mobility", label="Traffic and parking", score=100 - 3 * max(0, peak_park - 85), note=f"Peak parking occupancy now {peak_park:.0f}%")]
    for it in items: it["score"] = int(max(0, min(100, round(it["score"]))))
    recs = []
    for a in anomalies:
        ex = max(a["peak"] - a["expected"], 0)
        if a["metric"] == "energy": recs.append(dict(priority="High", zone=a["zone"], roles=["admin", "ops", "sustain"], title=f"Switch off idle load at {a['zone']}",
            why=f"Energy peaked at {a['peak']} kWh vs ~{a['expected']} expected for that hour (robust z={a['z']}) across {a['hours']}h.", action="Send facilities staff to check HVAC, lighting and equipment schedules; apply an auto-shutdown timer.", impact=f"~{ex*a['hours']:.0f} kWh avoidable (~Rs {ex*a['hours']*TARIFF:,.0f})"))
        if a["metric"] == "water": recs.append(dict(priority="High", zone=a["zone"], roles=["admin", "ops"], title=f"Inspect {a['zone']} for a possible water leak",
            why=f"Flow stays at {a['value']} KL/h against ~{a['expected']} expected, including low-use night hours (z={a['z']}).", action="Isolate the line, check meters and tanks for overflow, and inspect pipes before the next peak.", impact=f"~{ex*24:.0f} KL/day at risk"))
        if a["metric"] == "pm25": recs.append(dict(priority="High", zone=a["zone"], roles=["admin", "ops", "sustain"], title=f"Pollution hotspot at {a['zone']}",
            why=f"PM2.5 reached {a['peak']} ug/m3 vs ~{a['expected']} expected (z={a['z']}). Synthetic sensor reading, not an official measurement.", action="Send an advisory, stop vehicle idling or open burning nearby, and verify with a calibrated monitor.", impact="Lower exposure for staff and visitors"))
    for n, r in risks.items():
        if r["eta"] is not None and r["eta"] < 6: recs.append(dict(priority="High" if r["eta"] < 4 else "Medium", zone=n, roles=["admin", "ops", "sustain"], title=f"Re-route a collection vehicle to {n}",
            why=f"Bins are {r['fill']}% full and filling fast; projected to pass 95% in about {max(r['eta'],0):.1f}h.", action="Add an extra pickup today and move this stop first on the route.", impact="Prevents overflow, odour and pest risk"))
    if pm_mean > 55: recs.append(dict(priority="Medium", zone="Campus", roles=["admin", "sustain"], title="Issue a low-air-quality advisory", why=f"24h mean PM2.5 is {pm_mean:.0f} ug/m3.", action="Move outdoor activity indoors and run purifiers in sensitive areas.", impact="Reduces exposure"))
    recs.sort(key=lambda r: {"High": 0, "Medium": 1}.get(r["priority"], 2))
    hist = lambda s: [dict(ts=str(a), v=round(float(b), 1)) for a, b in s.iloc[-72:].items()]
    return dict(facility=dict(id=facility, name=f["name"], kind=f["kind"]), zones=zones, anomalies=anomalies, recommendations=recs,
        scorecard=dict(overall=int(np.mean([i["score"] for i in items])), items=items, waste=risks),
        forecast=dict(energy=dict(history=hist(agg.energy), future=forecast(agg.energy), unit="kWh/h"), pm25=dict(history=hist(agg.pm25), future=forecast(agg.pm25), unit="ug/m3")),
        assumptions=["Data is synthetic and simulates hourly IoT feeds for 14 days; replace generate() with real sensors or open datasets.",
          "Anomalies: robust z-score (median/MAD) of each reading against its normal hour-of-day pattern; flagged when z > 4.",
          "Forecast: ridge regression on hour-of-day harmonics, weekend flag and trend; the band is an ~80% interval from residual spread.",
          f"Cost uses Rs {TARIFF:.0f}/kWh and emissions use {GRID_KG_PER_KWH} kgCO2/kWh. Replace with your tariff and state grid factor.",
          "Scores are rule-of-thumb indices for prioritising attention. PM2.5 reference is the Indian 24h standard (60 ug/m3). Decision support, not an official measurement."])

class Scenario(BaseModel):
    facility: str = "hospital"; setpoint_delta: float = 0; shift_pct: float = 0; collections: int = 1

@app.post("/api/simulate")
def simulate(s: Scenario):
    if s.facility not in FAC: raise HTTPException(404, "Unknown facility")
    df = generate(s.facility); month = df.groupby("ts").energy.sum().iloc[-24 * 7:].sum() / 7 * 30
    saved = 0.40 * 0.06 * min(max(s.setpoint_delta, 0), 4) * month   # assume 40% cooling share, 6% saving per degC
    flex = 0.15 * month * s.shift_pct / 100                             # assume 15% of load shiftable, Rs 2/kWh off-peak gap
    out = []
    for zname, kind, *_ in FAC[s.facility]["zones"]:
        if kind != "waste": continue
        d = np.diff(df[df.zone == zname].waste.values); rate = float(np.median(d[d > 0]))
        base, new = 10 + rate * 24, 10 + rate * 24 / max(1, s.collections)
        out.append(dict(zone=zname, peak_fill_now=round(base), peak_fill_new=round(new), overflow=bool(new > 100)))
    return dict(monthly_kwh=round(month), kwh_saved=round(saved), rupees_saved=round(saved * TARIFF + flex * 2.0),
                co2_tonnes_saved=round(saved * GRID_KG_PER_KWH / 1000, 2), waste=out,
                notes="Assumes 40% cooling share, 6% saving per degC, 15% shiftable load, Rs 2/kWh off-peak gap. Indicative only.")
