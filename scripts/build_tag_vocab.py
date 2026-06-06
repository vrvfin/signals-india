"""
build_tag_vocab.py  —  signals-india / Workflow A (daily research summarisation)

Builds the CONTROLLED tag vocabulary used to normalise tags coming out of the
Gemini summariser, so search across ~14k/yr document summaries stays consistent.

Outputs (write both to company_repo/_index/ on Drive):
  tag_vocabulary.csv      <- human-editable seed (add/curate tags here)
  tag_vocabulary.parquet  <- machine feed (loaded by the summariser + deep-dive)

Categories:
  CLOSED  = sector / subsector / doc_type / theme  -> Gemini MUST pick from this list
  OPEN    = company / promoter / source / temporal  -> validated/normalised at runtime
            (company resolved against the 5,525-ISIN universe; others free-text slugged)

How it is used:
  - The summariser injects the CLOSED slugs into research_doc_prompt.txt at
    [CONTROLLED_VOCABULARY]. Gemini returns slugs only.
  - A post-step normalises any stray free text via the `aliases` column
    (pipe-separated) -> canonical slug. Unmatched closed-category terms fall back
    to <category>_other and are logged for you to curate into this file later.
"""
import pandas as pd

rows = []
def add(slug, ttype, display, status="closed", aliases="", notes=""):
    rows.append(dict(tag_slug=slug, tag_type=ttype, display_name=display,
                     status=status, aliases=aliases, notes=notes))

# ---------------- SECTORS (closed) ----------------
sectors = {
 "auto":"Automobiles", "auto_ancillary":"Auto ancillaries",
 "banks_private":"Private banks|private sector banks", "banks_public":"PSU banks|public sector banks",
 "nbfc":"NBFC|non banking finance", "housing_finance":"HFC|housing finance",
 "microfinance":"MFI|microfinance", "insurance":"Insurance",
 "capital_markets":"Capital markets|broking|exchanges", "amc":"AMC|asset management|mutual fund",
 "cement":"Cement", "chemicals":"Chemicals", "specialty_chemicals":"Specialty chemicals",
 "agrochemicals":"Agrochemicals|crop protection", "fertilisers":"Fertilisers|fertilizers",
 "construction":"Construction|EPC", "infrastructure":"Infrastructure",
 "roads_highways":"Roads|highways|HAM", "defence":"Defence|defense",
 "aerospace":"Aerospace", "fmcg":"FMCG|consumer staples",
 "consumer_durables":"Consumer durables|appliances", "retail":"Retail",
 "ecommerce":"Ecommerce|e-commerce", "healthcare":"Healthcare",
 "pharma":"Pharma|pharmaceuticals", "cdmo":"CDMO|CRAMS|contract manufacturing",
 "hospitals":"Hospitals", "diagnostics":"Diagnostics|pathology labs",
 "it_services":"IT|information technology|it services|software services",
 "software_products":"Software products|SaaS", "internet_platforms":"Internet|new age|platforms",
 "telecom":"Telecom|telecommunications", "media":"Media|entertainment",
 "metals_steel":"Steel", "metals_nonferrous":"Non-ferrous|aluminium|copper|zinc",
 "mining":"Mining", "oil_gas":"Oil & gas|oil and gas|oilgas|petroleum|o&g",
 "refining":"Refining|OMC", "gas_distribution":"CGD|city gas|gas distribution",
 "power_generation":"Power generation|gencos", "power_transmission":"Power transmission|grid",
 "renewables":"Renewables|solar|wind|green energy", "ev":"EV|electric vehicles",
 "batteries":"Batteries|cells|energy storage", "realty":"Real estate|realty",
 "hotels":"Hotels|hospitality", "logistics":"Logistics|3PL|warehousing",
 "ports":"Ports", "shipping":"Shipping", "railways":"Railways|rail",
 "paper":"Paper", "sugar":"Sugar", "textiles":"Textiles|apparel",
 "footwear":"Footwear", "capital_goods":"Capital goods", "industrial_machinery":"Industrial machinery",
 "electronics_manufacturing":"EMS|electronics manufacturing", "semiconductors":"Semiconductors|chips",
 "packaging":"Packaging", "paints":"Paints", "jewellery":"Jewellery|gems",
 "agritech":"Agritech", "food_processing":"Food processing", "qsr":"QSR|quick service restaurants",
 "education":"Education|edtech", "fintech":"Fintech",
 "other_sector":"Other / unclassified sector",
}
for s,(d) in sectors.items():
    disp = d.split("|")[0]; al = d if "|" in d else ""
    add(s,"sector",disp,"closed",al)

# ---------------- SUBSECTORS (closed, a light set) ----------------
subsectors = {
 "two_wheelers":"Two-wheelers|2W", "four_wheelers":"Passenger vehicles|PV|4W",
 "commercial_vehicles":"CV|commercial vehicles|trucks", "tractors":"Tractors",
 "tyres":"Tyres|tires", "generic_pharma":"Generics|US generics",
 "api_bulk_drug":"API|bulk drugs", "branded_formulations":"Branded formulations|domestic formulations",
 "wires_cables":"Wires & cables", "transformers":"Transformers",
 "data_center_reit":"Data centre|data center REIT", "qib_smallcap":"Smallcap",
 "midcap":"Midcap", "largecap":"Largecap", "sme":"SME",
}
for s,d in subsectors.items():
    disp=d.split("|")[0]; al=d if "|" in d else ""
    add(s,"subsector",disp,"closed",al)

# ---------------- DOC TYPES (closed; aligned to research_doc_prompt classifier) ----------------
doctypes = {
 "single_company_ar":"Annual report (single company)",
 "single_company_note":"Analyst / broker note (single company)",
 "single_company_drhp":"DRHP / IPO prospectus",
 "single_company_rating":"Credit rating report",
 "single_company_policy":"Company policy / regulatory filing",
 "multi_company_seminar":"Seminar / conference (multi-company)",
 "multi_company_sector":"Sector / thematic report",
 "govt_policy":"Government policy / budget",
 "macro_report":"Macro / RBI / global markets",
 "concall":"Earnings concall transcript",
 "results":"Quarterly results filing",
 "presentation":"Investor presentation",
 "other":"Other / unclassified",
}
for s,d in doctypes.items():
    add(s,"doc_type",d,"closed")

# ---------------- THEMES (closed seed; curate over time) ----------------
themes = {
 "capex_cycle":"Capex cycle", "capacity_expansion":"Capacity expansion",
 "order_book_growth":"Order book growth", "china_plus_one":"China+1|china plus one|c+1",
 "pli_scheme":"PLI|production linked incentive|production-linked incentive",
 "import_substitution":"Import substitution", "export_growth":"Export growth",
 "premiumisation":"Premiumisation|premiumization", "rural_demand":"Rural demand",
 "urban_demand":"Urban demand", "margin_expansion":"Margin expansion",
 "margin_pressure":"Margin pressure", "deleveraging":"Deleveraging|debt reduction",
 "debt_raise":"Debt raise|fund raise", "working_capital_stress":"Working capital stress",
 "credit_growth":"Credit growth|loan growth", "asset_quality":"Asset quality|NPA|GNPA",
 "nim_trend":"NIM|net interest margin", "deposit_growth":"Deposit growth",
 "monsoon":"Monsoon", "commodity_inflation":"Commodity inflation|input cost",
 "commodity_deflation":"Commodity deflation", "rate_hike":"Rate hike",
 "rate_cut":"Rate cut", "inr_depreciation":"INR depreciation|rupee weakness",
 "gst_change":"GST change", "budget_impact":"Union budget|budget impact",
 "ev_transition":"EV transition", "renewable_energy":"Renewable energy transition",
 "data_center_demand":"Data centre demand", "defence_indigenisation":"Defence indigenisation|atmanirbhar",
 "real_estate_upcycle":"Real estate upcycle", "demand_slowdown":"Demand slowdown",
 "market_share_gain":"Market share gain", "new_product_launch":"New product launch",
 "ma_consolidation":"M&A|consolidation|acquisition", "promoter_pledge":"Promoter pledge",
 "governance_concern":"Governance concern|red flag",
}
for s,d in themes.items():
    disp=d.split("|")[0]; al=d if "|" in d else ""
    add(s,"theme",disp,"closed",al)

# ---------------- OPEN categories (resolution rules, not enumerated) ----------------
add("<company>","company","Resolved to ISIN","open","",
    "Resolve mention -> ISIN via universe/master_list (NSE+BSE+SME). Store ISIN as canonical; keep symbol+name as aliases.")
add("<promoter>","promoter","Promoter / KMP name","open","",
    "Free text -> lowercase_underscore slug. Curate recurring promoters into closed rows over time.")
add("<source>","source","Publisher / broker / house","open","",
    "Free text -> slug (e.g. kotak_securities). Prefer intake subfolder name when present.")
add("<fy_or_quarter>","temporal","FY / quarter","open","",
    "Pattern fyYY or qXfyYY (e.g. fy26, q4fy26). Derived from doc content or file date fallback.")

df = pd.DataFrame(rows, columns=["tag_slug","tag_type","display_name","status","aliases","notes"])
df.to_csv("tag_vocabulary.csv", index=False)
df.to_parquet("tag_vocabulary.parquet", index=False)

print(f"rows: {len(df)}")
print(df["tag_type"].value_counts().to_string())
print("\nsample:")
print(df.head(3).to_string(index=False))
