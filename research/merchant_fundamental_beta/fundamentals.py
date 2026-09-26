"""
Hand-extracted fundamentals for VST / NRG / CEG / TLN, sourced from actual 10-K text via
EDGAR (fetch_10k.py -> _filings/*.txt), for merchant stress-beta orthogonalization fix #5
(untried-tasks-log item 3; docs/merchant-beta-orthogonalization-results.md).

Every number below has a citation (ticker, fiscal year, filing date, section). Where a
company does not disclose a given figure cleanly, the field is None and a note explains
the gap -- nothing here is interpolated to look complete. The one exception (documented
inline) is VST's 2023 spot-exposure carry-forward, which reuses an *adjacent real
disclosure* rather than an invented number.

Granularity: ANNUAL. None of the four companies discloses fuel mix, hedge ratios, or
regional segment capacity/revenue at quarterly resolution in a form usable for a
cross-sectional/time-series beta -- this is consistent with the task brief's expectation
("these companies typically disclose ... annually in the 10-K, not quarterly").
"""
from __future__ import annotations

# ── Per-ticker, per-fiscal-year fundamentals ────────────────────────────────────────
# iso_weights: {"ERCOT": w, "PJM_PROXY": w, "OTHER": w} -- fractions of GENERATION
#   REVENUE (VST, NRG) or GENERATION CAPACITY (CEG, TLN; revenue segment not disclosed
#   at this granularity for these two) assigned to the ISOs this pipeline's GSI actually
#   models (ERCOT, PJM) vs. everything else (CAISO, NYISO, ISO-NE, MISO's own separate
#   footprint, WECC, Canada -- no live GSI series in this codebase). PJM_PROXY is an
#   approximation: VST/NRG's "East" segment and CEG's Mid-Atlantic+Midwest segments each
#   bundle PJM together with MISO/NYISO/ISO-NE (VST/NRG) or PJM-west+MISO (CEG) -- the
#   10-Ks do not split PJM out alone within those segments. Weights sum to 1.0 per year.
#
# fuel_mix_pct: {"gas":, "coal":, "nuclear":, "renewable":, "oil":} of NET GENERATION
#   CAPACITY (MW), as disclosed. Sums to ~100 (rounding).
#
# spot_exposure_pct: 1 - (disclosed hedge % of expected generation), TIER-labeled:
#   "A" = real, company-disclosed, year-specific number.
#   "A-carry" = no number disclosed for this exact year; carried from the nearest real
#       disclosure in an adjacent year (VST 2023 only) -- flagged, not fabricated.
#   "B-policy" = a disclosed *target range*, not an audited realized figure (TLN).
#   "C-prior" = NOT disclosed anywhere found; falls back to the pre-existing
#       `pass_through` field already in grid_resilience/data/utility_node_map.py
#       (itself flagged there as "pending verification against 10-K revenue
#       disclosures" -- carried forward as-is, not newly verified by this task).

FUNDAMENTALS = {
    "VST": {
        2023: {
            "filed": "2024-02-29",
            "citation": "VST 10-K FY2023 (filed 2024-02-29), Item 1 Business -- "
                        "'Electricity Generation Operations' capacity table; Item 7A "
                        "hedging paragraph.",
            "fuel_mix_pct": {"gas": 66, "coal": 23, "nuclear": 7, "renewable": 4, "oil": 0},
            "iso_weights": {"ERCOT": 0.355, "PJM_PROXY": 0.391, "OTHER": 0.254},
            "iso_weights_note": "Revenue segments (Texas/East/West/Sunset), FY2023 10-K "
                                 "Note 21 Segment Information: Texas $3,823M, East $4,215M, "
                                 "West $914M, Sunset $1,831M (gen-only total $10,783M). "
                                 "Sunset (plants with post-2023 retirement dates, since "
                                 "folded back into Texas/East by geography in Q4 2024) is "
                                 "bucketed OTHER here for lack of a clean geographic split.",
            "spot_exposure_pct": 0.09,
            "spot_exposure_tier": "A-carry",
            "spot_exposure_note": "Not disclosed for 2023 itself in the filings pulled "
                                   "(would require the FY2022 10-K, not fetched). Carried "
                                   "from the FY2023 10-K's own forward-looking 2024/2025 "
                                   "average (see 2024/2025 rows) as the nearest real anchor.",
        },
        2024: {
            "filed": "2025-02-28",
            "citation": "VST 10-K FY2023 (filed 2024-02-29), Item 7A: 'we have hedged "
                        "approximately 91% of our expected generation volumes on average "
                        "for the two-year period 2024 through 2025 (with approximately "
                        "98% hedged for 2024 and approximately 83% hedged for 2025)' -- "
                        "forward disclosure for the 2024 calendar year. Fuel mix / segment "
                        "revenue from VST 10-K FY2024 (filed 2025-02-28), Item 1 capacity "
                        "table and Note 20 Segment Information.",
            "fuel_mix_pct": {"gas": 59, "coal": 21, "nuclear": 16, "renewable": 4, "oil": 0},
            "iso_weights": {"ERCOT": 0.452, "PJM_PROXY": 0.474, "OTHER": 0.073},
            "iso_weights_note": "Revenue segments (Texas/East/West), FY2024 10-K Note 20: "
                                 "Texas $5,394M, East $5,661M, West $877M (gen-only total "
                                 "$11,932M); Sunset segment removed in Q4-2024, folded into "
                                 "Texas/East by plant geography.",
            "spot_exposure_pct": 0.02,
            "spot_exposure_tier": "A",
        },
        2025: {
            "filed": "2026-02-27",
            "citation": "VST 10-K FY2023 (filed 2024-02-29), Item 7A (same passage as "
                        "above) for the 2025 forward hedge %; VST 10-K FY2025 (filed "
                        "2026-02-27), Item 1 capacity table and Note 21 Segment "
                        "Information for fuel mix / region.",
            "fuel_mix_pct": {"gas": 62, "coal": 20, "nuclear": 15, "renewable": 3, "oil": 0},
            "iso_weights": {"ERCOT": 0.452, "PJM_PROXY": 0.521, "OTHER": 0.027},
            "iso_weights_note": "Revenue segments (Texas/East/West), FY2025 10-K Note 21: "
                                 "Texas $5,353M, East $6,174M, West $325M (gen-only total "
                                 "$11,852M).",
            "spot_exposure_pct": 0.17,
            "spot_exposure_tier": "A",
            "spot_exposure_note": "FY2025 10-K itself only says 'substantially all of our "
                                   "expected generation volumes were hedged' entering 2024 "
                                   "and 2025 (qualitative, backward-looking) -- the 83% "
                                   "quantified figure is the FY2023 10-K's forward estimate "
                                   "for calendar 2025, the most precise number available.",
        },
    },
    "NRG": {
        2023: {
            "filed": "2024-02-23",
            "citation": "NRG 10-K FY2023 (filed 2024-02-23), Item 1 'Plant Operations' "
                        "generation-portfolio-by-fuel-and-region table; Item 1 GHG section "
                        "('less than 5% of the Company's consolidated revenues were "
                        "derived from coal-fired operating assets'); Note (segment "
                        "revenue) in Item 8.",
            "fuel_mix_pct": {"gas": 43.6, "coal": 51.3, "nuclear": 0, "renewable": 1.6, "oil": 3.5},
            "fuel_mix_note": "Capacity-based (MW): gas 5,712 / coal 6,727 / oil 455 / "
                              "solar 216 / battery 2 of 13,112 total. NRG sold its 44% "
                              "stake in South Texas Project (nuclear) on 2023-11-01 -- "
                              "zero nuclear capacity in this and all later years. Coal is "
                              "51% of CAPACITY but <5% of REVENUE every year checked -- "
                              "a capacity-vs-revenue divergence worth flagging on its own "
                              "(low coal utilization / capacity-payment-driven economics).",
            "iso_weights": {"ERCOT": 0.384, "PJM_PROXY": 0.459, "OTHER": 0.157},
            "iso_weights_note": "Revenue segments (Texas/East/West-Services-Other), "
                                 "Note 19 Segment Reporting: Texas $10,476M, East "
                                 "$12,547M, West/Services/Other $4,281M (excl. Vivint "
                                 "Smart Home $1,549M, unrelated to power generation). "
                                 "NRG's East segment text: 'within the control areas of "
                                 "PJM, NYISO, ISO-NE and MISO' -- not split further.",
            "spot_exposure_pct": None,
            "spot_exposure_tier": "C-prior",
            "spot_exposure_note": "NO clean '% of expected generation hedged' figure "
                                   "found anywhere in 3 years of NRG 10-Ks searched -- NRG "
                                   "describes hedging via VaR / mark-to-market on 'economic "
                                   "hedge' open positions, never as a single %. Falls back "
                                   "to utility_node_map.py's pre-existing pass_through=0.85 "
                                   "(itself an unverified placeholder per "
                                   "generate_executive_summary_v2.py's own disclaimer).",
        },
        2024: {
            "filed": "2025-02-26",
            "citation": "NRG 10-K FY2024 (filed 2025-02-26), Item 1 generation-portfolio "
                        "table + GHG section; Note 19 Segment Reporting.",
            "fuel_mix_pct": {"gas": 43.5, "coal": 51.4, "nuclear": 0, "renewable": 1.6, "oil": 3.5},
            "fuel_mix_note": "MW: gas 5,685 / coal 6,727 / oil 455 / solar 214 of 13,081.",
            "iso_weights": {"ERCOT": 0.406, "PJM_PROXY": 0.446, "OTHER": 0.148},
            "iso_weights_note": "Revenue: Texas $10,653M, East $11,707M, "
                                 "West/Services/Other $3,886M.",
            "spot_exposure_pct": None,
            "spot_exposure_tier": "C-prior",
        },
        2025: {
            "filed": "2026-02-24",
            "citation": "NRG 10-K FY2025 (filed 2026-02-24), Item 1 generation-portfolio "
                        "table + GHG section; Note 19 Segment Reporting.",
            "fuel_mix_pct": {"gas": 42.9, "coal": 51.7, "nuclear": 0, "renewable": 1.7, "oil": 3.7},
            "fuel_mix_note": "MW: gas 5,262 / coal 6,345 / oil 455 / solar 214 of 12,276. "
                              "Excludes the LSP Portfolio (13 GW gas, LS Power) and CPower "
                              "acquisitions, which closed 2026-01-30 -- AFTER this fiscal "
                              "year-end, not reflected in this table (roughly doubles "
                              "NRG's generation capacity; next 10-K, FY2026, will be the "
                              "first to show it).",
            "iso_weights": {"ERCOT": 0.389, "PJM_PROXY": 0.499, "OTHER": 0.112},
            "iso_weights_note": "Revenue: Texas $11,139M, East $14,263M, "
                                 "West/Services/Other $3,202M (excl. Vivint $2,144M).",
            "spot_exposure_pct": None,
            "spot_exposure_tier": "C-prior",
        },
    },
    "CEG": {
        2023: {
            "filed": "2024-02-16",
            "citation": "CEG 10-K FY2023 (filed 2024-02-16), Item 1 Business -- "
                        "generating-resources pie chart (ceg-20231231_g1.jpg, read via "
                        "vision, not text-extractable -- rendered as an image in the "
                        "filing) and 5-segment geographic table.",
            "fuel_mix_pct": {"gas": 25.6, "coal": 0, "nuclear": 66.7, "renewable": 7.7, "oil": 0},
            "fuel_mix_note": "MW (from pie-chart image, not machine-extractable HTML "
                              "text): Nuclear 22,070 / Natural gas and oil 8,461 / "
                              "Renewable (wind+hydro+solar) 2,563 of 33,094 total. CEG "
                              "owns no coal generation (10-K: 'neither own nor operate any "
                              "coal-fueled generating assets').",
            "iso_weights": {"ERCOT": 0.143, "PJM_PROXY": 0.665, "OTHER": 0.192},
            "iso_weights_note": "Capacity-based 5-segment table (Item 1): Mid-Atlantic "
                                 "10,393 MW (32%, 'eastern half of PJM'), Midwest 11,605 MW "
                                 "(35%, 'western half of PJM and ... MISO'), New York 3,093 "
                                 "MW (9%, NYISO), ERCOT 4,734 MW (14%), Other Power Regions "
                                 "3,269 MW (10%, New England/South/West/Canada). Revenue-"
                                 "based segment split not disclosed at this granularity; "
                                 "capacity used instead (flagged, differs from VST/NRG "
                                 "which use revenue).",
            "spot_exposure_pct": None,
            "spot_exposure_tier": "C-prior",
            "spot_exposure_note": "No clean %-hedged figure found -- CEG describes hedging "
                                   "qualitatively ('portfolio hedging activities are "
                                   "generally concentrated in the prompt three years'), not "
                                   "as a single disclosed %. Falls back to "
                                   "utility_node_map.py's pass_through=0.80 (unverified "
                                   "placeholder).",
        },
        2024: {
            "filed": "2025-02-18",
            "citation": "CEG 10-K FY2024 (filed 2025-02-18), Item 1 pie chart "
                        "(ceg-20241231_g1.jpg) + 5-segment table.",
            "fuel_mix_pct": {"gas": 22.2, "coal": 0, "nuclear": 69.7, "renewable": 8.1, "oil": 0},
            "fuel_mix_note": "MW: Nuclear 22,068 / Gas+oil 7,045 / Renewable 2,563 of "
                              "31,676. Includes the Sep-2024 Microsoft PPA for the Three "
                              "Mile Island / Crane Clean Energy Center restart (PJM, "
                              "expected online 2028 -- not yet in this year's capacity "
                              "count).",
            "iso_weights": {"ERCOT": 0.150, "PJM_PROXY": 0.695, "OTHER": 0.156},
            "iso_weights_note": "Mid-Atlantic 10,387 MW (33%), Midwest 11,608 MW (37%), "
                                 "New York 3,093 MW (10%), ERCOT 4,740 MW (15%), Other "
                                 "1,848 MW (5%).",
            "spot_exposure_pct": None,
            "spot_exposure_tier": "C-prior",
        },
        2025: {
            "filed": "2026-02-24",
            "citation": "CEG 10-K FY2025 (filed 2026-02-24), Item 1 pie chart "
                        "(ceg-20251231_g1.jpg) + 5-segment table.",
            "fuel_mix_pct": {"gas": 22.2, "coal": 0, "nuclear": 69.7, "renewable": 8.1, "oil": 0},
            "fuel_mix_note": "MW: Nuclear 22,069 / Gas+oil 7,046 / Renewable 2,561 of "
                              "31,676 -- essentially unchanged from FY2024. IMPORTANT: on "
                              "2026-01-07 (AFTER this fiscal year-end, before the 10-K's "
                              "2026-02-24 filing date) CEG closed its acquisition of "
                              "Calpine Corp (~23 GW net new gas/geothermal, heavy Texas/"
                              "California/Northeast footprint) -- explicitly excluded from "
                              "this 10-K ('Unless otherwise noted, information in this "
                              "Form 10-K excludes Calpine'). CEG's true current fuel mix "
                              "and ISO footprint as of this task's date (2026-09-18) are "
                              "materially different from what any filed 10-K yet shows; "
                              "the fundamentals-anchored beta built here necessarily runs "
                              "one full fiscal year behind that acquisition.",
            "iso_weights": {"ERCOT": 0.150, "PJM_PROXY": 0.695, "OTHER": 0.156},
            "iso_weights_note": "Mid-Atlantic 10,386 MW (33%), Midwest 11,606 MW (37%), "
                                 "New York 3,093 MW (10%), ERCOT 4,742 MW (15%), Other "
                                 "1,849 MW (5%). Pre-Calpine.",
            "spot_exposure_pct": None,
            "spot_exposure_tier": "C-prior",
        },
    },
    "TLN": {
        # TLN's CIK (0001622536) also carries the pre-restructuring, unrelated Talen
        # Energy's 2016 10-K (FY2015) -- excluded here as a different economic entity
        # (Chapter 11 emergence 2023; re-listed 2023-06-02 per utility_node_map.py notes).
        2024: {
            "filed": "2025-02-28",
            "citation": "TLN 10-K FY2024 (filed 2025-02-28), Item 2 Properties -- "
                        "'GENERATION FLEET AS OF DECEMBER 31, 2024' plant-level table "
                        "(MW ownership, fuel type, ISO, state); Item 1 hedging paragraph.",
            "fuel_mix_pct": {"gas": 46.0, "coal": 19.8, "nuclear": 20.9, "renewable": 0, "oil": 0},
            "fuel_mix_note": "MW ownership (plant-level table): Susquehanna nuclear 2,228; "
                              "gas (Martins Creek 1,705 [listed dual gas/oil] + Montour "
                              "1,528 + H.A.Wagner 843 [dual gas/oil] + Lower Mt. Bethel 608 "
                              "+ Camden 145 + Dartmouth[ISO-NE] 80) = 4,909; coal (Brandon "
                              "Shores 1,289 + Conemaugh 386 + Keystone 213 + Colstrip[WECC] "
                              "222) = 2,110; Brunner Island (dual 'Coal/Natural Gas') 1,429 "
                              "reported separately here as 13.4% dual-fuel, not folded into "
                              "either pure category. Total fleet 10,676 MW ownership.",
            "iso_weights": {"ERCOT": 0.0, "PJM_PROXY": 0.972, "OTHER": 0.028},
            "iso_weights_note": "PJM (MAAC/BGE/AEP zones) 10,374 MW ownership of 10,676 "
                                 "total; WECC (Colstrip) 222 MW; ISO-NE (Dartmouth) 80 MW. "
                                 "Plant-level MW-ownership table, Item 2 Properties -- the "
                                 "most precise region split of any of the four names (no "
                                 "aggregation into a bundled 'East' segment the way VST/NRG "
                                 "report it).",
            "spot_exposure_pct": 0.30,
            "spot_exposure_tier": "B-policy",
            "spot_exposure_note": "TLN states a target hedge RANGE, not a realized %: 'We "
                                   "target a hedge range of 60-80% of our expected "
                                   "generation for the prompt 12 months and ratably scale "
                                   "the hedge percentage down further out in time.' 0.30 = "
                                   "1 - midpoint(60%,80%). Identical language in both the "
                                   "FY2024 and FY2025 10-Ks -- a stated policy, not an "
                                   "audited year-specific figure like VST's.",
        },
        2025: {
            "filed": "2026-02-26",
            "citation": "TLN 10-K FY2025 (filed 2026-02-26), Item 2 Properties -- "
                        "'GENERATION FLEET AS OF DECEMBER 31, 2025' plant-level table; "
                        "Item 1 hedging paragraph (same target-range language as FY2024).",
            "fuel_mix_pct": {"gas": 61.5, "coal": 16.0, "nuclear": 17.1, "renewable": 0, "oil": 5.4},
            "fuel_mix_note": "MW ownership: Susquehanna nuclear 2,245; gas (Guernsey 1,771 "
                              "+ Martins Creek 1,710 + Montour 1,505 + Brunner Island [now "
                              "'Natural Gas/Coal', converted, plant type changed from "
                              "'Intermediate' with primary listed fuel now gas-first] 1,419 "
                              "+ Freedom 1,049 + Lower Mt. Bethel 607) = 8,061; coal "
                              "(Brandon Shores 1,273 + Conemaugh 392 + Keystone 213) = "
                              "1,878 [Colstrip 222 WECC kept separate in OTHER]; H.A.Wagner "
                              "now listed as pure 'Fuel Oil' 702. Total fleet 13,108 MW "
                              "ownership -- +22.8% vs FY2024, driven by the Freedom (1,049 "
                              "MW) and Guernsey (1,771 MW) gas acquisitions that closed "
                              "during 2025. Nuclear share diluted 20.9%->17.1% purely from "
                              "the denominator growing, not a Susquehanna change.",
            "iso_weights": {"ERCOT": 0.0, "PJM_PROXY": 0.983, "OTHER": 0.017},
            "iso_weights_note": "PJM 12,886 MW of 13,108 total; WECC (Colstrip) 222 MW. "
                                 "ISO-NE Dartmouth plant no longer in the FY2025 table "
                                 "(divested or reclassified -- not confirmed from this "
                                 "text alone).",
            "spot_exposure_pct": 0.30,
            "spot_exposure_tier": "B-policy",
        },
    },
}


def get_fundamentals(ticker: str, year: int) -> dict | None:
    """Return the fundamentals record for `ticker`'s fiscal year `year`, or the nearest
    earlier year on file if `year` isn't covered (annual step-function; a company's
    disclosed fundamentals are treated as valid until the next 10-K supersedes them)."""
    records = FUNDAMENTALS.get(ticker)
    if not records:
        return None
    available = sorted(y for y in records if y <= year)
    if not available:
        available = sorted(records)  # fall back to earliest known year
        if not available or year < available[0]:
            return records.get(available[0]) if available else None
    return records[available[-1]]


# Pre-existing, unverified placeholder from grid_resilience/data/utility_node_map.py
# ("pending verification against 10-K revenue disclosures" per
# generate_executive_summary_v2.py) -- used ONLY as the fallback for NRG/CEG's
# undisclosed spot-exposure field (spot_exposure_tier == "C-prior"), never presented as
# newly-verified data.
PASS_THROUGH_FALLBACK = {"VST": 1.00, "NRG": 0.85, "CEG": 0.80, "TLN": 0.90}


def spot_exposure(ticker: str, year: int) -> tuple[float, str]:
    """Return (spot_exposure_pct, tier_label), falling back to PASS_THROUGH_FALLBACK
    when the 10-K discloses no usable hedge/spot % for this ticker."""
    rec = get_fundamentals(ticker, year)
    if rec and rec.get("spot_exposure_pct") is not None:
        return float(rec["spot_exposure_pct"]), rec.get("spot_exposure_tier", "A")
    return PASS_THROUGH_FALLBACK.get(ticker, 1.0), "C-prior"


def iso_weights(ticker: str, year: int) -> dict:
    """Return {"ERCOT": w, "PJM_PROXY": w, "OTHER": w} for ticker/year (step-function)."""
    rec = get_fundamentals(ticker, year)
    if rec:
        return rec["iso_weights"]
    return {"ERCOT": 0.0, "PJM_PROXY": 0.0, "OTHER": 1.0}
