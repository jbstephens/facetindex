"""Synthetic wide/sparse attribute table shaped like the problem statement.

~200 text columns: sparse boolean flags (several spellings), low-cardinality categoricals,
numerics stored as text, short free text, a few long free-text columns, some junk and
some fully-empty columns. Everything is VARCHAR, like the serving table.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

WORDS = ("alpha beta gamma delta sigma vector atlas nimbus harbor summit quartz cobalt lumen orbit "
         "pioneer ridge canyon meadow forge anvil beacon cedar falcon granite horizon iris juniper "
         "kestrel lantern marble nova onyx prairie quill raven saffron timber umber violet willow "
         "zenith aurora basalt cinder dune ember fjord glacier").split()
REGIONS = ["NA", "EMEA", "APAC", "LATAM", "ANZ"]
TIERS = ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]
INDUSTRIES = ["Software", "Fintech", "Health", "Retail", "Manufacturing", "Energy", "Media", "Education",
              "Logistics", "Agriculture", "Telecom", "Insurance", "Real Estate", "Travel", "Gaming", "Legal"]
TLDS = [".com", ".io", ".co", ".net", ".org", ".ai"]


def _mask(rng, n, fill):
    return rng.random(n) < fill


def _strings_from_choices(rng, n, choices, fill, p=None) -> pa.Array:
    idx = rng.choice(len(choices), size=n, p=p).astype(np.int32)
    arr = pa.DictionaryArray.from_arrays(pa.array(idx), pa.array(choices, pa.string())).cast(pa.string())
    return _apply_null(arr, _mask(rng, n, fill))


def _apply_null(arr: pa.Array, keep: np.ndarray) -> pa.Array:
    return pc.if_else(pa.array(keep), arr, pa.scalar(None, pa.string()))


def _zipf_p(k, s=1.1):
    w = 1.0 / np.arange(1, k + 1) ** s
    return w / w.sum()


def generate(
    rows: int = 240_000,
    columns: int = 200,
    seed: int = 7,
    long_text_columns: int = 4,
    long_text_fill: float = 0.3,
    long_text_kb: tuple[int, int] = (1, 8),
) -> pa.Table:
    rng = np.random.default_rng(seed)
    n = rows
    cols: dict[str, pa.Array] = {}
    cols["id"] = pa.array([f"org_{i:08d}" for i in rng.permutation(n)], pa.string())

    # ---- short text: name, domain, url, city, ...
    w1 = rng.choice(WORDS, n)
    w2 = rng.choice(WORDS, n)
    names = pa.array(np.char.add(np.char.add(np.char.capitalize(w1), " "), np.char.capitalize(w2)))
    domains = pa.array(np.char.add(np.char.add(w1, w2), rng.choice(TLDS, n)))
    cols["name"] = names
    cols["domain"] = _apply_null(domains, _mask(rng, n, 0.93))
    cols["website_url"] = _apply_null(pc.binary_join_element_wise(pa.scalar("https://www."), domains, ""),
                                      _mask(rng, n, 0.85))
    cols["hq_city"] = _strings_from_choices(rng, n, [w.capitalize() + " City" for w in WORDS[:30]], 0.8,
                                            _zipf_p(30, 0.8))

    # ---- categoricals
    cat_specs = [("region", REGIONS, 0.97), ("tier", TIERS, 0.9), ("industry", INDUSTRIES, 0.88),
                 ("employee_band", ["1-10", "11-50", "51-200", "201-1000", "1001-5000", "5000+"], 0.75),
                 ("funding_stage", ["Seed", "Series A", "Series B", "Series C", "Public", "Bootstrapped"], 0.5),
                 ("risk_level", ["low", "medium", "high"], 0.6), ("maturity_level", ["1", "2", "3", "4", "5"], 0.55)]
    for name, choices, fill in cat_specs:
        cols[name] = _strings_from_choices(rng, n, choices, fill, _zipf_p(len(choices), 0.7))

    # ---- flags with several spellings; sparse truth, variable fill
    spellings = [("true", "false"), ("True", "False"), ("yes", "no"), ("1", "0"), ("t", "f")]
    flag_names = ["has_mobile_app", "has_api", "has_sso", "has_free_tier", "has_enterprise_plan",
                  "has_partner_program", "has_open_roles", "has_recent_funding", "has_security_page",
                  "has_status_page", "has_blog", "has_podcast", "has_events", "has_marketplace",
                  "has_integrations_page", "has_pricing_page", "has_careers_page", "has_press_page",
                  "has_customer_logos", "has_case_studies", "has_g2_profile", "has_soc2", "has_iso27001",
                  "has_gdpr_statement", "has_accessibility_statement"]
    for i, name in enumerate(flag_names):
        rate = float(rng.uniform(0.01, 0.2))
        fill = float(rng.uniform(0.7, 1.0))
        t, f = spellings[i % len(spellings)]
        truth = rng.random(n) < rate
        arr = pc.if_else(pa.array(truth), pa.scalar(t), pa.scalar(f))
        cols[name] = _apply_null(arr, _mask(rng, n, fill))

    # ---- numerics stored as text
    def num(name, values, fill, decimals=None):
        if decimals is None:
            arr = pc.cast(pa.array(values.astype(np.int64)), pa.string())
        else:
            arr = pa.array(np.char.mod(f"%.{decimals}f", values))
        cols[name] = _apply_null(arr, _mask(rng, n, fill))

    num("headcount", np.exp(rng.normal(4.5, 1.6, n)).astype(np.int64) + 1, 0.9)
    num("revenue_usd", np.exp(rng.normal(14, 2.2, n)), 0.55)
    num("founded_year", rng.integers(1950, 2026, n), 0.8)
    num("fit_score", rng.beta(2, 3, n) * 100, 0.95, 1)
    num("intent_score", rng.beta(1.5, 4, n) * 100, 0.7, 2)
    num("growth_ratio", rng.normal(1.1, 0.4, n), 0.6, 3)
    num("open_roles_count", rng.poisson(3, n), 0.65)
    num("tech_stack_count", rng.poisson(12, n), 0.8)
    num("review_rating", rng.uniform(1, 5, n), 0.4, 1)
    num("review_count", rng.negative_binomial(2, 0.02, n), 0.4)
    # a numeric column polluted with junk strings (should still classify numeric at 98%)
    vals = pc.cast(pa.array(rng.integers(0, 1000, n)), pa.string())
    junk = rng.random(n) < 0.01
    cols["alexa_rank_legacy"] = _apply_null(pc.if_else(pa.array(junk), pa.scalar("n/a"), vals), _mask(rng, n, 0.5))

    # ---- long text (out-of-line in the serving db); built from sentence chunks for speed
    sentences = [" ".join(rng.choice(WORDS, 14)) + "." for _ in range(400)]
    kb_lo, kb_hi = long_text_kb
    for j in range(long_text_columns):
        keep = _mask(rng, n, long_text_fill)
        idx = np.flatnonzero(keep)
        texts = [None] * n
        for i in idx:
            k = int(rng.integers(kb_lo * 12, kb_hi * 12 + 1))
            texts[i] = " ".join(sentences[s] for s in rng.integers(0, len(sentences), k))
        cols[f"ai_summary_{j+1}" if j % 2 == 0 else f"fit_rationale_{(j // 2) + 1}"] = pa.array(texts, pa.string())

    # ---- fully empty / almost empty columns (the provider dropped them)
    cols["deprecated_attribute_1"] = pa.nulls(n, pa.string())
    cols["deprecated_attribute_2"] = pa.array([""] * n, pa.string())

    # ---- pad with generated attributes of mixed kinds to reach `columns`
    k = 0
    while len(cols) < columns + 1:
        kind = k % 4
        if kind == 0:
            m = int(rng.integers(3, 21))
            cols[f"attr_cat_{k:03d}"] = _strings_from_choices(
                rng, n, [f"val_{i}" for i in range(m)], float(rng.uniform(0.2, 1.0)), _zipf_p(m, 0.9))
        elif kind == 1:
            fill = float(rng.uniform(0.2, 1.0))
            if k % 8 == 1:
                num(f"attr_num_{k:03d}", rng.normal(50, 15, n), fill, 2)
            else:
                num(f"attr_num_{k:03d}", rng.integers(0, 10_000, n), fill)
        elif kind == 2:
            rate = float(rng.uniform(0.01, 0.2))
            t, f = spellings[k % len(spellings)]
            arr = pc.if_else(pa.array(rng.random(n) < rate), pa.scalar(t), pa.scalar(f))
            cols[f"attr_flag_{k:03d}"] = _apply_null(arr, _mask(rng, n, float(rng.uniform(0.5, 1.0))))
        else:
            a = rng.choice(WORDS, n)
            b = rng.choice(WORDS, n)
            arr = pa.array(np.char.add(np.char.add(a, "-"), b))
            cols[f"attr_text_{k:03d}"] = _apply_null(arr, _mask(rng, n, float(rng.uniform(0.3, 1.0))))
        k += 1

    return pa.table(cols)


def write(path: str, **kwargs) -> pa.Table:
    t = generate(**kwargs)
    pq.write_table(t, path, compression="zstd", row_group_size=65536)
    return t
