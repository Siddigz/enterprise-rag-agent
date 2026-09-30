"""Schema inference, drift detection and normalization onto the canonical order schema.

Each incoming batch is profiled column by column. Every column is scored against each canonical field by
combining name similarity (aliases, camelCase/snake_case tokens, fuzzy match) with a value-profile check
(does the data *look* like a date, a currency code, a status...). High-confidence matches are accepted
automatically; ambiguous ones are passed to an optional resolver (the LLM) before being accepted or left
unmapped. Two versions of a source's schema are then diffed to produce drift events.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Any

# ---------------------------------------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    kind: str
    aliases: tuple[str, ...]
    description: str


CANONICAL_FIELDS: dict[str, FieldSpec] = {
    "order_id": FieldSpec(
        "order_id", ("order_id", "order_number", "order_ref", "order_no", "so_number"), "Sales order identifier"
    ),
    "customer_id": FieldSpec(
        "customer_id", ("customer_id", "cust_id", "customer_ref", "customer", "client_id"), "Customer identifier"
    ),
    "order_date": FieldSpec(
        "date", ("order_date", "placed_at", "order_dt", "created_at", "date"), "Date the order was placed"
    ),
    "region": FieldSpec("region", ("region", "sales_region", "territory"), "Sales region"),
    "sku": FieldSpec("sku", ("sku", "item_code", "product", "product_code", "item"), "Product SKU"),
    "quantity": FieldSpec("int", ("quantity", "units", "qty"), "Units ordered"),
    "unit_price": FieldSpec("money", ("unit_price", "price", "unit_cost"), "Price per unit"),
    "amount": FieldSpec(
        "money",
        ("total_amount", "amount", "amount_cents", "line_value", "net_value", "order_total", "value"),
        "Order total (pre-tax)",
    ),
    "currency": FieldSpec("currency", ("currency", "currency_code", "ccy"), "ISO 4217 currency code"),
    "status": FieldSpec(
        "status",
        ("status", "order_status", "fulfilment_status", "fulfillment_status", "state"),
        "Order lifecycle status",
    ),
}

STATUS_SYNONYMS = {
    "pending": "pending",
    "open": "pending",
    "awaiting": "pending",
    "new": "pending",
    "placed": "pending",
    "shipped": "shipped",
    "in_transit": "shipped",
    "dispatched": "shipped",
    "delivered": "delivered",
    "completed": "delivered",
    "fulfilled": "delivered",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "void": "cancelled",
    "returned": "returned",
    "refunded": "returned",
    "rma": "returned",
}
REGION_SYNONYMS = {
    "na": "NA",
    "north america": "NA",
    "us": "NA",
    "amer": "NA",
    "emea": "EMEA",
    "europe": "EMEA",
    "eu": "EMEA",
    "apac": "APAC",
    "asia pacific": "APAC",
    "asia": "APAC",
}
CURRENCIES = {"USD", "EUR", "GBP", "AUD", "CAD", "JPY", "SGD", "CHF", "NZD", "INR"}
DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"]

# ---------------------------------------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------------------------------------


def name_tokens(name: str) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return [t for t in re.split(r"[^a-zA-Z0-9]+", s.lower()) if t]


def norm_name(name: str) -> str:
    return "_".join(name_tokens(name))


def _is_int(v: str) -> bool:
    return bool(re.fullmatch(r"-?\d+", v))


def _is_decimal(v: str) -> bool:
    return bool(re.fullmatch(r"-?\d+\.\d+", v))


def detect_date_format(values: list[str]) -> str | None:
    """Return the single strptime format all values parse with, preferring unambiguous evidence."""
    candidates = []
    for fmt in DATE_FORMATS:
        try:
            for v in values:
                datetime.strptime(v, fmt)
            candidates.append(fmt)
        except ValueError:
            continue
    if not candidates:
        return None
    # %m/%d/%Y and %d/%m/%Y both parse when every day <= 12; default to the US reading only then.
    return candidates[0]


@dataclass
class ColumnProfile:
    name: str
    kind: str  # order_id | customer_id | date | region | status | currency | sku | int | money | text
    n: int
    null_rate: float
    samples: list[str]
    date_format: str | None = None
    unit: str | None = None  # money: "cents" | "dollars"
    vocab: list[str] = field(default_factory=list)


def profile_column(name: str, raw_values: Iterable[Any]) -> ColumnProfile:
    values = ["" if v is None else str(v).strip() for v in raw_values]
    present = [v for v in values if v != ""]
    n = len(values)
    null_rate = 1 - len(present) / n if n else 1.0
    sample = present[:500]
    prof = ColumnProfile(name=name, kind="text", n=n, null_rate=round(null_rate, 4), samples=present[:5])
    if not sample:
        return prof

    def share(pred: Callable[[str], bool]) -> float:
        return sum(1 for v in sample if pred(v)) / len(sample)

    lower = {v.lower() for v in sample}
    if share(lambda v: bool(re.fullmatch(r"(SO-?)\d{1,8}", v, re.I))) > 0.95:
        prof.kind = "order_id"
    elif share(lambda v: bool(re.fullmatch(r"C-?\d{3,6}", v, re.I))) > 0.95:
        prof.kind = "customer_id"
    elif share(lambda v: bool(re.fullmatch(r"SKU-\d+", v, re.I))) > 0.95:
        prof.kind = "sku"
    elif lower <= set(STATUS_SYNONYMS):
        prof.kind = "status"
        prof.vocab = sorted(set(sample))
    elif lower <= set(REGION_SYNONYMS):
        prof.kind = "region"
        prof.vocab = sorted(set(sample))
    elif all(v.upper() in CURRENCIES for v in sample):
        prof.kind = "currency"
    elif (fmt := detect_date_format(sample)) is not None:
        prof.kind = "date"
        prof.date_format = fmt
    elif share(_is_int) == 1.0:
        ints = [int(v) for v in sample]
        # Monetary amounts exported in minor units look like large integers; names usually say so.
        if "cents" in name_tokens(name) or (statistics.median(ints) > 10_000 and max(ints) > 100_000):
            prof.kind, prof.unit = "money", "cents"
        elif all(0 <= i <= 10_000_000 for i in ints) and max(ints) <= 1000:
            prof.kind = "int"
        else:
            prof.kind = "order_id" if "order" in name_tokens(name) else "int"
    elif share(lambda v: _is_decimal(v) or _is_int(v)) == 1.0:
        prof.kind, prof.unit = "money", "dollars"
    return prof


# ---------------------------------------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------------------------------------

KIND_COMPAT: dict[tuple[str, str], float] = {
    ("int", "order_id"): 0.6,  # numeric order numbers
    ("money", "int"): 0.2,
    ("int", "money"): 0.3,
    ("text", "customer_id"): 0.3,
    ("text", "sku"): 0.3,
}


def name_score(column: str, spec: FieldSpec) -> float:
    n = norm_name(column)
    aliases = [norm_name(a) for a in spec.aliases]
    if n in aliases:
        return 1.0
    toks = set(name_tokens(column))
    best = 0.0
    for a in aliases:
        at = set(a.split("_"))
        jacc = len(toks & at) / len(toks | at)
        best = max(best, jacc * 0.9, SequenceMatcher(None, n, a).ratio() * 0.85)
    return best


def value_score(profile: ColumnProfile, spec: FieldSpec) -> float:
    if profile.kind == spec.kind:
        return 1.0
    return KIND_COMPAT.get((profile.kind, spec.kind), 0.0)


@dataclass
class ColumnMapping:
    column: str
    field: str | None
    confidence: float
    method: str  # heuristic | llm | unmapped
    kind: str
    date_format: str | None = None
    unit: str | None = None
    value_map: dict[str, str] | None = None
    candidate: str | None = None  # best guess for columns left for review

    def transform_signature(self) -> tuple:
        return (self.date_format, self.unit, tuple(sorted((self.value_map or {}).items())))


@dataclass
class SchemaMapping:
    source: str
    columns: list[str]
    mappings: dict[str, ColumnMapping]
    profiles: dict[str, ColumnProfile]

    def by_field(self) -> dict[str, ColumnMapping]:
        return {m.field: m for m in self.mappings.values() if m.field}

    def mapping_json(self) -> dict:
        return {c: asdict(m) for c, m in self.mappings.items()}

    def profile_json(self) -> dict:
        return {c: asdict(p) for c, p in self.profiles.items()}

    @classmethod
    def from_json(cls, source: str, columns: list[str], mapping: dict, profile: dict) -> SchemaMapping:
        return cls(
            source=source,
            columns=columns,
            mappings={c: ColumnMapping(**m) for c, m in mapping.items()},
            profiles={c: ColumnProfile(**p) for c, p in profile.items()},
        )


# resolver(source, column, profile, candidate_fields) -> (field or None, confidence)
Resolver = Callable[[str, str, ColumnProfile, list[str]], tuple[str | None, float]]


def _value_map(profile: ColumnProfile, field_name: str) -> dict[str, str] | None:
    if field_name == "status":
        return {v: STATUS_SYNONYMS[v.lower()] for v in profile.vocab if v.lower() in STATUS_SYNONYMS}
    if field_name == "region":
        return {v: REGION_SYNONYMS[v.lower()] for v in profile.vocab if v.lower() in REGION_SYNONYMS}
    return None


def infer_mapping(
    source: str,
    rows: list[dict[str, Any]],
    auto_accept: float = 0.75,
    min_score: float = 0.45,
    resolver: Resolver | None = None,
) -> SchemaMapping:
    columns: list[str] = []
    for r in rows:
        for c in r:
            if c not in columns:
                columns.append(c)
    profiles = {c: profile_column(c, (r.get(c) for r in rows)) for c in columns}

    scored = []
    for c in columns:
        for f, spec in CANONICAL_FIELDS.items():
            ns, vs = name_score(c, spec), value_score(profiles[c], spec)
            if vs == 0.0 and ns < 1.0:
                continue  # the data is clearly not this field
            scored.append((0.55 * ns + 0.45 * vs, c, f))
    scored.sort(reverse=True)

    taken_cols: set[str] = set()
    taken_fields: set[str] = set()
    mappings: dict[str, ColumnMapping] = {}

    def assign(c: str, f: str | None, conf: float, method: str) -> None:
        p = profiles[c]
        mappings[c] = ColumnMapping(
            column=c,
            field=f,
            confidence=round(conf, 3),
            method=method,
            kind=p.kind,
            date_format=p.date_format if f == "order_date" else None,
            unit=p.unit if f in ("amount", "unit_price") else None,
            value_map=_value_map(p, f) if f else None,
        )
        taken_cols.add(c)
        if f:
            taken_fields.add(f)

    ambiguous: list[tuple[float, str, str]] = []
    for score, c, f in scored:
        if c in taken_cols or f in taken_fields:
            continue
        if score >= auto_accept:
            assign(c, f, score, "heuristic")
        elif score >= min_score:
            ambiguous.append((score, c, f))

    for score, c, f in ambiguous:
        if c in taken_cols:
            continue
        open_fields = [x for x in CANONICAL_FIELDS if x not in taken_fields]
        if resolver is not None:
            chosen, conf = resolver(source, c, profiles[c], open_fields)
            if chosen and chosen in open_fields:
                assign(c, chosen, conf, "llm")
            elif chosen is None:
                assign(c, None, conf, "unmapped")
        else:
            # No resolver: never guess. Leave it unmapped and surface it for review.
            assign(c, None, score, "needs_review")
            mappings[c].candidate = f

    for c in columns:
        if c not in taken_cols:
            assign(c, None, 0.0, "unmapped")
    return SchemaMapping(source=source, columns=columns, mappings=mappings, profiles=profiles)


# ---------------------------------------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------------------------------------


@dataclass
class DriftChange:
    kind: str  # column_renamed | column_added | column_removed | format_changed | unit_changed | vocab_changed
    column: str
    detail: dict

    def to_dict(self) -> dict:
        return asdict(self)


def diff_schemas(old: SchemaMapping, new: SchemaMapping) -> list[DriftChange]:
    changes: list[DriftChange] = []
    old_cols, new_cols = set(old.columns), set(new.columns)
    removed = [c for c in old.columns if c not in new_cols]
    added = [c for c in new.columns if c not in old_cols]
    old_by_field = {m.field: m for m in old.mappings.values() if m.field}

    renamed_from: set[str] = set()
    for a in list(added):
        fm = new.mappings[a].field
        match = next((r for r in removed if r not in renamed_from and old.mappings[r].field == fm and fm), None)
        if match:
            renamed_from.add(match)
            added.remove(a)
            changes.append(DriftChange("column_renamed", a, {"from": match, "field": fm}))
    for r in removed:
        if r not in renamed_from:
            changes.append(DriftChange("column_removed", r, {"field": old.mappings[r].field}))
    for a in added:
        changes.append(DriftChange("column_added", a, {"field": new.mappings[a].field}))

    # Same canonical field in both versions: compare how its values must be transformed.
    for f, nm in new.by_field().items():
        om = old_by_field.get(f)
        if om is None:
            continue
        if om.date_format != nm.date_format and om.date_format and nm.date_format:
            changes.append(DriftChange("format_changed", nm.column, {"from": om.date_format, "to": nm.date_format}))
        if om.unit != nm.unit and om.unit and nm.unit:
            changes.append(DriftChange("unit_changed", nm.column, {"from": om.unit, "to": nm.unit}))
        if om.value_map and nm.value_map and set(om.value_map) != set(nm.value_map):
            new_vals = sorted(set(nm.value_map) - set(om.value_map))
            if new_vals:
                changes.append(DriftChange("vocab_changed", nm.column, {"new_values": new_vals}))
        if om.column == nm.column and om.kind != nm.kind:
            changes.append(DriftChange("type_changed", nm.column, {"from": om.kind, "to": nm.kind}))
    return changes


# ---------------------------------------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------------------------------------


def normalize_order_id(v: Any) -> str | None:
    if v is None or str(v).strip() == "":
        return None
    digits = re.sub(r"\D", "", str(v))
    return f"SO-{int(digits):06d}" if digits else None


def normalize_customer_id(v: Any) -> str | None:
    if v is None or str(v).strip() == "":
        return None
    digits = re.sub(r"\D", "", str(v))
    return f"C-{int(digits):04d}" if digits else None


def _parse_date(v: str, fmt: str | None) -> date | None:
    if not v:
        return None
    for f in ([fmt] if fmt else []) + DATE_FORMATS:
        try:
            return datetime.strptime(v, f).date()
        except ValueError:
            continue
    return None


def normalize_row(row: dict[str, Any], mapping: SchemaMapping, defaults: dict[str, Any] | None = None) -> dict:
    out: dict[str, Any] = dict.fromkeys(CANONICAL_FIELDS)
    for c, m in mapping.mappings.items():
        if not m.field or c not in row:
            continue
        raw = row[c]
        v = "" if raw is None else str(raw).strip()
        if v == "":
            continue
        f = m.field
        if f == "order_id":
            out[f] = normalize_order_id(v)
        elif f == "customer_id":
            out[f] = normalize_customer_id(v)
        elif f == "order_date":
            out[f] = _parse_date(v, m.date_format)
        elif f in ("amount", "unit_price"):
            num = float(v)
            out[f] = round(num / 100, 2) if m.unit == "cents" else round(num, 2)
        elif f == "quantity":
            out[f] = int(float(v))
        elif f in ("status", "region"):
            vm = m.value_map or {}
            syn = STATUS_SYNONYMS if f == "status" else REGION_SYNONYMS
            out[f] = vm.get(v) or syn.get(v.lower()) or v
        elif f == "currency":
            out[f] = v.upper()
        else:
            out[f] = v
    for k, v in (defaults or {}).items():
        if out.get(k) is None:
            out[k] = v
    if out["amount"] is None and out["quantity"] is not None and out["unit_price"] is not None:
        out["amount"] = round(out["quantity"] * out["unit_price"], 2)
    if out["unit_price"] is None and out["amount"] is not None and out["quantity"]:
        out["unit_price"] = round(out["amount"] / out["quantity"], 2)
    return out
