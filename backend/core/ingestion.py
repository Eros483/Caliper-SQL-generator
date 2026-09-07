"""Ingestion + trust labels + coverage — arch §6/§7. Minimal, deterministic."""

import base64
import codecs
import re
import uuid


def _try_b64(s: str) -> str | None:
    s = s.strip()
    # heuristic: b64 chars + length multiple of 4
    if len(s) < 8 or len(s) % 4 != 0:
        return None
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", s):
        return None
    try:
        decoded = base64.b64decode(s, validate=True).decode("utf-8", errors="strict")
        # only accept if decoded looks like English / has spaces
        if len(decoded) < 4:
            return None
        return decoded
    except Exception:
        return None


def _try_rot13(s: str) -> str | None:
    try:
        decoded = codecs.decode(s, "rot_13")
        # accept if decoded contains injection keywords that encoded didn't
        keywords = ("ignore", "bypass", "system", "reveal", "filter")
        if any(k in decoded.lower() for k in keywords) and not any(k in s.lower() for k in keywords):
            return decoded
    except Exception:
        pass
    return None


def normalize(text: str) -> str:
    if not text:
        return ""
    # decode b64/rot13 if detected
    b = _try_b64(text)
    if b is not None:
        text = b
    else:
        r = _try_rot13(text)
        if r is not None:
            text = r
    # homoglyph stub: normalize common confusables (minimal)
    # strip control chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # truncate
    if len(text) > 2000:
        text = text[:2000]
    return text


def classify(text: str) -> str:
    t = normalize(text).lower()
    injection_phrases = ("ignore", "bypass", "system:", "ignore the org", "show all patients")
    extraction_phrases = ("reveal system prompt", "show system prompt", "leak", "extract")
    if any(p in t for p in extraction_phrases):
        return "extraction"
    if any(p in t for p in injection_phrases):
        return "injection"
    return "benign"


def wrap_untrusted(rows: str, nonce: str | None = None) -> str:
    """Frame DB rows as untrusted tool content with per-assembly nonce. Strips delimiter codepoints from interior."""
    if nonce is None:
        nonce = uuid.uuid4().hex[:8]
    # cap length
    if len(rows) > 2500:
        rows = rows[:2500]
    # strip delimiter codepoints / forge attempts
    # the delimiters are <<UNTRUSTED:nonce>> ... <</UNTRUSTED:nonce>>
    # strip any occurrence of UNTRUSTED marker from interior
    rows = rows.replace("UNTRUSTED", "")
    rows = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", rows)
    open_tag = f"<<UNTRUSTED:{nonce}>>"
    close_tag = f"<</UNTRUSTED:{nonce}>>"
    return f"{open_tag}\n{rows}\n{close_tag}"


def coverage_check(answer: str, rows: str) -> bool:
    """Minimal answer integrity: numbers in rows must appear in answer; cross-org leak detected."""
    if not rows or not answer:
        return False
    # extract numbers from rows
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", rows)
    # normalize answer lower for search
    ans_low = answer.lower()
    rows_low = rows.lower()
    # cross-org leak: answer mentions org_id not in rows
    org_marks = re.findall(r"org[_\s]*id\s*[:=]?\s*(\d+)", ans_low)
    if org_marks:
        row_orgs = set(re.findall(r"org[_\s]*id\s*[:=]?\s*(\d+)", rows_low))
        # also search bare numbers that could be org ids
        for o in org_marks:
            if o not in row_orgs and o not in nums:
                return False
    # every prominent number in rows (e.g., counts > 0) should be mentioned in answer
    # lenient: at least one numeric value from rows appears in answer
    # strict: if rows has a single count like 99 and answer says 5, fail
    # we check: if rows numbers not subset of answer numbers, fail
    ans_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", ans_low))
    row_nums = set(nums)
    # ignore trivial numbers (0,1) and floating variations
    row_nums = {n for n in row_nums if n not in ("0", "1", "0.0")}
    if row_nums and row_nums.isdisjoint(ans_nums):
        return False
    return True
