"""Narrow hard floor for requests for sexual content involving minors."""

from __future__ import annotations

import re

MINOR_REFUSAL = "未成年者の性的な内容は扱えません。"

_MINOR = re.compile(
    r"児童|未成年|子ども|子供|ロリ|ショタ|\b(?:loli|shota|underage|minors?|children|child|"
    r"js|jc|jk)\b|(?:[0-9]|1[0-7])\s*(?:歳|才|years? old\b|yo\b)",
    re.IGNORECASE,
)
_SEXUAL_INTENT = re.compile(
    r"性的(?:な|に)?(?:描写|内容|画像|行為|興奮)|エロ|ポルノ(?:を|の|画像|動画|を描|を作)|"
    r"ヌード|裸(?:の|に)|セックス|性行為|自慰|わいせつ|脱がせ|犯す|"
    r"\b(?:explicit sex|sexual content|erotic|porn(?:ography)?|nude|naked|sex scene|"
    r"masturbat\w*|undress)\b",
    re.IGNORECASE,
)
_LEGAL_CONTEXT = re.compile(r"禁止法|罰則|法律|判例|報道|ニュース|事件|被害|対策|統計|歴史|law|penalt|news|report", re.IGNORECASE)
_CONTENT_REQUEST = re.compile(r"描|作|生成|見せ|書|脱がせ|show|create|generate|write|undress", re.IGNORECASE)


def blocks_minor_sexual_request(text: str) -> bool:
    """Block clear requests for sexualized minor content, not legal/news discussion."""
    if not text or not (_MINOR.search(text) and _SEXUAL_INTENT.search(text)):
        return False
    return not (_LEGAL_CONTEXT.search(text) and not _CONTENT_REQUEST.search(text))
