"""Intent Router: decides Chat / Vision / Generate / Edit from the message + conversation state.

Rule-based on purpose: deterministic, instant (no GPU / model swap needed on L4), and unit
tested. Misroutes are handled by the manual mode selector in the UI (Auto / Chat / Vision /
Generate / Edit), and the decision + reason is shown to the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    CHAT = "chat"
    VISION = "vision"
    GENERATE = "generate"
    EDIT = "edit"
    RESTORE = "restore"


class Mode(str, Enum):
    AUTO = "auto"
    CHAT = "chat"
    VISION = "vision"
    GENERATE = "generate"
    EDIT = "edit"


class ImageTarget(str, Enum):
    NONE = "none"
    UPLOADED = "uploaded"  # images attached to this message
    LATEST = "latest"  # most recent image in the session
    SELECTED = "selected"  # image chosen in the lineage strip
    NTH = "nth"  # chronological image n turns before latest
    VARIATION = "variation"  # 1-based sibling in a multi-variation assistant turn
    ROOT = "root"  # original of the selected/latest lineage
    ROOT_AND_LATEST = "root_and_latest"  # original + current (compare)
    PARENT_AND_LATEST = "parent_and_latest"  # previous revision + current (compare)


@dataclass
class RouteContext:
    has_uploads: bool = False
    upload_count: int = 0
    has_session_image: bool = False
    has_selected_image: bool = False
    # user turns since the last image appeared in the conversation (0 = in the previous turn)
    turns_since_last_image: int | None = None


@dataclass
class RouteDecision:
    intent: Intent
    target: ImageTarget = ImageTarget.NONE
    reason: str = ""
    compare: bool = False
    n: int = 0
    warnings: list[str] = field(default_factory=list)
    search_appearance: bool = False


def _rx(*words: str) -> re.Pattern:
    return re.compile("|".join(words), re.IGNORECASE)


IMAGE_NOUNS = _rx(
    r"画像", r"イラスト", r"絵(?!文字)", r"写真", r"ポスター", r"ロゴ", r"アイコン", r"壁紙", r"バナー", r"サムネ",
    r"スケッチ", r"イメージ図", r"\bimage", r"\bpicture", r"\bphoto", r"illustration", r"\blogo", r"wallpaper",
    r"poster", r"\bicon", r"artwork",
)  # fmt: skip
DRAW_VERBS = _rx(
    r"描いて", r"描け", r"描く", r"描き", r"\bdraw", r"\bpaint", r"illustrate", r"sketch", r"render",
)  # fmt: skip
MAKE_VERBS = _rx(r"生成", r"作って", r"作成", r"つくって", r"出して", r"\bgenerate", r"\bcreate", r"\bmake")
# Things that are generated but are not images
TEXT_ARTIFACTS = _rx(
    r"コード", r"文章", r"テキスト", r"スクリプト", r"関数", r"SQL", r"メール", r"要約", r"JSON", r"YAML", r"正規表現",
    r"データを", r"テストデータ", r"サンプルデータ", r"表を", r"リスト", r"プログラム", r"クラス", r"README", r"手順",
    r"パスワード", r"\bcode", r"\bscript", r"\bfunction", r"\bemail", r"\bsummary", r"\bregex", r"\btable",
)  # fmt: skip
EDIT_WORDS = _rx(
    r"変更", r"変えて", r"かえて", r"編集", r"消して", r"削除", r"除去", r"取り除", r"追加", r"加えて", r"足して",
    r"増やして", r"減らして", r"明るく", r"暗く", r"濃く", r"薄く", r"鮮やか", r"背景", r"色を", r"色に", r"差し替え",
    r"置き換え", r"入れ替え", r"にして", r"風に", r"っぽく", r"らしく", r"もう少し", r"もっと", r"少し", r"修正", r"直して",
    r"大きく", r"小さく", r"塗り", r"着せ", r"かぶせ", r"元に戻", r"透明",
    r"\bedit", r"\bchange", r"\breplace", r"\bremove", r"\badd\b", r"make it", r"\bmore\b", r"\bless\b",
    r"brighter", r"darker", r"background", r"\bturn (it|the)",
)  # fmt: skip
QUESTION_WORDS = _rx(
    r"何", r"なに", r"なん", r"どこ", r"どれ", r"誰", r"だれ", r"いくつ", r"何個", r"どう", r"どんな", r"なぜ", r"教えて",
    r"説明", r"読んで", r"読み取", r"書いてある", r"写って", r"映って", r"判定", r"評価", r"分析", r"確認して", r"？", r"\?",
    r"\bwhat", r"\bwhere", r"\bwho", r"\bhow", r"\bwhy", r"describe", r"explain", r"\bread\b", r"identify",
)  # fmt: skip
IMAGE_REFERENCE = _rx(
    r"この画像", r"その画像", r"さっきの", r"今の画像", r"生成した", r"作った画像", r"編集した", r"元画像", r"元の画像",
    r"前の画像", r"画像", r"写真", r"イラスト", r"絵", r"this image", r"the image", r"that image", r"the picture",
)  # fmt: skip
COMPARE_WORDS = _rx(
    r"違い", r"比較", r"比べ", r"差分", r"変化", r"変わった", r"difference", r"compare", r"diff\b"
)
ORIGINAL_WORDS = _rx(r"元画像", r"元の画像", r"最初の", r"オリジナル", r"original", r"first")
PREVIOUS_WORDS = _rx(r"前の画像", r"ひとつ前", r"1つ前", r"一つ前", r"直前", r"previous", r"last one")
TWO_BACK = _rx(r"2個前", r"ふたつ前", r"二つ前", r"two images ago")
ONE_BACK = _rx(r"1個前", r"ひとつ前", r"一つ前", r"前の画像", r"one image ago")
VARIATION_WORDS = re.compile(
    r"([1-4])(?:枚目|番目|つ目)|バリエーション\s*([1-4])|variation\s*([1-4])", re.IGNORECASE
)
THIS_IMAGE = _rx(r"これを編集", r"これを変えて", r"edit this")
RESTORE_WORDS = _rx(
    r"元に戻して|元に戻す|オリジナルに戻して|オリジナルに戻す", r"\brevert\b", r"undo to original"
)

RECENT_IMAGE_TURNS = 3

_APPEARANCE_REQUEST = _rx(
    r"検索",
    r"調べ",
    r"ネット",
    r"公式",
    r"外見",
    r"見た目",
    r"別人にしない",
    r"違う人",
    r"look up",
    r"search the web",
    r"official appearance",
    r"different person",
)
_CHARACTER_CONTEXT = _rx(r"アニメ|漫画|マンガ|ゲーム|作品|キャラ|登場|anime|manga|game|character")
_JAPANESE_NAME = re.compile(r"[一-龯]{2,6}(?:さん|ちゃん|君)|[一-龯]{4,6}を描")
_ENGLISH_NAME = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")


# "<作品>の<キャラ名>の画像を…" / "<キャラ名>を描いて" — a named subject before a picture noun or verb
_SUBJECT_CHARS = r"[^\s、。,.!?！？をがはにでとへ]"
_SUBJECT_PATTERNS = (
    re.compile(rf"(?P<subject>{_SUBJECT_CHARS}{{2,30}}?)の(?:画像|イラスト|絵|写真|姿|ビジュアル|キャラ)"),
    re.compile(rf"(?P<subject>{_SUBJECT_CHARS}{{2,30}}?)(?:を|が)(?:描|生成|作|書|出力)"),
)
_GENERIC_SUBJECTS = frozenset(
    "風景 景色 夕焼け 夕日 朝焼け 空 海 山 森 花 猫 犬 鳥 動物 街 街並み 町 夜景 部屋 家 建物 料理 食べ物 "
    "車 人物 人 女性 男性 女の子 男の子 少女 少年 子供 赤ちゃん 背景 ロゴ アイコン 未来都市 都市 宇宙 "
    "ドラゴン ロボット ネコ イヌ ウサギ ライオン ペンギン キャラクター イラスト ポスター デザイン パターン "
    "肖像 肖像画 ポートレート 顔 全身 成人 大人 テクスチャ サイバーパンク ファンタジー アニメ マンガ 漫画".split()
)
_PICTURE_TAIL = re.compile(r"の?(?:画像|イラスト|絵|写真|姿|ビジュアル|キャラ)$")
_LATIN_PROPER = re.compile(r"(?<!^)(?<![.!?] )\b[A-Z][a-z]{2,}\b")


def _named_subject(text: str) -> bool:
    """A proper-noun-looking subject ("作品の人物名") rather than a generic scene or object."""
    for pattern in _SUBJECT_PATTERNS:
        for m in pattern.finditer(text):
            subject = _PICTURE_TAIL.sub("", m.group("subject"))
            parts = [p for p in subject.split("の") if p]
            if parts and parts[0] in _GENERIC_SUBJECTS:
                continue
            if not parts or subject in _GENERIC_SUBJECTS or parts[-1] in _GENERIC_SUBJECTS:
                continue
            # "作品の名前" (two or more parts) or a bare name of 3+ kanji/katakana/latin
            if len(parts) >= 2 or re.fullmatch(r"[一-龯ァ-ヶー・A-Za-z]{3,12}", subject):
                return True
    return bool(_LATIN_PROPER.search(text))


def wants_appearance_search(text: str) -> bool:
    """Recognize an identity-sensitive character request, not a generic drawing."""
    return bool(
        _JAPANESE_NAME.search(text)
        or _named_subject(text)
        or _ENGLISH_NAME.search(text)
        or (_CHARACTER_CONTEXT.search(text) and _APPEARANCE_REQUEST.search(text))
        or re.search(r"別人にしない|違う人にしない|different person", text, re.I)
    )


def route(text: str, ctx: RouteContext, mode: Mode | str = Mode.AUTO) -> RouteDecision:
    decision = _route(text, ctx, mode)
    if decision.intent in (Intent.GENERATE, Intent.EDIT):
        explicit = bool(_APPEARANCE_REQUEST.search(text or ""))
        decision.search_appearance = wants_appearance_search(text or "") and (
            explicit or ctx.upload_count < 2
        )
    return decision


def _route(text: str, ctx: RouteContext, mode: Mode | str = Mode.AUTO) -> RouteDecision:
    mode = Mode(mode)
    text = (text or "").strip()
    if mode is not Mode.AUTO:
        return _manual(mode, text, ctx)

    edit = bool(EDIT_WORDS.search(text))
    question = bool(QUESTION_WORDS.search(text))
    compare = bool(COMPARE_WORDS.search(text))
    wants_image = _wants_new_image(text)

    variation = VARIATION_WORDS.search(text)
    if ctx.has_session_image and variation and not question:
        number = next(int(group) for group in variation.groups() if group)
        intent = Intent.EDIT if edit or wants_image else Intent.VISION
        return RouteDecision(intent, ImageTarget.VARIATION, f"バリエーション{number}を対象", n=number)
    if ctx.has_selected_image and THIS_IMAGE.search(text):
        return RouteDecision(Intent.EDIT, ImageTarget.SELECTED, "選択画像を編集")

    if ctx.has_session_image and RESTORE_WORDS.search(text):
        remaining = RESTORE_WORDS.sub("", text).strip(" 。！!？?")
        if not remaining:
            return RouteDecision(
                Intent.RESTORE,
                ImageTarget.SELECTED if ctx.has_selected_image else ImageTarget.LATEST,
                "元画像を新しい版として復元",
            )
        if edit or wants_image:
            return RouteDecision(Intent.EDIT, ImageTarget.ROOT, "元画像を編集")

    if ctx.has_session_image and TWO_BACK.search(text) and not compare:
        return RouteDecision(
            Intent.EDIT if edit or wants_image else Intent.VISION, ImageTarget.NTH, "2個前の画像を対象", n=2
        )
    if ctx.has_session_image and ONE_BACK.search(text) and not compare:
        return RouteDecision(
            Intent.EDIT if edit or wants_image else Intent.VISION, ImageTarget.NTH, "1個前の画像を対象", n=1
        )

    if ctx.has_uploads:
        if not text:
            return RouteDecision(Intent.VISION, ImageTarget.UPLOADED, "画像のみ添付 → 画像の説明")
        if compare:
            return RouteDecision(Intent.VISION, ImageTarget.UPLOADED, "添付画像の比較", compare=True)
        if (edit or wants_image) and not _is_pure_question(text, question, edit):
            return RouteDecision(Intent.EDIT, ImageTarget.UPLOADED, "添付画像 + 編集/生成の指示")
        return RouteDecision(Intent.VISION, ImageTarget.UPLOADED, "添付画像についての質問")

    if compare and ctx.has_session_image:
        if PREVIOUS_WORDS.search(text) and not ORIGINAL_WORDS.search(text):
            return RouteDecision(
                Intent.VISION, ImageTarget.PARENT_AND_LATEST, "直前の画像と比較", compare=True
            )
        return RouteDecision(
            Intent.VISION, ImageTarget.ROOT_AND_LATEST, "元画像と現在の画像を比較", compare=True
        )

    if wants_image and not _refers_to_existing(text, ctx):
        return RouteDecision(Intent.GENERATE, ImageTarget.NONE, "画像生成の依頼")

    if (
        ctx.has_selected_image
        and edit
        and IMAGE_REFERENCE.search(text)
        and not _is_pure_question(text, question, edit)
    ):
        return RouteDecision(Intent.EDIT, ImageTarget.SELECTED, "選択画像への編集指示")

    recent = ctx.has_session_image and (
        ctx.turns_since_last_image is not None and ctx.turns_since_last_image <= RECENT_IMAGE_TURNS
    )
    if recent and edit and not _is_pure_question(text, question, edit):
        return RouteDecision(
            Intent.EDIT,
            ImageTarget.SELECTED if ctx.has_selected_image else ImageTarget.LATEST,
            "画像への追加指示",
        )
    if ctx.has_session_image and question and IMAGE_REFERENCE.search(text):
        return RouteDecision(
            Intent.VISION,
            ImageTarget.SELECTED if ctx.has_selected_image else ImageTarget.LATEST,
            "会話中の画像についての質問",
        )
    if wants_image:
        return RouteDecision(Intent.GENERATE, ImageTarget.NONE, "画像生成の依頼")
    return RouteDecision(Intent.CHAT, ImageTarget.NONE, "通常のチャット")


def _wants_new_image(text: str) -> bool:
    if DRAW_VERBS.search(text):
        return True
    if MAKE_VERBS.search(text):
        if IMAGE_NOUNS.search(text):
            return True
        # "…を生成して" with no text artifact mentioned (e.g. "未来的なデータセンターを生成して")
        return bool(
            re.search(r"生成して|生成する|generate", text, re.IGNORECASE)
        ) and not TEXT_ARTIFACTS.search(text)
    return False


def _refers_to_existing(text: str, ctx: RouteContext) -> bool:
    return ctx.has_session_image and bool(
        re.search(r"この画像|その画像|さっきの|今の画像|this image|that image", text)
    )


def _is_pure_question(text: str, question: bool, edit: bool) -> bool:
    """'背景は何色？' is a question even though it contains an edit keyword."""
    if not question:
        return False
    imperative = re.search(
        r"(して|てください|にして|して下さい|please|can you|could you)", text, re.IGNORECASE
    )
    return not imperative


def _manual(mode: Mode, text: str, ctx: RouteContext) -> RouteDecision:
    has_any = ctx.has_uploads or ctx.has_session_image
    target = (
        ImageTarget.UPLOADED
        if ctx.has_uploads
        else ImageTarget.SELECTED
        if ctx.has_selected_image
        else ImageTarget.LATEST
    )
    if mode is Mode.CHAT:
        return RouteDecision(
            Intent.CHAT, ImageTarget.UPLOADED if ctx.has_uploads else ImageTarget.NONE, "手動: Chat"
        )
    if mode is Mode.GENERATE:
        # attached images act as references for image-conditioned generation
        if ctx.has_uploads:
            return RouteDecision(Intent.EDIT, ImageTarget.UPLOADED, "手動: Generate（添付画像を参照）")
        return RouteDecision(Intent.GENERATE, ImageTarget.NONE, "手動: Generate")
    if not has_any:
        return RouteDecision(
            Intent.CHAT,
            ImageTarget.NONE,
            f"手動: {mode.value} → 画像が無いため Chat",
            warnings=["対象の画像がありません。画像を添付するか先に生成してください。"],
        )
    if mode is Mode.VISION:
        if COMPARE_WORDS.search(text) and not ctx.has_uploads:
            return RouteDecision(
                Intent.VISION, ImageTarget.ROOT_AND_LATEST, "手動: Vision（比較）", compare=True
            )
        return RouteDecision(Intent.VISION, target, "手動: Vision")
    return RouteDecision(Intent.EDIT, target, "手動: Edit")
