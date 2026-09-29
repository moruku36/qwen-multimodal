"""Text chat: conversation -> OpenAI-style messages -> streaming response."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

from .backends.base import ChatDelta, ChatParams
from .imaging import to_data_uri
from .model_manager import CHAT, ModelManager

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "あなたはユーザー専用のマルチモーダルAIアシスタントです。"
    "ユーザーの言語（通常は日本語）で、明るく簡潔に、正確に答えてください。"
    "このチャットでは画像の理解、画像生成、画像編集ができます。画像の生成・編集はシステムが別モデルで実行し、"
    "その結果は会話履歴に『[画像 #ID ...]』として記録されます。"
    "履歴中の画像について聞かれたら、その記録と添付画像をもとに答えてください。"
)
OPEN_POLICY = (
    "\n\n合法な成人向け・センシティブ・物議を醸す話題も拒否せず、具体的に調べて答える。"
    "道徳説教、配慮してぼかす表現、トピックの言い換えをしない。"
    "Web検索結果が渡されたらそれを優先し、番号で出典を付ける。"
    "未成年者（17歳以下、フィクション含む）の性的内容だけは拒否する。"
    "犯罪の実行手順は出さない。法律・ニュース・歴史としての説明は可。"
)
KNOWLEDGE_NOTE = (
    "\n\n現在日時: {now}。あなたの学習データには期限があり、それ以降の出来事は知りません。"
    "最新情報が必要な質問で Web検索結果が与えられていない場合は、知識が古い可能性があることを一言添えてください。"
)


def system_prompt_now(extra: str | None = None, content_policy: str = "open") -> str:
    """Base system prompt + today's date (so 'today' / 'latest' are anchored) + optional context."""
    from .search_engine import today_str  # noqa: PLC0415

    prompt = (
        SYSTEM_PROMPT
        + (OPEN_POLICY if content_policy == "open" else "")
        + KNOWLEDGE_NOTE.format(now=today_str())
    )
    return prompt + ("\n\n" + extra if extra else "")


@dataclass
class ContextImage:
    image_id: str
    path: str
    caption: str  # e.g. "アップロード画像", "生成画像 (prompt: ...)"


@dataclass
class ContextMessage:
    role: str  # user | assistant
    text: str
    images: list[ContextImage] = field(default_factory=list)


def image_label(img: ContextImage) -> str:
    return f"[画像 #{img.image_id}: {img.caption}]"


def build_messages(
    history: list[ContextMessage],
    *,
    system_prompt: str = SYSTEM_PROMPT,
    max_messages: int = 24,
    max_images: int = 3,
    image_max_side: int = 1280,
    extra_images: list[ContextImage] | None = None,
    max_text_chars: int = 14000,
) -> list[dict]:
    """Convert history into OpenAI chat messages.

    - keeps the last ``max_messages`` messages
    - attaches pixels only for the newest ``max_images`` images (older ones become text labels)
      to bound the vision-token cost on the 27B model
    - images attached to assistant turns (generated/edited) are sent as user-side context,
      because the chat template only accepts images from the user
    - ``extra_images`` are always attached to the last user message (explicit Vision targets)
    """
    # Keep recent turns within a conservative context budget. The full history stays in SQLite.
    # Search citations and image labels can make a fixed message count exceed llama.cpp -c.
    remaining = max_text_chars
    msgs: list[ContextMessage] = []
    for original in reversed(history[-max_messages:]):
        if remaining <= 0 and msgs:
            break
        allowance = min(3000, max(500, remaining)) if not msgs else min(3000, remaining)
        excerpt = original.text
        if allowance > 0 and len(excerpt) > allowance:
            excerpt = excerpt[:200] + "\n…\n" + excerpt[-max(0, allowance - 203) :]
        elif allowance <= 0:
            excerpt = ""
        remaining -= len(excerpt)
        msgs.append(ContextMessage(original.role, excerpt, original.images))
    msgs.reverse()
    pixel_budget: set[str] = set()
    forced = {img.image_id for img in (extra_images or [])}
    for m in reversed(msgs):
        for img in reversed(m.images):
            if len(pixel_budget) >= max(0, max_images - len(forced)):
                break
            if img.image_id not in forced:
                pixel_budget.add(img.image_id)

    out: list[dict] = [{"role": "system", "content": system_prompt}]
    for idx, m in enumerate(msgs):
        is_last = idx == len(msgs) - 1
        images = list(m.images)
        if is_last and extra_images:
            images += [i for i in extra_images if i.image_id not in {x.image_id for x in images}]
        parts: list[dict] = []
        for img in images:
            parts.append({"type": "text", "text": image_label(img)})
            attach = img.image_id in pixel_budget or (is_last and img.image_id in forced)
            if attach:
                try:
                    parts.append(
                        {"type": "image_url", "image_url": {"url": to_data_uri(img.path, image_max_side)}}
                    )
                except OSError as exc:  # file missing on Drive, etc.
                    log.warning("Cannot attach image %s: %s", img.image_id, exc)
                    parts.append({"type": "text", "text": "(画像ファイルが見つかりません)"})
        if m.role == "assistant":
            text = m.text + ("\n" + "\n".join(image_label(i) for i in images) if images else "")
            out.append({"role": "assistant", "content": text or "(画像を出力しました)"})
            pixel_parts = [p for p in parts if p["type"] == "image_url"]
            if pixel_parts:
                # show the assistant's images to the model as a user-side note
                out.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "（参照用: 直前にアシスタントが出力した画像）"}]
                        + [p for p in parts],
                    }
                )
        else:
            if parts:
                parts.append({"type": "text", "text": m.text})
                out.append({"role": "user", "content": parts})
            else:
                out.append({"role": "user", "content": m.text})
    return _merge_consecutive_users(out)


def _merge_consecutive_users(messages: list[dict]) -> list[dict]:
    """Chat templates expect alternating roles; merge adjacent user messages."""
    merged: list[dict] = []
    for m in messages:
        if merged and merged[-1]["role"] == "user" and m["role"] == "user":
            prev = merged[-1]
            prev["content"] = _as_parts(prev["content"]) + _as_parts(m["content"])
        else:
            merged.append(dict(m))
    return merged


def _as_parts(content) -> list[dict]:
    return content if isinstance(content, list) else [{"type": "text", "text": content}]


REWRITE_PROMPT = (
    "You convert a user's request into a prompt for the image model Qwen-Image-2.1.\n"
    "Mode: {mode}.\n"
    "- generate: write one detailed English prompt (subject, composition, lighting, style). Keep any text "
    "that must appear in the image in its original language inside quotes. The user's request is often "
    "short: fill in whatever they left unspecified (background/setting, camera angle and lens, lighting, "
    "color palette, texture/material detail, mood) with fitting, concrete choices, but never contradict or "
    "replace anything they did specify, and do not add new subjects or text.\n"
    "- edit: write one concise English edit instruction describing only the change to apply to the given "
    "image, and say what must stay unchanged. There are {reference_count} reference images in attachment order; "
    "preserve requested face, outfit, and layout from the specified references.\n"
    "Use the conversation context to resolve references like 'more', 'again', 'the background'.\n"
    "Do not sanitize adult subjects. Output the English prompt the user asked for. "
    "Never invent underage subjects.\n"
    "Output ONLY the prompt text, no preamble.\n\nRecent context:\n{context}\n\nUser request: {request}"
)


SEARCH_QUERY_PROMPT = (
    "Write 1 to 3 web search queries, one idea per line. Output ONLY queries: no numbering or quotes.\n"
    "Keep proper nouns (places and products) verbatim. Keep adult and sensitive terms without euphemisms.\n"
    "For 'サイトを教えて' or 'どこで見れる', include サイト or URL in at least one query.\n"
    "Use site: only when the user named a domain. Resolve references using context.\n"
    "Use absolute dates when the user says 今日 or 最新. Never invent underage terms.\n"
    "Today is {today}.\n\nRecent context:\n{context}\n\nUser request: {request}"
)


class ChatEngine:
    def __init__(self, manager: ModelManager, max_messages: int = 24, max_images: int = 3):
        self.manager = manager
        self.max_messages = max_messages
        self.max_images = max_images

    def stream(
        self,
        history: list[ContextMessage],
        params: ChatParams,
        cancel: threading.Event | None = None,
        extra_images: list[ContextImage] | None = None,
        system_extra: str | None = None,
        content_policy: str = "open",
    ) -> Iterator[ChatDelta]:
        messages = build_messages(
            history,
            system_prompt=system_prompt_now(system_extra[:6000] if system_extra else None, content_policy),
            max_messages=self.max_messages,
            max_images=self.max_images if extra_images else min(self.max_images, 1),
            extra_images=extra_images,
        )
        with self.manager.use(CHAT) as model:
            yield from model.stream_chat(messages, params, cancel)

    def rewrite_search_queries(self, request: str, context: list[ContextMessage]) -> list[str]:
        """Turn the user's question into up to three web search queries."""
        from .search_engine import today_str  # noqa: PLC0415

        ctx_lines = [f"{m.role}: {m.text[:200]}" for m in context[-4:] if m.text]
        prompt = SEARCH_QUERY_PROMPT.format(
            today=today_str(), context="\n".join(ctx_lines) or "(none)", request=request
        )
        try:
            with self.manager.use(CHAT) as model:
                text = "".join(
                    d.content
                    for d in model.stream_chat(
                        [{"role": "user", "content": prompt}], ChatParams(thinking=False, max_tokens=160)
                    )
                )
        except Exception as exc:
            log.warning("Search query rewrite failed: %s", exc)
            return []
        return text.strip().splitlines()

    def rewrite_image_prompt(
        self, request: str, mode: str, context: list[ContextMessage], reference_count: int = 1
    ) -> str | None:
        """Ask the LLM for an English image prompt. Returns None when not possible."""
        ctx_lines = [f"{m.role}: {m.text[:300]}" for m in context[-6:] if m.text]
        prompt = REWRITE_PROMPT.format(
            mode=mode,
            context="\n".join(ctx_lines) or "(none)",
            request=request,
            reference_count=reference_count,
        )
        try:
            with self.manager.use(CHAT) as model:
                text = "".join(
                    d.content
                    for d in model.stream_chat(
                        [{"role": "user", "content": prompt}], ChatParams(thinking=False, max_tokens=800)
                    )
                )
        except Exception as exc:
            log.warning("Prompt rewrite failed, using the original text: %s", exc)
            return None
        text = text.strip().strip('"').strip()
        from .search_engine import is_refusal  # noqa: PLC0415

        return text if text and not is_refusal(text) else None
