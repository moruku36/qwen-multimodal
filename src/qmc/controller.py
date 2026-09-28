"""Chat Controller: one user turn -> route -> engine -> persisted history, as a stream of events.

The controller is UI-agnostic: it yields ``Event`` objects that the Gradio UI (or a future
FastAPI / OpenAI-compatible frontend) renders. It never imports gradio.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .backends.base import Cancelled, ChatParams
from .chat_engine import ChatEngine, ContextImage, ContextMessage
from .image_engine import ImageEngine, ImageOptions
from .imaging import InvalidImageError, load_user_image
from .model_manager import CHAT, IMAGE, ModelManager
from .policy import MINOR_REFUSAL, blocks_minor_sexual_request
from .router import ImageTarget, Intent, Mode, RouteContext, RouteDecision, route
from .search_engine import (
    WebSearchEngine,
    build_search_context,
    format_sources,
    is_refusal,
    needs_web_search,
    resolve_safesearch,
    usable_search_query,
)
from .session_manager import ImageRecord, MessageRecord, SessionManager
from .vision_engine import VisionEngine

log = logging.getLogger(__name__)


@dataclass
class Event:
    kind: str  # status | route | reasoning | text | image | error | done
    data: Any = None


@dataclass
class TurnOptions:
    mode: Mode | str = Mode.AUTO
    thinking: bool = False
    image: ImageOptions = field(default_factory=ImageOptions)
    prompt_rewrite: str = "auto"  # auto | on | off
    web_search: str = "auto"  # auto | on | off
    content_policy: str | None = None  # None -> application default


_SENTINEL = object()


class ChatController:
    def __init__(
        self,
        sessions: SessionManager,
        manager: ModelManager,
        chat: ChatEngine,
        vision: VisionEngine,
        images: ImageEngine,
        *,
        max_image_side: int = 2048,
        max_upload_mb: int = 30,
        after_turn: Callable[[], None] | None = None,
        search: WebSearchEngine | None = None,
        content_policy: str = "open",
        search_safesearch: str = "auto",
    ):
        self.sessions = sessions
        self.manager = manager
        self.chat = chat
        self.vision = vision
        self.images = images
        self.max_image_side = max_image_side
        self.max_upload_mb = max_upload_mb
        self.after_turn = after_turn
        self.search = search
        self.content_policy = content_policy
        self.search_safesearch = search_safesearch
        self.cancel_event = threading.Event()
        self._status_q: queue.Queue = queue.Queue()
        manager.on_status = self._status_q.put

    # ------------------------------------------------------------------ public API
    def cancel(self) -> None:
        self.cancel_event.set()

    def handle(
        self,
        session_id: str,
        text: str,
        files: list[str] | None = None,
        options: TurnOptions | None = None,
    ) -> Iterator[Event]:
        """Process a new user message."""
        options = options or TurnOptions()
        text = (text or "").strip()
        files = [f for f in (files or []) if f]
        if not text and not files:
            yield Event("error", "メッセージが空です。テキストを入力するか画像を添付してください。")
            return

        uploads: list[tuple[Any, dict]] = []
        for f in files:
            try:
                uploads.append(load_user_image(f, self.max_image_side, self.max_upload_mb))
            except InvalidImageError as exc:
                yield Event("error", str(exc))
        if not text and not uploads:
            return

        user_msg_id = self.sessions.add_message(session_id, "user", text)
        for img, info in uploads:
            meta = {"original_size": list(info["original_size"]), "downscaled": info["downscaled"]}
            self.sessions.add_image(session_id, img, "uploaded", message_id=user_msg_id, meta=meta)
        yield from self._run_turn(session_id, user_msg_id, options)

    def regenerate(self, session_id: str, options: TurnOptions | None = None) -> Iterator[Event]:
        """Drop the last answer and answer the last user message again."""
        last = self.sessions.last_user_message(session_id)
        if last is None:
            yield Event("error", "再生成するメッセージがありません。")
            return
        later = [m for m in self.sessions.get_messages(session_id) if m.id > last.id]
        if later:
            self.sessions.delete_messages_from(session_id, later[0].id)
        yield from self._run_turn(session_id, last.id, options or TurnOptions())

    # ------------------------------------------------------------------ turn
    def _run_turn(self, session_id: str, user_msg_id: int, options: TurnOptions) -> Iterator[Event]:
        self.cancel_event.clear()
        _drain(self._status_q)
        messages = self.sessions.get_messages(session_id)
        user_msg = next(m for m in messages if m.id == user_msg_id)
        decision = route(user_msg.content, self._route_context(session_id, user_msg), options.mode)
        self.sessions.update_message(user_msg_id, intent=decision.intent.value)
        yield Event("route", decision)
        if blocks_minor_sexual_request(user_msg.content):
            self.sessions.add_message(
                session_id, "assistant", MINOR_REFUSAL,
                intent=decision.intent.value, meta={"policy": "blocked_minor"},
            )
            if self.after_turn:
                self.after_turn()
            yield Event("text", MINOR_REFUSAL)
            yield Event("done", None)
            return
        for w in decision.warnings:
            yield Event("status", w)
        try:
            if decision.intent in (Intent.CHAT, Intent.VISION):
                yield from self._answer(session_id, user_msg, decision, options)
            else:
                yield from self._image(session_id, user_msg, decision, options)
        except Cancelled:
            yield Event("status", "停止しました")
        except Exception as exc:
            log.exception("turn failed")
            msg = f"エラー: {exc}"
            self.sessions.add_message(
                session_id, "assistant", msg, intent=decision.intent.value, meta={"error": True}
            )
            yield Event("error", msg)
        finally:
            if self.after_turn:
                try:
                    self.after_turn()
                except Exception as exc:  # e.g. Drive unmounted
                    log.warning("after_turn hook failed: %s", exc)
        yield Event("done", None)

    def _route_context(self, session_id: str, user_msg: MessageRecord) -> RouteContext:
        prior = [i for i in self.sessions.session_images(session_id) if (i.message_id or 0) < user_msg.id]
        turns = None
        if prior:
            last_img_msg = prior[-1].message_id or 0
            turns = sum(
                1
                for m in self.sessions.get_messages(session_id)
                if m.role == "user" and last_img_msg < m.id < user_msg.id
            )
        return RouteContext(
            has_uploads=bool(user_msg.images), has_session_image=bool(prior), turns_since_last_image=turns
        )

    # ------------------------------------------------------------------ helpers
    def _context(self, session_id: str, upto_id: int) -> list[ContextMessage]:
        out = []
        for m in self.sessions.get_messages(session_id):
            if m.id > upto_id or m.meta.get("error"):
                continue
            imgs = [ContextImage(i.id, str(self.sessions.image_path(i)), i.caption) for i in m.images]
            out.append(ContextMessage(m.role, m.content, imgs))
        return out

    def _ctx_image(self, img: ImageRecord) -> ContextImage:
        return ContextImage(img.id, str(self.sessions.image_path(img)), img.caption)

    def _resolve_targets(
        self, session_id: str, user_msg: MessageRecord, target: ImageTarget
    ) -> list[ImageRecord]:
        if target is ImageTarget.UPLOADED:
            return list(user_msg.images)
        prior = [i for i in self.sessions.session_images(session_id) if (i.message_id or 0) < user_msg.id]
        if not prior:
            return []
        latest = prior[-1]
        if target is ImageTarget.LATEST:
            return [latest]
        lineage = self.sessions.lineage(latest.id)
        if target is ImageTarget.PARENT_AND_LATEST:
            if len(lineage) >= 2:
                return [lineage[-2], latest]
            return prior[-2:]
        # ROOT_AND_LATEST: original of this lineage; fall back to the first image of the session
        first = lineage[0] if len(lineage) >= 2 else prior[0]
        return [first, latest] if first.id != latest.id else [latest]

    def _pump(self, producer: Callable[[queue.Queue], None]) -> Iterator[Any]:
        """Run ``producer`` in a thread; yield its items and model-manager status messages."""
        q: queue.Queue = queue.Queue()
        error: list[BaseException] = []

        def target():
            try:
                producer(q)
            except BaseException as exc:
                error.append(exc)
            finally:
                q.put(_SENTINEL)

        t = threading.Thread(target=target, daemon=True)
        t.start()
        while True:
            for status in _drain(self._status_q):
                yield Event("status", status)
            try:
                item = q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                break
            yield item
        t.join()
        for status in _drain(self._status_q):
            yield Event("status", status)
        if error:
            raise error[0]

    # ------------------------------------------------------------------ chat / vision
    def _answer(
        self, session_id: str, user_msg: MessageRecord, decision: RouteDecision, options: TurnOptions
    ) -> Iterator[Event]:
        history = self._context(session_id, user_msg.id)
        policy = options.content_policy or self.content_policy
        params = ChatParams(thinking=options.thinking)
        if not self.manager.is_loaded(CHAT):
            yield Event("status", "Qwen3.8-27B をロード中…（初回・モデル切替時は1〜2分かかります）")

        if decision.intent is Intent.VISION:
            targets = [
                self._ctx_image(i) for i in self._resolve_targets(session_id, user_msg, decision.target)
            ]
            if not targets:
                raise ValueError("対象の画像が見つかりません。画像を添付してください。")
            if not user_msg.content:
                history[-1] = ContextMessage(
                    "user", "この画像について詳しく説明してください。", history[-1].images
                )

            def gen():
                return self.vision.stream(
                    history, targets, params, self.cancel_event, compare=decision.compare,
                    content_policy=policy,
                )
        else:
            system_extra, sources_md, search_meta = None, "", None
            for ev in self._web_search(session_id, user_msg, options):
                if isinstance(ev, Event):
                    yield ev
                else:
                    system_extra, sources_md, search_meta = ev

            def gen():
                return self.chat.stream(history, params, self.cancel_event, system_extra=system_extra,
                                        content_policy=policy)

        content, reasoning = [], []

        def producer(q: queue.Queue) -> None:
            for delta in gen():
                q.put(delta)

        t0 = time.time()
        cancelled = False
        try:
            for item in self._pump(producer):
                if isinstance(item, Event):
                    yield item
                    continue
                if item.reasoning:
                    reasoning.append(item.reasoning)
                    yield Event("reasoning", item.reasoning)
                if item.content:
                    content.append(item.content)
                    yield Event("text", item.content)
        except Cancelled:
            cancelled = True
        text = "".join(content).strip()
        if cancelled:
            text += "\n\n（停止しました）"
        meta = {"duration_s": round(time.time() - t0, 2), "cancelled": cancelled}
        if decision.intent is not Intent.VISION and search_meta:
            meta["web_search"] = search_meta
            if sources_md and not cancelled:
                text += sources_md
                yield Event("text", sources_md)
        self.sessions.add_message(
            session_id,
            "assistant",
            text,
            intent=decision.intent.value,
            model=getattr(self.manager.get(CHAT), "label", CHAT),
            reasoning="".join(reasoning) or None,
            meta=meta,
        )

    def _web_search(self, session_id: str, user_msg: MessageRecord, options: TurnOptions):
        """Yields status Events, then one (system_extra, sources_md, meta) tuple."""
        text = user_msg.content
        policy = options.content_policy or self.content_policy
        safesearch = resolve_safesearch(self.search_safesearch, policy)
        if not needs_web_search(text, options.web_search):
            yield (None, "", None)
            return
        if self.search is None or not self.search.available:
            if options.web_search == "on":
                yield Event("status", "⚠️ Web検索が使えません（ddgs 未インストール / APIキー未設定）")
            yield (None, "", None)
            return
        yield Event("status", "🔎 検索クエリを作成中…")
        history = self._context(session_id, user_msg.id)[:-1]
        rewritten = self.chat.rewrite_search_query(text, history)
        query = usable_search_query(text, rewritten)
        log.info("Web search query: original=%r effective=%r", text, query)
        if policy == "open" and self.search.provider_name == "tavily":
            yield Event("status", "⚠️ Tavily は成人向け検索を規約で禁じています。Brave / DDG を推奨")
        yield Event("status", f"🔎 Web検索中: {query}")
        resp = self.search.search(query, safesearch=safesearch)
        meta = {"query": query, "original": text, "rewritten": rewritten,
                "provider": resp.provider, "urls": [r.url for r in resp.results],
                "safesearch": safesearch}
        if resp.error or not resp.results:
            yield Event("status", resp.error or f"🔎 「{query}」の検索結果がありませんでした")
            meta["error"] = resp.error
            yield (f"Web検索を行いましたが「{query}」の結果は得られませんでした。"
                   "検索結果が無いことを伝え、未確認の情報を断定しないでください。", "", meta)
            return
        yield Event("status", f"📄 {len(resp.results)}件の結果を読み込みました（{resp.provider}）")
        yield (build_search_context(resp), format_sources(resp), meta)

    # ------------------------------------------------------------------ generate / edit
    def _image(
        self, session_id: str, user_msg: MessageRecord, decision: RouteDecision, options: TurnOptions
    ) -> Iterator[Event]:
        is_edit = decision.intent is Intent.EDIT
        sources = self._resolve_targets(session_id, user_msg, decision.target) if is_edit else []
        if is_edit and not sources:
            raise ValueError("編集する画像がありません。画像を添付するか、先に画像を生成してください。")
        instruction = user_msg.content or ("この画像を高品質に整えてください" if is_edit else "")
        if not instruction:
            raise ValueError("画像の内容を入力してください。")

        effective = instruction
        if self._should_rewrite(options.prompt_rewrite):
            yield Event("status", "プロンプトを最適化中…")
            rewritten = self.chat.rewrite_image_prompt(
                instruction, "edit" if is_edit else "generate", self._context(session_id, user_msg.id)[:-1]
            )
            if rewritten and not is_refusal(rewritten):
                effective = rewritten
                yield Event("status", f"画像プロンプト: {effective}")

        pil_sources = [self.sessions.load_image(s) for s in sources]
        parent_seeds = set()
        for s in sources:
            parent_seeds |= {i.seed for i in self.sessions.lineage(s.id) if i.seed is not None}
        if is_edit:
            request = self.images.build_edit_request(effective, pil_sources, options.image, parent_seeds)
        else:
            request = self.images.build_request(effective, options.image)

        if not self.manager.is_loaded(IMAGE):
            yield Event("status", "Qwen-Image-2.1 をロード中…（初回はダウンロードで時間がかかります）")

        def producer(q: queue.Queue) -> None:
            def progress(step: int, total: int) -> None:
                q.put(Event("status", f"{'編集' if is_edit else '生成'}中… {step}/{total} step"))

            q.put(self.images.run(request, progress=progress, cancel=self.cancel_event))

        t0 = time.time()
        result = None
        for item in self._pump(producer):
            if isinstance(item, Event):
                yield item
            else:
                result = item
        duration = round(time.time() - t0, 2)
        image_label = getattr(self.manager.get(IMAGE), "label", IMAGE)
        verb = "編集" if is_edit else "生成"
        summary = f"画像を{verb}しました（{result.width}×{result.height}, {request.steps} step, seed {request.seed}, {duration:.0f}s）"
        msg_id = self.sessions.add_message(
            session_id,
            "assistant",
            summary,
            intent=decision.intent.value,
            model=image_label,
            meta={"prompt": instruction, "effective_prompt": effective, "duration_s": duration},
        )
        parent = sources[-1] if sources else None
        record = self.sessions.add_image(
            session_id,
            result,
            "edited" if is_edit else "generated",
            message_id=msg_id,
            parent_id=parent.id if parent else None,
            seed=request.seed,
            meta={"prompt": instruction, "effective_prompt": effective},
        )
        self.sessions.add_generation(
            session_id,
            kind="edit" if is_edit else "generate",
            prompt=instruction,
            effective_prompt=effective,
            params=request.params_dict(),
            model=image_label,
            duration_s=duration,
            message_id=msg_id,
            image_id=record.id,
            source_image_ids=[s.id for s in sources],
        )
        yield Event("text", summary)
        yield Event("image", {"path": str(self.sessions.image_path(record)), "id": record.id})

    def _should_rewrite(self, setting: str) -> bool:
        if setting == "off":
            return False
        if setting == "on":
            return True
        # auto: only when the chat model is already resident (never pay a model swap for it)
        return self.manager.is_loaded(CHAT)


def _drain(q: queue.Queue) -> list:
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items
