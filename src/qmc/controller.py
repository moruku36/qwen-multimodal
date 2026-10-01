"""Chat Controller: one user turn -> route -> engine -> persisted history, as a stream of events.

The controller is UI-agnostic: it yields ``Event`` objects that the Gradio UI (or a future
FastAPI / OpenAI-compatible frontend) renders. It never imports gradio.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .agent import GitHubReader, ResearchAgent, find_github_repos, needs_agent
from .asr import ASRBackend
from .backends.base import Cancelled, ChatParams
from .chat_engine import (
    ChatEngine,
    ContextImage,
    ContextMessage,
    appearance_card_for_references,
    card_summary_ja,
)
from .image_engine import MAX_CONDITION_IMAGES, ImageEngine, ImageOptions
from .imaging import (
    InvalidImageError,
    load_user_image,
    render_pdf_pages,
    sample_video_frames,
    save_png,
    slice_tall_image,
    webm_has_video,
)
from .model_manager import CHAT, IMAGE, ModelManager
from .policy import MINOR_REFUSAL, blocks_minor_sexual_request
from .router import ImageTarget, Intent, Mode, RouteContext, RouteDecision, route
from .search_engine import (
    WebSearchEngine,
    appearance_fallback_query,
    appearance_rewrite_conflicts,
    build_search_context,
    fetch_page_text,
    filter_appearance_results,
    format_sources,
    is_refusal,
    needs_web_search,
    preserves_adult_terms,
    resolve_safesearch,
    safe_appearance_queries,
    search_digest,
    usable_search_queries,
)
from .session_manager import ImageRecord, MessageRecord, SessionManager
from .tts import TTSBackend, speakable_text
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
    prompt_rewrite: str = "on"  # auto | on | off
    web_search: str = "on"  # auto | on | off
    agent: str | None = None  # auto | on | off; None -> application default
    content_policy: str | None = None  # None -> application default
    selected_image_id: str | None = None
    mask_image: Any = None


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
        agent_setting: str = "auto",
        agent_max_steps: int = 14,
        agent_max_chars: int = 30000,
        github_token: str | None = None,
        pdf_max_pages: int = 6,
        video_max_seconds: int = 30,
        video_max_mb: int = 80,
        asr: ASRBackend | None = None,
        tts: TTSBackend | None = None,
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
        self.agent_setting = agent_setting
        self.agent_max_steps = agent_max_steps
        self.agent_max_chars = agent_max_chars
        self.github = GitHubReader(github_token)
        self.pdf_max_pages = pdf_max_pages
        self.video_max_seconds = video_max_seconds
        self.video_max_mb = video_max_mb
        self.asr = asr
        self.tts = tts
        self.last_tts_error: str | None = None
        self.cancel_event = threading.Event()
        self._status_q: queue.Queue = queue.Queue()
        manager.on_status = self._status_q.put

    # ------------------------------------------------------------------ public API
    def cancel(self) -> None:
        self.cancel_event.set()

    def read_last_answer(self, session_id: str) -> str | None:
        self.last_tts_error = None
        if self.tts is None:
            return None
        messages = self.sessions.get_messages(session_id)
        if not messages or messages[-1].role != "assistant" or messages[-1].meta.get("error"):
            return None
        spoken = speakable_text(messages[-1].content)
        if not spoken:
            return None
        suffix = ".wav" if self.tts.__class__.__name__ == "MockTTS" else ".mp3"
        output = (
            self.sessions.data_dir / "sessions" / session_id / "audio" / f"{uuid.uuid4().hex[:8]}{suffix}"
        )
        try:
            return str(self.tts.synthesize(spoken, output))
        except Exception as exc:
            log.warning("TTS failed: %s", exc)
            self.last_tts_error = f"読み上げをスキップしました: {exc}"
            return None

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
        transcripts = []
        for f in files:
            path = Path(f)
            suffix = path.suffix.lower()
            try:
                if suffix == ".pdf":
                    for index, page in enumerate(
                        render_pdf_pages(path, self.pdf_max_pages, self.max_image_side, self.max_upload_mb), 1
                    ):
                        uploads.append((page, {"pdf_page": index, "pdf_name": path.name}))
                elif suffix in {".mp4", ".mov", ".mkv"} or (suffix == ".webm" and webm_has_video(path)):
                    frames, duration = sample_video_frames(
                        path, max_mb=self.video_max_mb, max_duration_s=self.video_max_seconds
                    )
                    extra_paths = []
                    for index, frame in enumerate(frames[1:], 2):
                        rel = (
                            Path("sessions")
                            / session_id
                            / "vision_frames"
                            / f"{uuid.uuid4().hex[:8]}-{index}.png"
                        )
                        save_png(frame, self.sessions.data_dir / rel)
                        extra_paths.append(rel.as_posix())
                    uploads.append(
                        (
                            frames[0],
                            {
                                "video": True,
                                "frames": len(frames),
                                "duration_s": duration,
                                "frame_paths": extra_paths,
                            },
                        )
                    )
                elif suffix in {".wav", ".mp3", ".m4a", ".webm", ".ogg"}:
                    if self.asr is None:
                        raise ValueError("音声認識のパッケージがありません")
                    transcript = self.asr.transcribe(path)
                    if transcript:
                        transcripts.append(transcript)
                elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".mpo"}:
                    image, info = load_user_image(path, self.max_image_side, self.max_upload_mb)
                    meta = {"original_size": list(info["original_size"]), "downscaled": info["downscaled"]}
                    slices = slice_tall_image(path, self.max_image_side)
                    if slices:
                        rel_paths = []
                        for index, band in enumerate(slices):
                            rel = (
                                Path("sessions")
                                / session_id
                                / "vision_slices"
                                / f"{uuid.uuid4().hex[:8]}-{index}.png"
                            )
                            save_png(band, self.sessions.data_dir / rel)
                            rel_paths.append(rel.as_posix())
                        meta.update({"sliced": True, "bands": len(slices), "slice_paths": rel_paths})
                    uploads.append((image, meta))
                else:
                    raise ValueError(f"未対応のファイル形式です: {path.name}")
            except (InvalidImageError, ValueError, OSError, RuntimeError) as exc:
                if "音声認識のパッケージがありません" in str(exc):
                    yield Event("status", str(exc))
                yield Event("error", str(exc))
        if transcripts:
            text = "\n".join([*transcripts, text] if text else transcripts).strip()
            yield Event("status", f"🎙 文字起こし: {' / '.join(transcripts)}")
        if not text and uploads and all("pdf_page" in meta for _, meta in uploads):
            text = "この資料の内容をページ順に要約し、表や数字は省略せず書いてください。"
        if not text and any(meta.get("video") for _, meta in uploads):
            text = "この動画で何が起きているか、時系列で説明してください。"
        if not text and not uploads:
            return

        user_msg_id = self.sessions.add_message(session_id, "user", text)
        for img, meta in uploads:
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
        decision = route(user_msg.content, self._route_context(session_id, user_msg, options), options.mode)
        if options.selected_image_id and options.selected_image_id not in {
            image.id for image in self.sessions.session_images(session_id)
        }:
            decision.warnings.append("選択した画像が見つからないため、最新の画像を使います")
        self.sessions.update_message(user_msg_id, intent=decision.intent.value)
        yield Event("route", decision)
        if blocks_minor_sexual_request(user_msg.content):
            self.sessions.add_message(
                session_id,
                "assistant",
                MINOR_REFUSAL,
                intent=decision.intent.value,
                meta={"policy": "blocked_minor"},
            )
            if self.after_turn:
                self.after_turn()
            yield Event("text", MINOR_REFUSAL)
            yield Event("done", None)
            return
        for w in decision.warnings:
            yield Event("status", w)
        try:
            if decision.intent is Intent.RESTORE:
                yield from self._restore(session_id, user_msg, options)
            elif decision.intent in (Intent.CHAT, Intent.VISION):
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

    def _route_context(self, session_id: str, user_msg: MessageRecord, options: TurnOptions) -> RouteContext:
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
            has_uploads=bool(user_msg.images),
            upload_count=len(user_msg.images),
            has_session_image=bool(prior),
            has_selected_image=bool(options.selected_image_id),
            turns_since_last_image=turns,
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
        self,
        session_id: str,
        user_msg: MessageRecord,
        target: ImageTarget,
        selected_image_id: str | None = None,
        n: int = 0,
    ) -> list[ImageRecord]:
        if target is ImageTarget.UPLOADED:
            return list(user_msg.images)
        prior = [i for i in self.sessions.session_images(session_id) if (i.message_id or 0) < user_msg.id]
        if not prior:
            return []
        latest = prior[-1]
        selected = next((image for image in prior if image.id == selected_image_id), latest)
        if target is ImageTarget.LATEST:
            return [latest]
        if target is ImageTarget.SELECTED:
            return [selected]
        if target is ImageTarget.NTH:
            image = self.sessions.nth_previous_image(session_id, n)
            return [image] if image and image.id in {item.id for item in prior} else []
        if target is ImageTarget.VARIATION:
            selected_group = selected.meta.get("variation_of")
            if not selected_group:
                selected_group = next(
                    (
                        item.meta.get("variation_of")
                        for item in reversed(prior)
                        if item.meta.get("variation_of")
                    ),
                    None,
                )
            groups = (
                [item for item in prior if item.meta.get("variation_of") == selected_group]
                if selected_group
                else []
            )
            sibling = next((item for item in groups if item.meta.get("variation") == n), None)
            return [sibling] if sibling else []
        if target is ImageTarget.ROOT:
            return [self.sessions.lineage(selected.id)[0]]
        lineage = self.sessions.lineage(latest.id)
        if target is ImageTarget.PARENT_AND_LATEST:
            if len(lineage) >= 2:
                return [lineage[-2], latest]
            return prior[-2:]
        # ROOT_AND_LATEST: original of this lineage; fall back to the first image of the session
        first = lineage[0] if len(lineage) >= 2 else prior[0]
        return [first, latest] if first.id != latest.id else [latest]

    def _restore(self, session_id: str, user_msg: MessageRecord, options: TurnOptions) -> Iterator[Event]:
        selected = self._resolve_targets(
            session_id, user_msg, ImageTarget.SELECTED, options.selected_image_id
        )
        if not selected:
            raise ValueError("復元する画像がありません")
        current = selected[0]
        root = self.sessions.lineage(current.id)[0]
        summary = f"元画像に戻しました（rev{current.revision + 1}）"
        message_id = self.sessions.add_message(
            session_id, "assistant", summary, intent=Intent.RESTORE.value, model="コピー"
        )
        record = self.sessions.add_image(
            session_id,
            self.sessions.load_image(root),
            "edited",
            message_id=message_id,
            parent_id=current.id,
            meta={"restore": True, "restored_from": root.id},
        )
        yield Event("text", summary)
        yield Event("image", {"path": str(self.sessions.image_path(record)), "id": record.id})

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
            records = self._resolve_targets(
                session_id, user_msg, decision.target, options.selected_image_id, decision.n
            )
            targets = []
            sliced_ids = set()
            for record in records:
                if record.meta.get("frame_paths"):
                    targets.append(self._ctx_image(record))
                    for index, rel in enumerate(record.meta["frame_paths"], 2):
                        targets.append(
                            ContextImage(
                                f"{record.id}-frame-{index}",
                                str(self.sessions.data_dir / rel),
                                f"{record.caption} フレーム {index}",
                            )
                        )
                elif record.meta.get("slice_paths"):
                    sliced_ids.add(record.id)
                    for index, rel in enumerate(record.meta["slice_paths"], 1):
                        targets.append(
                            ContextImage(
                                f"{record.id}-slice-{index}",
                                str(self.sessions.data_dir / rel),
                                f"{record.caption} 分割 {index}",
                            )
                        )
                else:
                    targets.append(self._ctx_image(record))
            if not targets:
                raise ValueError("対象の画像が見つかりません。画像を添付してください。")
            if sliced_ids and history:
                last = history[-1]
                history[-1] = ContextMessage(
                    last.role, last.text, [image for image in last.images if image.image_id not in sliced_ids]
                )
            if not user_msg.content:
                history[-1] = ContextMessage(
                    "user", "この画像について詳しく説明してください。", history[-1].images
                )

            # Multimodal answer: when the question also needs the web ("これは何？最新の情報も調べて"),
            # describe the image first so the search queries are about what is actually in it.
            system_extra, sources_md, search_meta = None, "", None
            if (
                user_msg.content
                and options.web_search != "off"
                and needs_web_search(user_msg.content, "auto")
            ):
                yield Event("status", "🖼️ 画像の内容を確認して検索クエリに反映します…")
                caption = self._caption_for_search(history, targets, policy)
                augmented = (
                    replace(user_msg, content=f"{user_msg.content}\n（画像の内容: {caption}）")
                    if caption
                    else user_msg
                )
                for ev in self._web_search(session_id, augmented, replace(options, web_search="auto")):
                    if isinstance(ev, Event):
                        yield ev
                    else:
                        system_extra, sources_md, search_meta = ev

            def gen():
                return self.vision.stream(
                    history,
                    targets,
                    params,
                    self.cancel_event,
                    compare=decision.compare,
                    content_policy=policy,
                    **({"system_extra": system_extra} if system_extra else {}),
                )
        else:
            system_extra, sources_md, search_meta = None, "", None
            extra_chars = 6000
            if needs_agent(user_msg.content, options.agent or self.agent_setting):
                for ev in self._research(user_msg, options):
                    if isinstance(ev, Event):
                        yield ev
                    else:
                        system_extra, sources_md, search_meta = ev
                        extra_chars = self.agent_max_chars + 2000
            else:
                for ev in self._web_search(session_id, user_msg, options):
                    if isinstance(ev, Event):
                        yield ev
                    else:
                        system_extra, sources_md, search_meta = ev

            stream_kwargs = {"extra_chars": extra_chars} if extra_chars != 6000 else {}

            def gen():
                return self.chat.stream(
                    history,
                    params,
                    self.cancel_event,
                    system_extra=system_extra,
                    content_policy=policy,
                    **stream_kwargs,
                )

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
                    if decision.intent is Intent.VISION:
                        yield Event("reasoning", item.reasoning)
                if item.content:
                    content.append(item.content)
                    if decision.intent is Intent.VISION:
                        yield Event("text", item.content)
        except RuntimeError as exc:
            detail = str(exc).lower()
            if not any(
                word in detail
                for word in ("context", "too many tokens", "exceed", "slot", "prompt is too long")
            ):
                raise
            # One retry with only the current turn. Previous turns remain in the session.
            yield Event("status", "会話が長いため、今回の質問だけで再試行します。履歴は残ります")
            history = history[-1:]
            if decision.intent is Intent.VISION:
                targets = targets[:4]
            params.max_tokens = 1024
            for item in self._pump(producer):
                if isinstance(item, Event):
                    yield item
                else:
                    if item.reasoning:
                        reasoning.append(item.reasoning)
                    if item.content:
                        content.append(item.content)
                        if decision.intent is Intent.VISION:
                            yield Event("text", item.content)
        except Cancelled:
            cancelled = True
        text = "".join(content).strip()
        search_failed = bool(search_meta and (search_meta.get("error") or not search_meta.get("urls")))
        if (
            not cancelled
            and decision.intent is not Intent.VISION
            and policy == "open"
            and is_refusal(text)
            and not search_failed  # nothing to prefer over the model's answer without search results
        ):
            yield Event("status", "拒否だったため検索結果を優先して再生成します")
            if search_meta is None:
                for ev in self._web_search(session_id, user_msg, replace(options, web_search="on")):
                    if isinstance(ev, Event):
                        yield ev
                    else:
                        system_extra, sources_md, search_meta = ev
            else:
                history[-1] = ContextMessage(
                    history[-1].role,
                    history[-1].text
                    + "\n検索結果にある固有名詞とURLを列挙してください。拒否・説教・言い換えをしないでください。未成年者の性的内容以外は拒否しないでください。",
                    history[-1].images,
                )
            content, reasoning = [], []
            try:
                for item in self._pump(producer):
                    if isinstance(item, Event):
                        yield item
                    else:
                        if item.reasoning:
                            reasoning.append(item.reasoning)
                        if item.content:
                            content.append(item.content)
            except Cancelled:
                cancelled = True
            text = "".join(content).strip()
        if decision.intent is not Intent.VISION and text:
            yield Event("text", text)
            if reasoning:
                yield Event("reasoning", "".join(reasoning))
        if cancelled:
            text += "\n\n（停止しました）"
        meta = {"duration_s": round(time.time() - t0, 2), "cancelled": cancelled}
        if search_meta:
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

    def _caption_for_search(self, history, targets, policy: str) -> str:
        """One short description of the attached image(s), used only to build better search queries."""
        try:
            last = history[-1]
            asked = [
                *history[:-1],
                ContextMessage(
                    last.role,
                    "この画像に写っている主な被写体・人物・作品名・文字を、1〜2文で簡潔に書いてください。",
                    last.images,
                ),
            ]
            text = "".join(
                d.content
                for d in self.vision.stream(
                    asked,
                    targets,
                    ChatParams(thinking=False, max_tokens=120),
                    self.cancel_event,
                    content_policy=policy,
                )
            ).strip()
        except Cancelled:
            raise
        except Exception as exc:
            log.warning("Image caption for search failed: %s", exc)
            return ""
        return "" if is_refusal(text) else text[:300]

    def _research(self, user_msg: MessageRecord, options: TurnOptions):
        """Let the model read repo files / pages itself. Yields status Events, then one
        (evidence, sources_md, meta) tuple (evidence is None if nothing could be read)."""
        text = user_msg.content
        repos = find_github_repos(text)
        yield Event("status", "🔎 調査エージェントを開始します（自分でファイルを読みに行きます）")

        def web_search(query: str) -> str:
            if self.search is None or not self.search.available:
                raise RuntimeError("Web検索が使えません")
            resp = self.search.search(query)
            return build_search_context(resp, max_chars=6000)

        def fetch(url: str) -> str:
            return fetch_page_text(url, limit=8000)

        agent = ResearchAgent(
            self.chat.complete_text,
            self.github,
            search=web_search if options.web_search != "off" else None,
            fetch=fetch,
            max_steps=self.agent_max_steps,
            max_chars=self.agent_max_chars,
        )
        result = None
        for item in agent.run(text, repos, self.cancel_event):
            if isinstance(item, str):
                yield Event("status", item)
            else:
                result = item
        if result is None or not result.observations:
            yield (None, "", None)
            return
        read = sum(1 for o in result.observations if o.source and "github_read" in o.call)
        yield Event("status", f"調査完了: {len(result.observations)} 回の参照（ファイル {read} 件）")
        meta = {
            "purpose": "agent",
            "repos": repos,
            "steps": len(result.observations),
            "sources": [o.source for o in result.observations if o.source],
            "finished": result.finished,
        }
        yield (agent.evidence(result), result.sources_md(), meta)

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
        rewritten = self.chat.rewrite_search_queries(text, history)
        queries = usable_search_queries(text, rewritten)
        query = queries[0]
        log.info("Web search queries: original=%r effective=%r", text, queries)
        if policy == "open" and self.search.provider_name == "tavily":
            yield Event("status", "⚠️ Tavily は成人向け検索を規約で禁じています。Brave / DDG を推奨")
        yield Event("status", f"🔎 Web検索中: {' / '.join(queries)}")
        resp = self.search.search_many(queries, safesearch=safesearch)
        meta = {
            "query": query,
            "queries": queries,
            "original": text,
            "rewritten": rewritten,
            "provider": resp.provider,
            "urls": [r.url for r in resp.results],
            "safesearch": safesearch,
        }
        if resp.error or not resp.results:
            yield Event("status", resp.error or f"🔎 「{query}」の検索結果がありませんでした")
            meta["error"] = resp.error
            reason = resp.error or "検索結果が見つかりませんでした"
            yield (
                f"Web検索を行いましたが「{query}」の結果は得られませんでした。"
                "検索結果が無いことを伝え、未確認の情報を断定しないでください。",
                f"\n\n---\n⚠️ Web検索で結果を取得できませんでした（{resp.provider}）: {reason}",
                meta,
            )
            return
        yield Event("status", f"📄 {len(resp.results)}件の結果を読み込みました（{resp.provider}）")
        yield (build_search_context(resp), format_sources(resp), meta)

    # ------------------------------------------------------------------ generate / edit
    def _image(
        self, session_id: str, user_msg: MessageRecord, decision: RouteDecision, options: TurnOptions
    ) -> Iterator[Event]:
        turn_started = time.monotonic()
        is_edit = decision.intent is Intent.EDIT
        sources = (
            self._resolve_targets(
                session_id, user_msg, decision.target, options.selected_image_id, decision.n
            )
            if is_edit
            else []
        )
        if is_edit and user_msg.images and options.selected_image_id:
            selected = self._resolve_targets(
                session_id, user_msg, ImageTarget.SELECTED, options.selected_image_id
            )
            if selected and selected[0].id not in {s.id for s in sources}:
                sources.append(selected[0])
        if len(sources) > MAX_CONDITION_IMAGES:
            sources = sources[: MAX_CONDITION_IMAGES - 1] + sources[-1:]
            yield Event("status", f"参照画像は上限{MAX_CONDITION_IMAGES}枚に絞りました")
        if is_edit and not sources:
            raise ValueError("編集する画像がありません。画像を添付するか、先に画像を生成してください。")
        if is_edit:
            yield Event("status", f"参照画像 {len(sources)}枚で編集します")
            if sources[-1].meta.get("variation"):
                yield Event("status", f"編集対象: var{sources[-1].meta['variation']} (id={sources[-1].id})")
        instruction = user_msg.content or ("この画像を高品質に整えてください" if is_edit else "")
        if not instruction:
            raise ValueError("画像の内容を入力してください。")

        effective = instruction
        policy = options.content_policy or self.content_policy
        appearance_card, sources_md, search_meta = None, "", None
        unverified = False
        unverified_reason = ""
        answer_context = None  # search results reused for the written explanation (compound requests)
        if decision.search_appearance:
            if options.web_search == "off":
                yield Event("status", "外見検索はオフです。指定された参照画像と指示を優先します")
            elif self.search is None or not self.search.available:
                unverified = True
                unverified_reason = "Web検索が使えません（ddgs 未インストール / APIキー未設定）"
                yield Event("status", f"⚠️ 外見を検索できませんでした: {unverified_reason}")
            else:
                yield Event("status", "外見・公式設定を検索中…")
                rewritten_queries = self.chat.rewrite_appearance_queries(
                    instruction, self._context(session_id, user_msg.id)[:-1]
                )
                queries = safe_appearance_queries(instruction, rewritten_queries)
                safesearch = resolve_safesearch(self.search_safesearch, policy)
                if policy == "open" and self.search.provider_name == "tavily":
                    yield Event("status", "⚠️ Tavily は成人向け検索を規約で禁じています。Brave / DDG を推奨")
                resp = self.search.search_many(
                    queries, safesearch=safesearch, result_filter=filter_appearance_results
                )
                search_meta = {
                    "query": queries[0],
                    "queries": queries,
                    "original": instruction,
                    "provider": resp.provider,
                    "urls": [r.url for r in resp.results],
                    "safesearch": safesearch,
                    "purpose": "appearance",
                }
                if resp.error:
                    search_meta["error"] = resp.error
                if resp.results:
                    yield Event(
                        "status", f"📄 {len(resp.results)}件の結果を読み込みました（{resp.provider}）"
                    )
                    context_text = build_search_context(resp)
                    answer_context = context_text
                    appearance_card = self.chat.build_appearance_card(
                        instruction, context_text, self._context(session_id, user_msg.id)[:-1]
                    )
                    sources_md = format_sources(resp)
                    if appearance_card:
                        yield Event("status", "外見カードを作成しました")
                    else:
                        digest = search_digest(resp)
                        if digest:
                            # never throw away a successful search: fall back to raw excerpts
                            appearance_card = "NOTES (unverified web excerpts):\n" + digest
                            yield Event(
                                "status", "外見カードを抽出できなかったため、検索結果の抜粋を参考にします"
                            )
                else:
                    unverified = True
                    unverified_reason = resp.error or "外見の検索結果が0件でした"
                    yield Event(
                        "status",
                        f"⚠️ 外見の検索結果を取得できませんでした（{resp.provider}）: {unverified_reason}",
                    )
                log.info(
                    "Appearance search: topic=%r queries=%r results=%d card=%s",
                    appearance_fallback_query(instruction),
                    queries,
                    len(resp.results),
                    "yes" if appearance_card else "no",
                )
        if appearance_card and sources:
            appearance_card = appearance_card_for_references(appearance_card)
        answer_text = ""
        if decision.also_answer and not is_edit and not self.cancel_event.is_set():
            yield Event("status", "解説を作成中…")
            note = (
                "この返答のあとに、依頼された画像が自動で生成されます。ここでは依頼された解説だけを書き、"
                "画像そのものは出力しないでください。"
            )
            system_extra = "\n\n".join(
                x for x in (answer_context, appearance_card and f"外見メモ:\n{appearance_card}", note) if x
            )
            answer_parts: list[str] = []

            def answer_producer(q: queue.Queue) -> None:
                for delta in self.chat.stream(
                    self._context(session_id, user_msg.id),
                    ChatParams(thinking=options.thinking),
                    self.cancel_event,
                    system_extra=system_extra,
                    content_policy=policy,
                    extra_chars=self.agent_max_chars,
                ):
                    q.put(delta)

            try:
                for item in self._pump(answer_producer):
                    if isinstance(item, Event):
                        yield item
                    elif item.content:
                        answer_parts.append(item.content)
            except Cancelled:
                raise
            answer_text = "".join(answer_parts).strip()
            if answer_text and is_refusal(answer_text) and policy == "open":
                answer_text = ""  # a refusal must not replace the image the user asked for
            if answer_text:
                yield Event("text", answer_text + "\n\n")

        if self._should_rewrite(options.prompt_rewrite):
            yield Event("status", "プロンプトを最適化中…")
            rewrite_args = (
                instruction,
                "edit" if is_edit else "generate",
                self._context(session_id, user_msg.id)[:-1],
                len(sources),
            )
            rewritten = (
                self.chat.rewrite_image_prompt(*rewrite_args, appearance_card=appearance_card)
                if appearance_card
                else self.chat.rewrite_image_prompt(*rewrite_args)
            )
            if (
                rewritten
                and not is_refusal(rewritten)
                and (policy != "open" or preserves_adult_terms(instruction, rewritten))
                and (not appearance_card or not appearance_rewrite_conflicts(rewritten, appearance_card))
            ):
                effective = rewritten
        if appearance_card and self._should_rewrite(options.prompt_rewrite):
            effective = f"{effective}\nIdentity traits from web sources:\n{appearance_card}"
        if unverified and not sources:
            effective += "\nAppearance is unverified. Do not invent specific face, hair or outfit traits."
        yield Event("status", f"画像プロンプト: {effective}")

        pil_sources = [self.sessions.load_image(s) for s in sources]
        parent_seeds = set()
        for s in sources:
            parent_seeds |= {i.seed for i in self.sessions.lineage(s.id) if i.seed is not None}
        if is_edit:
            request = self.images.build_edit_request(effective, pil_sources, options.image, parent_seeds)
            mask = options.mask_image
            if mask is None and user_msg.meta.get("mask_path"):
                from PIL import Image

                with Image.open(self.sessions.data_dir / user_msg.meta["mask_path"]) as image:
                    mask = image.copy()
            if mask is not None:
                request.mask_image = mask.convert("L").resize(pil_sources[-1].size)
        else:
            request = self.images.build_request(effective, options.image)

        if not self.manager.is_loaded(IMAGE) and self.manager.get(IMAGE).uses_local_gpu:
            yield Event("status", "Qwen-Image-2.1 をロード中…（初回はダウンロードで時間がかかります）")

        variations = 1 if is_edit else max(1, min(int(options.image.variations), 4))
        if self.images.profile.key == "l4" and variations > 2:
            yield Event("status", "⚠️ L4で4枚生成すると時間がかかります。順番に生成します")
        preparation_s = time.monotonic() - turn_started
        t0 = time.monotonic()
        yield Event(
            "status",
            f"画像処理: {request.output_resolution}帯 / {request.steps} steps / {variations}枚（順次）",
        )
        image_label = getattr(self.manager.get(IMAGE), "label", IMAGE)
        msg_id = None
        records = []
        for index in range(variations):
            if self.cancel_event.is_set():
                break
            next_seed = request.seed if index == 0 else (request.seed + index * 7919) % (2**31)
            current_request = replace(request, seed=next_seed)
            if variations > 1:
                yield Event("status", f"バリエーション {index + 1}/{variations}")

            image_started = time.monotonic()

            def producer(
                q: queue.Queue, image_request=current_request, started=image_started, variation=index + 1
            ) -> None:
                def progress(step: int, total: int) -> None:
                    elapsed = time.monotonic() - started
                    q.put(
                        Event(
                            "status",
                            f"{'編集' if is_edit else '生成'}中… {variation}/{variations}枚 "
                            f"{step}/{total} step / 今回の画像処理 {elapsed:.0f}s経過（ロード含む）",
                        )
                    )

                q.put(self.images.run(image_request, progress=progress, cancel=self.cancel_event))

            result = None
            try:
                for item in self._pump(producer):
                    if isinstance(item, Event):
                        yield item
                    else:
                        result = item
            except Cancelled:
                if not records:
                    raise
                break
            if result is None:
                break
            if msg_id is None:
                msg_id = self.sessions.add_message(
                    session_id, "assistant", "", intent=decision.intent.value, model=image_label
                )
            image_seconds = round(time.monotonic() - image_started, 2)
            image_meta = {"prompt": instruction, "effective_prompt": effective, "duration_s": image_seconds}
            if request.mask_image is not None:
                rel = Path("sessions") / session_id / "masks" / f"{uuid.uuid4().hex[:8]}.png"
                save_png(request.mask_image, self.sessions.data_dir / rel)
                image_meta["mask_path"] = rel.as_posix()
                self.sessions.update_message(user_msg.id, meta={**user_msg.meta, "mask_path": rel.as_posix()})
            if variations > 1:
                image_meta.update(
                    {"variation": index + 1, "variation_of": records[0].id if records else None}
                )
            parent = sources[-1] if sources else None
            record = self.sessions.add_image(
                session_id,
                result,
                "edited" if is_edit else "generated",
                message_id=msg_id,
                parent_id=parent.id if parent else None,
                seed=current_request.seed,
                meta=image_meta,
            )
            if variations > 1 and not records:
                image_meta["variation_of"] = record.id
                self.sessions.update_image_meta(record.id, image_meta)
            records.append(record)
            self.sessions.add_generation(
                session_id,
                kind="edit" if is_edit else "generate",
                prompt=instruction,
                effective_prompt=effective,
                params=current_request.params_dict(),
                model=image_label,
                duration_s=image_seconds,
                message_id=msg_id,
                image_id=record.id,
                source_image_ids=[s.id for s in sources],
            )
            yield Event("image", {"path": str(self.sessions.image_path(record)), "id": record.id})
        if not records:
            if self.cancel_event.is_set():
                raise Cancelled()
            return
        duration = round(time.monotonic() - t0, 2)
        verb = "編集" if is_edit else "生成"
        summary = (
            f"画像を{verb}しました（{records[0].width}×{records[0].height}, "
            f"{request.steps} step, {len(records)}枚, 画像処理 {duration:.0f}s（ロード含む）, "
            f"前処理 {preparation_s:.0f}s）"
        )
        note = (
            card_summary_ja(appearance_card)
            if appearance_card and not appearance_card.startswith("NOTES")
            else ""
        )
        if note:
            summary += f"\n\n🔎 {note}（Web検索で確認）"
        elif appearance_card:
            summary += "\n\n🔎 Web検索の抜粋を参考にしました（外見は未検証）"
        if unverified:
            summary += (
                f"\n\n⚠️ 外見を確認できませんでした（{unverified_reason}）。"
                "固有キャラの同一性を高めるには参照画像を添付してください。"
            )
        summary += sources_md
        shown = summary  # the explanation, if any, was already streamed to the UI
        if answer_text:
            summary = answer_text + "\n\n" + summary
        summary_meta = {
            "prompt": instruction,
            "effective_prompt": effective,
            "duration_s": duration,
            "variations": len(records),
            "preparation_s": round(preparation_s, 2),
        }
        if search_meta:
            summary_meta["web_search"] = search_meta
        self.sessions.update_message(
            msg_id,
            content=summary,
            meta=summary_meta,
        )
        yield Event("text", shown)

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
