"""Gradio multimodal chat UI. Thin layer: all logic lives in ChatController / SessionManager."""

from __future__ import annotations

import time
from collections.abc import Iterator

import gradio as gr

from .app import App
from .controller import Event, TurnOptions
from .gpu_manager import memory_snapshot
from .image_engine import ASPECT_RATIOS, BANDS, ImageOptions
from .router import Mode
from .search_engine import resolve_safesearch
from .session_manager import MessageRecord

MODE_CHOICES = [("Auto", Mode.AUTO.value), ("Chat", Mode.CHAT.value), ("Vision", Mode.VISION.value),
                ("Generate", Mode.GENERATE.value), ("Edit", Mode.EDIT.value)]  # fmt: skip
INTENT_LABEL = {
    "chat": "Chat",
    "vision": "Vision",
    "generate": "Generate",
    "edit": "Edit",
    "restore": "Restore",
}
STREAM_INTERVAL_S = 0.05

CSS = """
#status-panel {font-size: 0.85em}
#route-info {min-height: 1.5em; font-size: 0.85em; opacity: 0.8}
footer {display: none !important}
"""


# ---------------------------------------------------------------------- rendering (pure)
def render_message(m: MessageRecord, image_path) -> list[dict]:
    """Convert one stored message into Gradio 'messages' dicts (reasoning shown collapsible)."""
    out: list[dict] = []
    if m.role == "assistant" and m.reasoning:
        out.append(
            {
                "role": "assistant",
                "content": m.reasoning,
                "metadata": {"title": "🧠 思考プロセス", "status": "done"},
            }
        )
    content: list = []
    text = m.content
    if m.role == "assistant" and m.intent and not m.meta.get("error"):
        footer = INTENT_LABEL.get(m.intent, m.intent)
        if m.model:
            footer += f" · {m.model}"
        text = f"{text}\n\n*{footer}*" if text else f"*{footer}*"
    if text:
        content.append(text)
    for img in m.images:
        content.append({"path": str(image_path(img)), "alt_text": img.caption})
    if content:
        out.append({"role": m.role, "content": content})
    return out


def render_history(app: App, session_id: str | None) -> list[dict]:
    if not session_id or not app.sessions.session_exists(session_id):
        return []
    msgs: list[dict] = []
    for m in app.sessions.get_messages(session_id):
        msgs.extend(render_message(m, app.sessions.image_path))
    return msgs


def pending_messages(reasoning: str, text: str, status: str, errors: list[str]) -> list[dict]:
    out: list[dict] = []
    if reasoning:
        out.append(
            {
                "role": "assistant",
                "content": reasoning,
                "metadata": {"title": "🧠 思考中…", "status": "pending"},
            }
        )
    body = text
    if errors:
        body = (body + "\n\n" if body else "") + "\n".join(f"⚠️ {e}" for e in errors)
    if status and not text:
        body = (body + "\n\n" if body else "") + f"⏳ {status}"
    if body:
        out.append({"role": "assistant", "content": body})
    return out


def session_choices(app: App) -> list[tuple[str, str]]:
    return [(s["title"] or "(無題)", s["id"]) for s in app.sessions.list_sessions()]


def gallery_items(lineage, image_path) -> list[tuple[str, str]]:
    """Build Gallery values from a root-to-current image chain."""
    return [
        (str(image_path(image)), f"rev{image.revision}{' 元' if image.revision == 0 else ''}")
        for image in lineage
    ]


def lineage_view(app: App, session_id: str | None, selected_id: str | None = None):
    if not session_id or not app.sessions.session_exists(session_id):
        return [], [], None, "画像はまだありません"
    images = app.sessions.session_images(session_id)
    if not images:
        return [], [], None, "画像はまだありません"
    current = next((image for image in images if image.id == selected_id), images[-1])
    lineage = app.sessions.lineage(current.id)
    caption = f"対象: rev{current.revision} / {current.kind} / {current.width}×{current.height} / seed {current.seed}"
    return (
        gallery_items(lineage, app.sessions.image_path),
        [image.id for image in lineage],
        current.id,
        caption,
    )


def status_markdown(
    app: App, content_policy: str | None = None, selected_image_id: str | None = None, variations: int = 1
) -> str:
    policy = content_policy or app.cfg.content_policy
    snap = memory_snapshot()
    status = app.manager.status()
    loaded = ", ".join(status["loaded"]) or "なし"
    lines = [
        f"**GPU**: {app.gpu.name} ({app.gpu.total_gib:.0f} GiB)",
        f"**Mode**: {app.profile.mode} (`{app.profile.key}`)",
        f"**VRAM**: {snap.summary()}",
        f"**Loaded**: {loaded}",
        f"**Chat**: {app.chat_label}",
        f"**Image**: {app.image_label}",
        f"**Web検索**: {app.search_label}",
        f"**コンテンツ方針**: {policy} / safesearch={resolve_safesearch(app.cfg.search_safesearch, policy)}",
        f"**対象画像**: {(selected_image_id or 'なし')[:8]} / **バリエーション**: {variations}",
        f"**履歴**: {'Google Drive' if app.cfg.drive_mounted else '⚠️ ローカル（ランタイム終了で消えます）'}",
    ]
    lines += [f"ℹ️ {n}" for n in app.store.notes[-3:]]
    if policy == "open" and app.search_label == "tavily":
        lines.append("⚠️ Tavily は成人向け検索を規約で禁じています。Brave / DDG を推奨")
    return "  \n".join(lines)


def _options(
    mode,
    thinking,
    aspect,
    band,
    steps,
    seed,
    rewrite,
    web_search="auto",
    content_policy="open",
    variations=1,
    selected_image_id=None,
) -> TurnOptions:
    seed_val = None if seed is None or int(seed) < 0 else int(seed)
    return TurnOptions(
        mode=mode or Mode.AUTO.value,
        thinking=bool(thinking),
        prompt_rewrite=rewrite or "auto",
        web_search=web_search or "auto",
        content_policy=content_policy or "open",
        image=ImageOptions(aspect=aspect or "1:1", band=int(band) if band else None,
                           steps=int(steps) if steps else None, seed=seed_val,
                           variations=int(variations or 1)),
        selected_image_id=selected_image_id,
    )  # fmt: skip


# ---------------------------------------------------------------------- UI
def build_ui(app: App) -> gr.Blocks:
    profile = app.profile
    bands = [b for b in BANDS if b <= profile.image_max_band]
    default_band = min(app.cfg.image.default_band, profile.image_max_band)

    def stream_turn(session_id: str, events: Iterator[Event]):
        base = render_history(app, session_id)
        reasoning, text, status, errors = "", "", "", []
        route_md = ""
        last = 0.0
        for ev in events:
            if ev.kind == "route":
                base = render_history(app, session_id)
                d = ev.data
                route_md = f"→ **{INTENT_LABEL[d.intent.value]}**（{d.reason}）"
            elif ev.kind == "reasoning":
                reasoning += ev.data
            elif ev.kind == "text":
                text += ev.data
            elif ev.kind == "status":
                status = str(ev.data)
            elif ev.kind == "error":
                errors.append(str(ev.data))
            elif ev.kind == "done":
                break
            now = time.time()
            if ev.kind in ("reasoning", "text") and now - last < STREAM_INTERVAL_S:
                continue
            last = now
            yield base + pending_messages(reasoning, text, status, errors), route_md
        final = render_history(app, session_id)
        if errors and not any("エラー" in str(m.get("content")) for m in final[-1:]):
            final += pending_messages("", "", "", errors)
        yield final, route_md

    def ensure_session(session_id):
        if session_id and app.sessions.session_exists(session_id):
            return session_id
        return app.sessions.create_session()

    def on_submit(
        msg,
        audio_path,
        session_id,
        selected_id,
        mode,
        thinking,
        web_search,
        content_policy,
        aspect,
        band,
        steps,
        seed,
        rewrite,
        variations,
    ):
        msg = msg or {}
        text, files = msg.get("text", ""), msg.get("files", [])
        paths = [_file_path(f) for f in files]
        if audio_path:
            paths.append(_file_path(audio_path))
        session_id = ensure_session(session_id)
        opts = _options(
            mode,
            thinking,
            aspect,
            band,
            steps,
            seed,
            rewrite,
            web_search,
            content_policy,
            variations,
            selected_id,
        )
        events = app.controller.handle(session_id, text, paths, opts)
        for chat, route_md in stream_turn(session_id, events):
            yield chat, gr.update(value=None), gr.update(value=None), route_md, session_id, gr.update()
        yield (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            session_id,
            gr.update(choices=session_choices(app), value=session_id),
        )

    def on_regenerate(
        session_id,
        selected_id,
        mode,
        thinking,
        web_search,
        content_policy,
        aspect,
        band,
        steps,
        seed,
        rewrite,
        variations,
    ):
        if not session_id:
            yield gr.update(), "再生成できるメッセージがありません"
            return
        opts = _options(
            mode,
            thinking,
            aspect,
            band,
            steps,
            seed,
            rewrite,
            web_search,
            content_policy,
            variations,
            selected_id,
        )
        yield from stream_turn(session_id, app.controller.regenerate(session_id, opts))

    def refresh_lineage(session_id):
        return lineage_view(app, session_id)

    def on_gallery_select(session_id, ids, evt: gr.SelectData):
        if not ids or evt.index is None:
            return None, "画像を選択できませんでした"
        index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        selected = ids[int(index)]
        return selected, lineage_view(app, session_id, selected)[3]

    def on_relative(session_id, selected_id, steps_back):
        _, ids, _, _ = lineage_view(app, session_id, selected_id)
        if len(ids) <= steps_back:
            return selected_id, f"{steps_back}個前の画像はありません"
        target = ids[-1 - steps_back]
        return target, lineage_view(app, session_id, target)[3]

    def on_restore(session_id, selected_id):
        if not session_id:
            yield [], "画像はまだありません"
            return
        opts = TurnOptions(selected_image_id=selected_id, web_search="off")
        yield from stream_turn(session_id, app.controller.handle(session_id, "元に戻して", options=opts))

    def on_stop():
        app.controller.cancel()
        return "⏹ 停止リクエストを送信しました"

    def on_clear(session_id):
        if session_id:
            app.sessions.clear_session(session_id)
            app.store.sync()
        return [], ""

    def on_new():
        sid = app.sessions.create_session()
        return sid, [], gr.update(choices=session_choices(app), value=sid), ""

    def on_select(session_id):
        return session_id, render_history(app, session_id), ""

    def on_delete(session_id):
        if session_id:
            app.sessions.delete_session(session_id)
            app.store.sync()
        choices = session_choices(app)
        sid = choices[0][1] if choices else None
        return sid, render_history(app, sid), gr.update(choices=choices, value=sid)

    def on_unload(content_policy):
        app.manager.unload_all()
        return status_markdown(app, content_policy)

    with gr.Blocks(title="Qwen Multimodal Chat") as demo:
        first = session_choices(app)
        session_state = gr.State(first[0][1] if first else None)
        initial_gallery = lineage_view(app, session_state.value)
        selected_state = gr.State(initial_gallery[2])
        gallery_ids = gr.State(initial_gallery[1])

        with gr.Sidebar(open=True):
            gr.Markdown("### 💬 Qwen Multimodal")
            new_btn = gr.Button("＋ New Chat", variant="primary")
            sessions_radio = gr.Radio(
                choices=first, value=first[0][1] if first else None, label="過去のチャット", interactive=True
            )
            gr.Markdown("#### 🖼 画像の系譜")
            lineage_gallery = gr.Gallery(
                value=initial_gallery[0], columns=4, height=115, show_label=False, preview=False
            )
            selected_caption = gr.Markdown(initial_gallery[3])
            with gr.Row():
                target_btn = gr.Button("この画像を対象にする", size="sm")
                restore_btn = gr.Button("元に戻す", size="sm")
            with gr.Row():
                previous_btn = gr.Button("1個前", size="sm")
                two_back_btn = gr.Button("2個前を編集", size="sm")
            delete_btn = gr.Button("🗑 このチャットを削除", size="sm")
            gr.Markdown("---")
            status_md = gr.Markdown(status_markdown(app), elem_id="status-panel")
            with gr.Row():
                refresh_btn = gr.Button("↻ 状態更新", size="sm")
                unload_btn = gr.Button("⏏ モデル解放", size="sm")
            timer = gr.Timer(5.0)

        if app.cfg.share:
            gr.Markdown("⚠️ **share=True: この画面は公開URLでインターネットからアクセス可能です。**")
        chatbot = gr.Chatbot(
            value=render_history(app, session_state.value),
            height="68vh",
            show_label=False,
            buttons=["copy"],
            placeholder="何でも聞いてね。画像を添付すると理解・編集、「〜を描いて」で画像生成、「最新の〜」はWeb検索します。",
            allow_file_downloads=True,
        )
        route_md = gr.Markdown("", elem_id="route-info")
        textbox = gr.MultimodalTextbox(
            placeholder="メッセージを入力。画像は複数枚添付できます（顔・服装・構図の参照）。PDF・音声も可。",
            file_types=["image", ".pdf", ".wav", ".mp3", ".m4a", ".webm", ".ogg"],
            file_count="multiple",
            show_label=False,
            submit_btn=True,
            autofocus=True,
        )
        with gr.Row():
            audio_input = gr.Audio(
                sources=["microphone", "upload"], type="filepath", label="🎙 音声入力", scale=4
            )
            mic_submit_btn = gr.Button("🎙 音声を送信", size="sm", scale=1)
        with gr.Row():
            mode = gr.Radio(MODE_CHOICES, value=Mode.AUTO.value, label="モード", scale=4)
            thinking = gr.Checkbox(value=app.cfg.thinking_default, label="Thinking（深く考える）", scale=1)
            web_search = gr.Radio(
                [("自動", "auto"), ("常に", "on"), ("オフ", "off")],
                value=app.cfg.web_search,
                label="🌐 Web検索（最新情報）",
                scale=2,
            )
            content_policy = gr.Radio(
                [("開放", "open"), ("標準", "standard")],
                value=app.cfg.content_policy,
                label="コンテンツ方針",
                scale=2,
            )
        with gr.Row():
            stop_btn = gr.Button("⏹ Stop", size="sm")
            regen_btn = gr.Button("🔄 Regenerate", size="sm")
            clear_btn = gr.Button("🧹 Clear", size="sm")
        with gr.Accordion("🎨 画像生成・編集の設定", open=False):
            with gr.Row():
                aspect = gr.Dropdown(list(ASPECT_RATIOS), value="1:1", label="アスペクト比（生成時）")
                band = gr.Dropdown(bands, value=default_band, label="解像度帯（長辺の目安）")
                steps = gr.Slider(1, 60, value=profile.image_default_steps, step=1, label="Steps")
                seed = gr.Number(value=-1, precision=0, label="Seed（-1でランダム）")
            rewrite = gr.Radio(
                [("自動", "auto"), ("常にLLMで最適化", "on"), ("そのまま使う", "off")],
                value=app.cfg.prompt_rewrite,
                label="画像プロンプトの最適化（Chatモデルで英語プロンプト化）",
            )
            variations = gr.Radio(
                [(str(n), n) for n in (1, 4)], value=1, label="バリエーション（画像生成のみ）"
            )

        settings = [
            mode,
            thinking,
            web_search,
            content_policy,
            aspect,
            band,
            steps,
            seed,
            rewrite,
            variations,
        ]
        submit_inputs = [textbox, audio_input, session_state, selected_state, *settings]
        submit_outputs = [chatbot, textbox, audio_input, route_md, session_state, sessions_radio]
        submit_event = textbox.submit(
            on_submit,
            submit_inputs,
            submit_outputs,
            concurrency_id="gpu",
            concurrency_limit=1,
        )
        mic_event = mic_submit_btn.click(
            on_submit, submit_inputs, submit_outputs, concurrency_id="gpu", concurrency_limit=1
        )
        regen_event = regen_btn.click(
            on_regenerate,
            [session_state, selected_state, *settings],
            [chatbot, route_md],
            concurrency_id="gpu",
            concurrency_limit=1,
        )
        lineage_outputs = [lineage_gallery, gallery_ids, selected_state, selected_caption]
        submit_event.then(refresh_lineage, session_state, lineage_outputs)
        mic_event.then(refresh_lineage, session_state, lineage_outputs)
        regen_event.then(refresh_lineage, session_state, lineage_outputs)
        lineage_gallery.select(
            on_gallery_select, [session_state, gallery_ids], [selected_state, selected_caption]
        )
        target_btn.click(lambda selected: f"対象画像: {selected or 'なし'}", selected_state, selected_caption)
        previous_btn.click(
            lambda sid, selected: on_relative(sid, selected, 1),
            [session_state, selected_state],
            [selected_state, selected_caption],
        )
        two_back_btn.click(
            lambda sid, selected: on_relative(sid, selected, 2),
            [session_state, selected_state],
            [selected_state, selected_caption],
        )
        restore_event = restore_btn.click(
            on_restore,
            [session_state, selected_state],
            [chatbot, route_md],
            concurrency_id="gpu",
            concurrency_limit=1,
        )
        restore_event.then(refresh_lineage, session_state, lineage_outputs)
        stop_btn.click(on_stop, None, route_md, queue=False)
        clear_btn.click(on_clear, session_state, [chatbot, route_md]).then(
            refresh_lineage, session_state, lineage_outputs
        )
        new_btn.click(on_new, None, [session_state, chatbot, sessions_radio, route_md]).then(
            refresh_lineage, session_state, lineage_outputs
        )
        sessions_radio.input(on_select, sessions_radio, [session_state, chatbot, route_md]).then(
            refresh_lineage, session_state, lineage_outputs
        )
        delete_btn.click(on_delete, session_state, [session_state, chatbot, sessions_radio]).then(
            refresh_lineage, session_state, lineage_outputs
        )
        refresh_btn.click(
            lambda p, selected, n: status_markdown(app, p, selected, n),
            [content_policy, selected_state, variations],
            status_md,
        )
        content_policy.change(
            lambda p, selected, n: status_markdown(app, p, selected, n),
            [content_policy, selected_state, variations],
            status_md,
        )
        unload_btn.click(on_unload, content_policy, status_md, concurrency_id="gpu", concurrency_limit=1)
        timer.tick(
            lambda p, selected, n: status_markdown(app, p, selected, n),
            [content_policy, selected_state, variations],
            status_md,
            show_progress="hidden",
        )

        def on_load(content_policy):
            # re-read on every page load: the list built at startup is stale after new chats
            choices = session_choices(app)
            sid = choices[0][1] if choices else None
            view = lineage_view(app, sid)
            return (
                sid,
                render_history(app, sid),
                gr.update(choices=choices, value=sid),
                status_markdown(app, content_policy),
                *view,
            )

        demo.load(
            on_load, content_policy, [session_state, chatbot, sessions_radio, status_md, *lineage_outputs]
        )
    return demo


def _file_path(f) -> str:
    if isinstance(f, str):
        return f
    if isinstance(f, dict):
        return f.get("path") or f.get("name") or ""
    return getattr(f, "path", None) or getattr(f, "name", "")


def launch(app: App, demo: gr.Blocks | None = None, **kwargs):
    demo = demo or build_ui(app)
    auth = None
    if app.cfg.auth_user and app.cfg.auth_password:
        auth = (app.cfg.auth_user, app.cfg.auth_password)
    demo.queue(default_concurrency_limit=4)
    return demo.launch(
        server_name=app.cfg.server_host,
        server_port=app.cfg.server_port,
        share=app.cfg.share,
        auth=auth,
        allowed_paths=[str(app.cfg.data_dir.resolve())],
        css=CSS,
        **kwargs,
    )
