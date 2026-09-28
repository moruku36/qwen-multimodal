"""Gradio multimodal chat UI. Thin layer: all logic lives in ChatController / SessionManager."""

from __future__ import annotations

import time
from collections.abc import Iterator

import gradio as gr

from .app import App
from .controller import Event, TurnOptions
from .gpu_manager import memory_snapshot
from .image_engine import ASPECT_RATIOS, BANDS, ImageOptions
from .imaging import mask_from_editor
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
UI_THEME = gr.themes.Base(primary_hue="fuchsia", neutral_hue="slate").set(
    body_background_fill="#101012",
    body_background_fill_dark="#101012",
    body_text_color="#f4f4f5",
    body_text_color_dark="#f4f4f5",
    background_fill_primary="#18181b",
    background_fill_primary_dark="#18181b",
    background_fill_secondary="#222226",
    background_fill_secondary_dark="#222226",
    block_background_fill="#18181b",
    block_background_fill_dark="#18181b",
    button_primary_background_fill="#453449",
    button_primary_background_fill_dark="#453449",
    button_primary_background_fill_hover="#604364",
    button_primary_background_fill_hover_dark="#604364",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    input_background_fill="#222226",
    input_background_fill_dark="#222226",
)

CSS = """
#conversation-shell, #settings-shell {max-width: 940px; margin: 0 auto; padding: 0 12px 20px}
#chat-panel {border: 0 !important; box-shadow: none !important; background: transparent !important}
#chat-panel .placeholder {font-size: 1.55rem; font-weight: 600; color: var(--body-text-color)}
#composer {border-radius: 22px !important; overflow: hidden; border: 1px solid var(--border-color-primary) !important}
#session-list {min-height: 335px; max-height: 42vh; overflow-y: auto; padding: 4px 2px}
#session-list label {height: 43px; max-height: 43px; padding: 9px 10px; border-radius: 10px; cursor: pointer; overflow: hidden; white-space: nowrap; text-overflow: ellipsis}
#voice-submit {min-height: 56px !important; font-size: 1rem !important; font-weight: 650 !important}
#tts-switch {min-height: 44px; display: flex; align-items: center}
#voice-submit button {min-height: 56px !important}
#status-panel {font-size: 0.82em}
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
        (
            str(image_path(image)),
            f"var{getattr(image, 'meta', {}).get('variation')}"
            if getattr(image, "meta", {}).get("variation")
            else f"rev{image.revision}{' 元' if image.revision == 0 else ''}",
        )
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
    siblings = app.sessions.variation_siblings(current)
    if siblings:
        lineage = [image for image in lineage if image.id not in {s.id for s in siblings}] + siblings
        if selected_id is None:
            current = siblings[0]
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

    def stream_turn(session_id: str, events: Iterator[Event], read_aloud: bool = False):
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
            yield base + pending_messages(reasoning, text, status, errors), route_md, gr.update()
        final = render_history(app, session_id)
        if errors and not any("エラー" in str(m.get("content")) for m in final[-1:]):
            final += pending_messages("", "", "", errors)
        audio_path = app.controller.read_last_answer(session_id) if read_aloud else None
        if app.controller.last_tts_error and read_aloud:
            route_md += f" · {app.controller.last_tts_error}"
        yield final, route_md, gr.update(value=audio_path, visible=bool(audio_path))

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
        read_aloud,
        mask_enabled,
        mask_editor,
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
        if mask_enabled and selected_id:
            selected = app.sessions.get_image(selected_id)
            opts.mask_image = mask_from_editor(mask_editor, (selected.width, selected.height))
        events = app.controller.handle(session_id, text, paths, opts)
        for chat, route_md, speech in stream_turn(session_id, events, read_aloud):
            yield (
                chat,
                gr.update(value=None),
                gr.update(value=None),
                route_md,
                session_id,
                gr.update(),
                speech,
            )
        yield (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            session_id,
            gr.update(choices=session_choices(app), value=session_id),
            gr.update(),
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
        read_aloud,
    ):
        if not session_id:
            yield gr.update(), "再生成できるメッセージがありません", gr.update()
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
        yield from stream_turn(session_id, app.controller.regenerate(session_id, opts), read_aloud)

    def refresh_lineage(session_id):
        view = lineage_view(app, session_id)
        selected = app.sessions.get_image(view[2]) if view[2] else None
        return (*view, gr.update(value=str(app.sessions.image_path(selected)) if selected else None))

    def on_gallery_select(session_id, ids, evt: gr.SelectData):
        if not ids or evt.index is None:
            return None, "画像を選択できませんでした", gr.update()
        index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        selected = ids[int(index)]
        return (
            selected,
            lineage_view(app, session_id, selected)[3],
            gr.update(value=str(app.sessions.image_path(app.sessions.get_image(selected)))),
        )

    def on_relative(session_id, selected_id, steps_back):
        if not selected_id:
            return selected_id, f"{steps_back}個前の画像はありません", gr.update()
        lineage = app.sessions.lineage(selected_id)
        if len(lineage) <= steps_back:
            return selected_id, f"{steps_back}個前の画像はありません", gr.update()
        target = lineage[-1 - steps_back].id
        return (
            target,
            lineage_view(app, session_id, target)[3],
            gr.update(value=str(app.sessions.image_path(app.sessions.get_image(target)))),
        )

    def on_restore(session_id, selected_id):
        if not session_id:
            yield [], "画像はまだありません", gr.update()
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

        with gr.Sidebar(open=True, width=300):
            gr.Markdown("### Qwen Studio")
            new_btn = gr.Button("＋ 新しいチャット", variant="primary", size="lg")
            sessions_radio = gr.Radio(
                choices=first,
                value=first[0][1] if first else None,
                label="チャット履歴",
                interactive=True,
                elem_id="session-list",
            )
            delete_btn = gr.Button("このチャットを削除", size="sm")
            with gr.Accordion("画像の系譜・バリエーション", open=False):
                lineage_gallery = gr.Gallery(
                    value=initial_gallery[0], columns=4, height=135, show_label=False, preview=False
                )
                selected_caption = gr.Markdown(initial_gallery[3])
                with gr.Row():
                    previous_btn = gr.Button("1個前", size="sm")
                    two_back_btn = gr.Button("2個前", size="sm")
                    restore_btn = gr.Button("元に戻す", size="sm")
            with gr.Accordion("システム状態", open=False):
                status_md = gr.Markdown(status_markdown(app), elem_id="status-panel")
                with gr.Row():
                    refresh_btn = gr.Button("状態更新", size="sm")
                    unload_btn = gr.Button("モデル解放", size="sm")
            timer = gr.Timer(5.0)

        if app.cfg.share:
            gr.Markdown("⚠️ **share=True: この画面は公開URLでインターネットからアクセス可能です。**")
        with gr.Column(elem_id="conversation-shell"):
            chatbot = gr.Chatbot(
                value=render_history(app, session_state.value),
                height="52vh",
                show_label=False,
                buttons=["copy"],
                placeholder="今日は何を話しましょう？",
                allow_file_downloads=True,
                elem_id="chat-panel",
            )
            route_md = gr.Markdown("", elem_id="route-info")
            textbox = gr.MultimodalTextbox(
                placeholder="メッセージを入力、または＋から画像・PDF・動画を添付",
                file_types=["image", ".pdf", ".wav", ".mp3", ".m4a", ".webm", ".ogg", ".mp4", ".mov", ".mkv"],
                file_count="multiple",
                lines=2,
                max_lines=8,
                show_label=False,
                submit_btn=True,
                autofocus=True,
                elem_id="composer",
            )
            with gr.Row(equal_height=True):
                audio_input = gr.Audio(
                    sources=["microphone", "upload"], type="filepath", label="マイク・音声ファイル", scale=4
                )
                mic_submit_btn = gr.Button(
                    "🎙 音声を送信", size="lg", scale=2, min_width=170, elem_id="voice-submit"
                )
            speech_output = gr.Audio(label="回答の読み上げ", autoplay=True, interactive=False, visible=False)
            with gr.Row():
                thinking = gr.Checkbox(value=app.cfg.thinking_default, label="深く考える", scale=1)
                read_aloud = gr.Checkbox(
                    value=app.cfg.tts, label="🔊 回答を読み上げる", scale=1, elem_id="tts-switch"
                )
            with gr.Row():
                stop_btn = gr.Button("停止", size="sm")
                regen_btn = gr.Button("再生成", size="sm")
                clear_btn = gr.Button("会話をクリア", size="sm")
        with gr.Column(elem_id="settings-shell"):
            with gr.Accordion("詳細設定", open=False):
                mode = gr.Radio(MODE_CHOICES, value=Mode.AUTO.value, label="モード")
                web_search = gr.Radio(
                    [("自動", "auto"), ("常に", "on"), ("オフ", "off")],
                    value=app.cfg.web_search,
                    label="🌐 Web検索（最新情報）",
                )
                content_policy = gr.Radio(
                    [("開放", "open"), ("標準", "standard")],
                    value=app.cfg.content_policy,
                    label="コンテンツ方針",
                )
            with gr.Accordion("画像生成・編集の設定", open=False):
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
            with gr.Accordion("マスクで編集", open=False):
                mask_enabled = gr.Checkbox(label="マスクを使う", value=False)
                mask_editor = gr.ImageEditor(
                    value=str(app.sessions.image_path(app.sessions.get_image(initial_gallery[2])))
                    if initial_gallery[2]
                    else None,
                    type="pil",
                    label="編集する部分を塗る",
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
        submit_inputs = [
            textbox,
            audio_input,
            session_state,
            selected_state,
            *settings,
            read_aloud,
            mask_enabled,
            mask_editor,
        ]
        submit_outputs = [
            chatbot,
            textbox,
            audio_input,
            route_md,
            session_state,
            sessions_radio,
            speech_output,
        ]
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
            [session_state, selected_state, *settings, read_aloud],
            [chatbot, route_md, speech_output],
            concurrency_id="gpu",
            concurrency_limit=1,
        )
        lineage_outputs = [lineage_gallery, gallery_ids, selected_state, selected_caption, mask_editor]
        submit_event.then(refresh_lineage, session_state, lineage_outputs)
        mic_event.then(refresh_lineage, session_state, lineage_outputs)
        regen_event.then(refresh_lineage, session_state, lineage_outputs)
        lineage_gallery.select(
            on_gallery_select, [session_state, gallery_ids], [selected_state, selected_caption, mask_editor]
        )
        previous_btn.click(
            lambda sid, selected: on_relative(sid, selected, 1),
            [session_state, selected_state],
            [selected_state, selected_caption, mask_editor],
        )
        two_back_btn.click(
            lambda sid, selected: on_relative(sid, selected, 2),
            [session_state, selected_state],
            [selected_state, selected_caption, mask_editor],
        )
        restore_event = restore_btn.click(
            on_restore,
            [session_state, selected_state],
            [chatbot, route_md, speech_output],
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
            view = refresh_lineage(sid)
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
        theme=UI_THEME,
        css=CSS,
        **kwargs,
    )
