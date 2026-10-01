"""Small chat/vision interface for the Q8-only Colab notebook."""

from __future__ import annotations

import time

import gradio as gr

from .controller import TurnOptions
from .gpu_manager import memory_snapshot


def _history(app, sid):
    if not sid or not app.sessions.session_exists(sid):
        return []
    out = []
    for message in app.sessions.get_messages(sid):
        parts = ([message.content] if message.content else []) + [
            {"path": str(app.sessions.image_path(image)), "alt_text": image.caption}
            for image in message.images
        ]
        if parts:
            out.append({"role": message.role, "content": parts})
    return out


def _choices(app):
    return [(s["title"] or "(untitled)", s["id"]) for s in app.sessions.list_sessions()]


def build_ui(app) -> gr.Blocks:
    if not app.cfg.chat_only:
        raise ValueError("chat UI requires chat_only=True")

    def status():
        loaded = ", ".join(app.manager.status()["loaded"]) or "none"
        return (f"**GPU:** {app.gpu.name} �E **VRAM:** {memory_snapshot().summary()} �E "
                f"**Loaded:** {loaded} �E **Search:** {app.search_label}")

    def choose(sid):
        return sid, _history(app, sid)

    def new_session():
        sid = app.sessions.create_session()
        return sid, [], gr.update(choices=_choices(app), value=sid)

    def delete_session(sid):
        if sid:
            app.sessions.delete_session(sid)
        choices = _choices(app)
        next_sid = choices[0][1] if choices else None
        return next_sid, _history(app, next_sid), gr.update(choices=choices, value=next_sid)

    def stream(events, sid):
        base = _history(app, sid)
        answer = ""
        note = ""
        last = 0.0
        for event in events:
            if event.kind == "route":
                base = _history(app, sid)
            elif event.kind == "text":
                answer += str(event.data)
            elif event.kind in ("status", "error"):
                note = str(event.data)
            elif event.kind == "done":
                break
            now = time.time()
            if event.kind == "text" and now - last < 0.05:
                continue
            last = now
            pending = [{"role": "assistant", "content": answer or note}] if answer or note else []
            yield base + pending
        yield _history(app, sid)

    def submit(message, audio, sid, thinking, search):
        message = message or {}
        text = message.get("text", "")
        files = [f if isinstance(f, str) else f.get("path") for f in message.get("files", [])]
        if audio:
            files.append(audio)
        if not sid or not app.sessions.session_exists(sid):
            sid = app.sessions.create_session()
        options = TurnOptions(thinking=thinking, web_search=search)
        for view in stream(app.controller.handle(sid, text, [f for f in files if f], options), sid):
            yield view, gr.update(value=None), gr.update(value=None), sid, gr.update()
        yield gr.update(), gr.update(), gr.update(), sid, gr.update(choices=_choices(app), value=sid)

    def regenerate(sid, thinking, search):
        if sid:
            yield from stream(app.controller.regenerate(sid, TurnOptions(thinking=thinking, web_search=search)), sid)

    def release():
        app.manager.unload_all()
        return status()

    with gr.Blocks(title="Qwen Q8 Chat") as demo:
        first = _choices(app)
        sid = gr.State(first[0][1] if first else None)
        with gr.Sidebar(open=True, width=260):
            gr.Markdown("## Qwen Q8 Chat")
            new = gr.Button("New chat")
            sessions = gr.Radio(choices=first, value=sid.value, label="History")
            delete = gr.Button("Delete chat")
            thinking = gr.Checkbox(label="Thinking", value=app.cfg.thinking_default)
            search = gr.Radio([("Always", "on"), ("Auto", "auto"), ("Off", "off")],
                              value=app.cfg.web_search, label="Web search")
            release_btn = gr.Button("Release GPU model")
            status_md = gr.Markdown(status())
        chatbot = gr.Chatbot(value=_history(app, sid.value), type="messages", height=550)
        composer = gr.MultimodalTextbox(file_types=["image", ".pdf", ".mp4", ".mov", ".mkv"],
                                         placeholder="Message or attach an image�c", show_label=False)
        with gr.Row():
            mic = gr.Audio(sources=["microphone"], type="filepath", label="Voice input")
            stop = gr.Button("Stop", size="sm")
            retry = gr.Button("Regenerate", size="sm")
        submit_event = composer.submit(submit, [composer, mic, sid, thinking, search],
                                       [chatbot, composer, mic, sid, sessions],
                                       concurrency_id="gpu", concurrency_limit=1)
        retry.click(regenerate, [sid, thinking, search], chatbot,
                    concurrency_id="gpu", concurrency_limit=1)
        stop.click(app.controller.cancel, None, None, queue=False)
        release_btn.click(release, None, status_md, concurrency_id="gpu", concurrency_limit=1)
        new.click(new_session, None, [sid, chatbot, sessions])
        sessions.input(choose, sessions, [sid, chatbot])
        delete.click(delete_session, sid, [sid, chatbot, sessions])
        submit_event.then(status, None, status_md)
    return demo


def launch(app, demo=None, **kwargs):
    demo = demo or build_ui(app)
    auth = (app.cfg.auth_user, app.cfg.auth_password) if app.cfg.auth_user and app.cfg.auth_password else None
    demo.queue(default_concurrency_limit=4)
    return demo.launch(server_name=app.cfg.server_host, server_port=app.cfg.server_port,
                       share=app.cfg.share, auth=auth,
                       allowed_paths=[str(app.cfg.data_dir.resolve())], **kwargs)
