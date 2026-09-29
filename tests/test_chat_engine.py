from qmc.backends.base import ChatParams
from qmc.backends.llama_server import build_command
from qmc.backends.mock import MockChatModel
from qmc.backends.openai_compat import ThinkTagSplitter, build_payload, parse_sse_lines
from qmc.chat_engine import ChatEngine, ContextImage, ContextMessage, build_messages
from qmc.gpu_manager import PROFILES
from qmc.model_manager import ModelManager


def test_parse_sse_lines():
    lines = [
        b": ping",
        b"",
        b'data: {"a": 1}',
        "data: not-json",
        b'data: {"b": 2}',
        b"data: [DONE]",
        b'data: {"c": 3}',
    ]
    assert list(parse_sse_lines(lines)) == [{"a": 1}, {"b": 2}]


def test_think_tag_splitter_handles_split_tags():
    s = ThinkTagSplitter()
    out_r, out_c = "", ""
    for piece in ["<thi", "nk>reason", "ing</th", "ink>\n\nAnswer"]:
        d = s.feed(piece)
        out_r += d.reasoning
        out_c += d.content
    tail = s.flush()
    out_c += tail.content
    assert out_r == "reasoning"
    assert out_c.strip() == "Answer"


def test_sampling_defaults_follow_model_card():
    assert ChatParams(thinking=False).resolved()["presence_penalty"] == 1.5
    t = ChatParams(thinking=True).resolved()
    assert (t["temperature"], t["top_p"]) == (1.0, 0.95)
    assert ChatParams(thinking=False, temperature=0.2).resolved()["temperature"] == 0.2


def test_payload_sets_enable_thinking():
    p = build_payload("m", [], ChatParams(thinking=False))
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert p["stream"] is True


def test_build_command_contains_verified_flags(tmp_path):
    cmd = build_command(
        tmp_path / "llama-server",
        tmp_path / "m.gguf",
        tmp_path / "mmproj.gguf",
        host="127.0.0.1",
        port=8012,
        ctx_size=16384,
        fit_target_mib=2048,
        api_key="k",
    )
    joined = " ".join(cmd)
    for flag in [
        "--fit on",
        "--fit-target 2048",
        "-c 16384",
        "--jinja",
        "--mmproj",
        "--api-key k",
        "--reasoning-format deepseek",
        "--host 127.0.0.1",
    ]:
        assert flag in joined


def test_build_messages_limits_images(make_png):
    paths = [make_png(f"i{i}.png") for i in range(5)]
    history = [
        ContextMessage("user", f"msg {i}", [ContextImage(str(i), str(p), "upload")])
        for i, p in enumerate(paths)
    ]
    msgs = build_messages(history, max_images=2)
    image_parts = [
        p for m in msgs if isinstance(m["content"], list) for p in m["content"] if p["type"] == "image_url"
    ]
    assert len(image_parts) == 2
    text = str(msgs)
    assert "[画像 #0: upload]" in text  # old images remain as labels


def test_build_messages_assistant_images_become_user_context(make_png):
    p = make_png()
    history = [
        ContextMessage("user", "猫を描いて"),
        ContextMessage("assistant", "生成しました", [ContextImage("a1", str(p), "生成画像")]),
        ContextMessage("user", "これは何色？"),
    ]
    msgs = build_messages(history)
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]  # merged -> alternating
    assert any(part.get("type") == "image_url" for part in msgs[-1]["content"])


def test_extra_images_always_attached(make_png):
    a, b = make_png("a.png"), make_png("b.png")
    history = [ContextMessage("user", "違いは？")]
    msgs = build_messages(
        history, max_images=0, extra_images=[ContextImage("a", str(a), "元"), ContextImage("b", str(b), "今")]
    )
    assert sum(1 for p in msgs[-1]["content"] if p["type"] == "image_url") == 2


def test_engine_streams_via_manager():
    mm = ModelManager(profile=PROFILES["cpu"])
    chat = MockChatModel()
    mm.register(chat)
    engine = ChatEngine(mm)
    out = "".join(d.content for d in engine.stream([ContextMessage("user", "こんにちは")], ChatParams()))
    assert "こんにちは" in out
    assert chat.is_loaded


def test_rewrite_prompt_uses_llm():
    mm = ModelManager(profile=PROFILES["cpu"])
    mm.register(MockChatModel())
    engine = ChatEngine(mm)
    assert engine.rewrite_image_prompt("猫", "generate", []).startswith("[rewritten]")


def test_forced_images_are_not_duplicated(make_png):
    p = make_png()
    img = ContextImage("x", str(p), "生成画像")
    history = [
        ContextMessage("user", "描いて"),
        ContextMessage("assistant", "生成しました", [img]),
        ContextMessage("user", "違いは？"),
    ]
    msgs = build_messages(history, extra_images=[img])
    n = sum(
        1
        for m in msgs
        if isinstance(m["content"], list)
        for part in m["content"]
        if part["type"] == "image_url"
    )
    assert n == 1


def test_parse_appearance_card_tolerates_markdown_and_fullwidth_colon():
    from qmc.chat_engine import card_summary_ja, parse_appearance_card

    raw = (
        "- **NAME**: 松本乱菊\\n* **HAIR：** long wavy strawberry-blonde\\nEYES: blue-gray\\n"
        "**Signature Outfit**: black shihakusho with pink scarf\\nCONFIDENCE: low"
    ).replace("\\n", "\n")
    card = parse_appearance_card(raw)
    assert "HAIR: long wavy strawberry-blonde" in card
    assert "SIGNATURE OUTFIT: black shihakusho with pink scarf" in card
    assert "CONFIDENCE: low" in card  # kept, not discarded
    assert "髪: long wavy strawberry-blonde" in card_summary_ja(card)


def test_parse_appearance_card_rejects_all_unknown():
    from qmc.chat_engine import parse_appearance_card

    assert parse_appearance_card("NAME: X\nHAIR: UNKNOWN\nEYES: 不明") is None
    assert parse_appearance_card("できません") is None
