import pytest

from qmc.router import ImageTarget, Intent, Mode, RouteContext, route

NO_IMG = RouteContext()
UPLOAD = RouteContext(has_uploads=True)
AFTER_GEN = RouteContext(has_session_image=True, turns_since_last_image=0)
OLD_IMG = RouteContext(has_session_image=True, turns_since_last_image=10)


@pytest.mark.parametrize(
    ("text", "ctx", "intent", "target"),
    [
        # spec examples
        ("Kubernetesについて教えて", NO_IMG, Intent.CHAT, ImageTarget.NONE),
        ("これは何？", UPLOAD, Intent.VISION, ImageTarget.UPLOADED),
        ("宇宙服を着た猫を描いて", NO_IMG, Intent.GENERATE, ImageTarget.NONE),
        ("背景を東京の夜景に変更して", UPLOAD, Intent.EDIT, ImageTarget.UPLOADED),
        ("もう少し明るく", AFTER_GEN, Intent.EDIT, ImageTarget.LATEST),
        ("変更前と変更後の違いを教えて", AFTER_GEN, Intent.VISION, ImageTarget.ROOT_AND_LATEST),
        # MVP tests
        ("TerraformとPulumiの違いを教えて", NO_IMG, Intent.CHAT, ImageTarget.NONE),
        ("この画像に何が写っていますか？", UPLOAD, Intent.VISION, ImageTarget.UPLOADED),
        ("東京の夜景を背景にした未来的なデータセンターを生成して", NO_IMG, Intent.GENERATE, ImageTarget.NONE),
        ("もう少し夜を暗くして、ネオンを増やして", AFTER_GEN, Intent.EDIT, ImageTarget.LATEST),
        ("元画像と今の画像の違いを説明して", AFTER_GEN, Intent.VISION, ImageTarget.ROOT_AND_LATEST),
        # more
        ("", UPLOAD, Intent.VISION, ImageTarget.UPLOADED),
        ("Pythonでフィボナッチのコードを生成して", NO_IMG, Intent.CHAT, ImageTarget.NONE),
        ("テストデータを生成して", NO_IMG, Intent.CHAT, ImageTarget.NONE),
        ("富士山のイラストを作って", NO_IMG, Intent.GENERATE, ImageTarget.NONE),
        ("draw a red fox in the snow", NO_IMG, Intent.GENERATE, ImageTarget.NONE),
        ("Generate an image of a lighthouse", NO_IMG, Intent.GENERATE, ImageTarget.NONE),
        ("make it brighter", AFTER_GEN, Intent.EDIT, ImageTarget.LATEST),
        ("この画像の背景は何色？", AFTER_GEN, Intent.VISION, ImageTarget.LATEST),
        ("さっきの画像に写っているものを説明して", AFTER_GEN, Intent.VISION, ImageTarget.LATEST),
        ("前の画像との違いは？", AFTER_GEN, Intent.VISION, ImageTarget.PARENT_AND_LATEST),
        ("この画像をもとに水彩画風のイラストを生成して", UPLOAD, Intent.EDIT, ImageTarget.UPLOADED),
        ("この2枚の違いは？", UPLOAD, Intent.VISION, ImageTarget.UPLOADED),
        # an edit-sounding sentence long after the last image is just chat
        ("もう少し詳しく教えて", OLD_IMG, Intent.CHAT, ImageTarget.NONE),
        ("明るく元気な挨拶文を考えて", NO_IMG, Intent.CHAT, ImageTarget.NONE),
    ],
)
def test_auto_routing(text, ctx, intent, target):
    d = route(text, ctx)
    assert (d.intent, d.target) == (intent, target), d.reason


def test_compare_flag():
    assert route("元画像と今の画像の違いを説明して", AFTER_GEN).compare


@pytest.mark.parametrize(
    ("mode", "ctx", "intent"),
    [
        (Mode.CHAT, UPLOAD, Intent.CHAT),
        (Mode.VISION, UPLOAD, Intent.VISION),
        (Mode.VISION, AFTER_GEN, Intent.VISION),
        (Mode.GENERATE, NO_IMG, Intent.GENERATE),
        (Mode.GENERATE, UPLOAD, Intent.EDIT),
        (Mode.EDIT, AFTER_GEN, Intent.EDIT),
    ],
)
def test_manual_modes(mode, ctx, intent):
    assert route("何か", ctx, mode).intent == intent


def test_manual_vision_without_image_falls_back_with_warning():
    d = route("これは何", NO_IMG, "vision")
    assert d.intent == Intent.CHAT
    assert d.warnings


@pytest.mark.parametrize(
    ("text", "ctx", "expected"),
    [
        ("アニメ Bleach の松本乱菊さんを描いて。外見を検索して別人にしないで", NO_IMG, True),
        ("松本乱菊を描いて", NO_IMG, True),
        ("宇宙服を着た猫を描いて", NO_IMG, False),
        ("背景を夜景にして", UPLOAD, False),
        ("このキャラの外見を検索して同じ顔で描いて", UPLOAD, True),
        ("背景を検索して風景を描いて", NO_IMG, False),
        ("ブリーチの松本乱菊の画像を生成して", NO_IMG, True),
        ("ワンピースのルフィのイラストを描いて", NO_IMG, True),
        ("夕焼けの空の画像を生成して", NO_IMG, False),
        ("猫の画像を生成して", NO_IMG, False),
        ("ルフィを描いて", NO_IMG, True),
        ("エレン・イェーガーの画像を生成して", NO_IMG, True),
        ("ドラゴンを描いて", NO_IMG, False),
        ("成人の肖像を描いて", NO_IMG, False),
    ],
)
def test_appearance_search_route(text, ctx, expected):
    decision = route(text, ctx)
    assert decision.intent in (Intent.GENERATE, Intent.EDIT)
    assert decision.search_appearance is expected


def test_manual_generation_searches_named_character():
    decision = route("松本乱菊を描いて", NO_IMG, Mode.GENERATE)
    assert decision.intent is Intent.GENERATE and decision.search_appearance


def test_two_references_skip_implicit_search():
    ctx = RouteContext(has_uploads=True, upload_count=2)
    assert not route("松本乱菊さんを描いて", ctx).search_appearance


@pytest.mark.parametrize(
    ("text", "also_answer"),
    [
        ("ブリーチの松本乱菊について教えて。画像も生成して", True),
        ("乱菊とは？イラストも描いて", True),
        ("ブリーチの松本乱菊の画像を生成して", False),
        ("猫の画像を生成して", False),
    ],
)
def test_compound_generate_and_explain(text, also_answer):
    decision = route(text, NO_IMG)
    assert decision.intent is Intent.GENERATE and decision.also_answer is also_answer
    if also_answer:
        assert decision.search_appearance
