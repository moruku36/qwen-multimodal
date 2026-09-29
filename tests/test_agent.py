import json

from qmc.agent import GitHubReader, ResearchAgent, find_github_repos, needs_agent, parse_action
from qmc.controller import TurnOptions
from qmc.image_engine import ImageOptions
from tests.test_controller import app, kinds, run  # noqa: F401


class FakeGitHub(GitHubReader):
    def __init__(self):
        super().__init__()
        self.reads = []

    def default_branch(self, repo):
        return "main"

    def tree(self, repo, ref=None):
        return [("README.md", 100), ("src/main.py", 2000), ("tests/test_main.py", 500)]

    def read(self, repo, path, ref=None):
        self.reads.append(path)
        return f"# contents of {path}\n" + "x = 1\n" * 50


def scripted(*actions):
    it = iter(actions)
    return lambda prompt: next(it, json.dumps({"tool": "finish"}))


def test_find_repos_and_trigger():
    text = "https://github.com/moruku36/mattermost-local-sandbox このレポジトリを読んで https://github.com/o/r/blob/main/a.py"
    assert find_github_repos(text) == ["moruku36/mattermost-local-sandbox", "o/r"]
    assert needs_agent(text) and not needs_agent("こんにちは") and not needs_agent(text, "off")
    assert needs_agent("こんにちは", "on")


def test_parse_action_tolerates_prose():
    assert parse_action('Sure!\n```json\n{"tool":"finish"}\n```') == {"tool": "finish"}
    assert parse_action("no json") is None and parse_action("{bad") is None


def test_agent_reads_files_and_dedupes():
    gh = FakeGitHub()
    agent = ResearchAgent(
        scripted(
            '{"tool":"github_tree","repo":"o/r"}',
            '{"tool":"github_read","repo":"o/r","path":"src/main.py"}',
            '{"tool":"github_read","repo":"o/r","path":"src/main.py"}',  # duplicate
            '{"tool":"github_read","repo":"o/r","path":"missing.py","start":0}',
            "garbage",
            '{"tool":"finish"}',
        ),
        gh,
        max_steps=10,
    )
    items = list(agent.run("改善点を教えて", ["o/r"]))
    result = items[-1]
    assert result.finished and gh.reads.count("src/main.py") == 1
    assert any("Already done" in o.result for o in result.observations)
    assert "src/main.py" in agent.evidence(result)
    assert "https://github.com/o/r/blob/main/src/main.py" in result.sources_md()


def test_agent_respects_step_budget_and_cancel():
    import threading

    gh = FakeGitHub()
    calls = []

    def endless(prompt):
        calls.append(1)
        return json.dumps({"tool": "github_read", "repo": "o/r", "path": f"f{len(calls)}.py"})

    agent = ResearchAgent(endless, gh, max_steps=3)
    result = list(agent.run("x", ["o/r"]))[-1]
    assert len(calls) == 3 and not result.finished
    cancel = threading.Event()
    cancel.set()
    assert list(ResearchAgent(endless, gh).run("x", [], cancel))[-1].observations == []


def test_controller_uses_agent_for_github_url(app, monkeypatch):  # noqa: F811
    gh = FakeGitHub()
    app.controller.github = gh
    replies = iter(
        [
            '{"tool":"github_tree","repo":"o/r"}',
            '{"tool":"github_read","repo":"o/r","path":"src/main.py"}',
            '{"tool":"finish"}',
        ]
    )
    monkeypatch.setattr(app.controller.chat, "complete_text", lambda prompt, max_tokens=400: next(replies))
    sid = app.sessions.create_session()
    events = run(
        app,
        sid,
        "https://github.com/o/r を読んで改善点を教えて",
        options=TurnOptions(image=ImageOptions(steps=1), web_search="off"),
    )
    assert gh.reads == ["src/main.py"]
    text = "".join(kinds(events, "text"))
    assert "調査で読んだもの" in text and "src/main.py" in text
    assert any("調査完了" in str(e.data) for e in events if e.kind == "status")
    system = app.manager.get("chat").last_messages[0]["content"]
    assert "調査ログ" in system and "contents of src/main.py" in system


def test_agent_off_skips_research(app):  # noqa: F811
    sid = app.sessions.create_session()
    events = run(
        app,
        sid,
        "https://github.com/o/r を読んで",
        options=TurnOptions(image=ImageOptions(steps=1), web_search="off", agent="off"),
    )
    assert "調査で読んだもの" not in "".join(kinds(events, "text"))
