import asyncio
import importlib
import sys

import pytest

pytest.importorskip("chainlit")

import src.project.kele as kele

MODULE_NAME = "src.project.demo.demo_web_ui"


class FakeSystem:
    def __init__(self, reply="Socratic reply"):
        self.reply = reply
        self.inputs = []

    def process_student_input(self, text):
        self.inputs.append(text)
        return self.reply


class FakeUserSession:
    def __init__(self):
        self.store = {}

    def set(self, key, value):
        self.store[key] = value

    def get(self, key, default=None):
        return self.store.get(key, default)


class FakeMessage:
    sent = []

    def __init__(self, content):
        self.content = content

    async def send(self):
        FakeMessage.sent.append(self.content)
        return self


@pytest.fixture
def load_demo(monkeypatch):
    """Import the demo module with create_system stubbed out.

    The module instantiates the (heavy) teaching system at import time, so the
    factory must be patched before the import and the module dropped from the
    cache so each test re-runs module-level code.
    """
    created = []

    def _load(env=None, reply="Socratic reply"):
        for key in ("EXPERIMENT", "BERT_CKPT"):
            monkeypatch.delenv(key, raising=False)
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)

        captured = {}

        def fake_create_system(**kwargs):
            captured.update(kwargs)
            system = FakeSystem(reply=reply)
            created.append(system)
            return system

        monkeypatch.setattr(kele, "create_system", fake_create_system)
        sys.modules.pop(MODULE_NAME, None)
        module = importlib.import_module(MODULE_NAME)
        return module, captured

    FakeMessage.sent = []
    yield _load
    sys.modules.pop(MODULE_NAME, None)


def test_set_starters_returns_four_distinct_starters(load_demo):
    module, _ = load_demo()

    starters = asyncio.run(module.set_starters(user=None, language="en"))

    assert [s.label for s in starters] == [
        "Factor a quadratic",
        "Stuck on fractions",
        "Pythagorean theorem",
        "Why neg × neg = pos?",
    ]
    assert all(s.message for s in starters)


def test_module_preloads_system_with_env_defaults(load_demo):
    module, captured = load_demo()

    assert module._system is not None
    assert captured == {
        "debug": True,
        "experiment": "gemma4-31b-online",
        "bert_consultant": "results/state-clf-qwen3.5-0.8b-lora-wandb/final",
    }


def test_module_honors_experiment_and_bert_ckpt_env(load_demo):
    _, captured = load_demo(env={"EXPERIMENT": "custom-exp", "BERT_CKPT": "results/custom/final"})

    assert captured["experiment"] == "custom-exp"
    assert captured["bert_consultant"] == "results/custom/final"
    assert captured["debug"] is True


def test_on_chat_start_stores_preloaded_system(load_demo, monkeypatch):
    module, _ = load_demo()
    session = FakeUserSession()
    monkeypatch.setattr(module.cl, "user_session", session)

    asyncio.run(module.on_chat_start())

    assert session.get("system") is module._system


def test_on_message_processes_input_and_sends_response(load_demo, monkeypatch):
    module, _ = load_demo()
    session = FakeUserSession()
    system = FakeSystem(reply="What do you notice about the factors?")
    session.set("system", system)
    monkeypatch.setattr(module.cl, "user_session", session)
    monkeypatch.setattr(module.cl, "Message", FakeMessage)

    incoming = FakeMessage("Help me factor x² + 5x + 6.")
    asyncio.run(module.on_message(incoming))

    assert system.inputs == ["Help me factor x² + 5x + 6."]
    assert FakeMessage.sent == ["What do you notice about the factors?"]


def test_on_message_runs_blocking_call_off_the_event_loop(load_demo, monkeypatch):
    """process_student_input is synchronous and is dispatched via
    asyncio.to_thread, so it must run on a worker thread, not the loop thread."""
    module, _ = load_demo()
    session = FakeUserSession()
    monkeypatch.setattr(module.cl, "user_session", session)
    monkeypatch.setattr(module.cl, "Message", FakeMessage)

    main_thread = []
    worker_threads = []

    class ThreadProbeSystem(FakeSystem):
        def process_student_input(self, text):
            import threading

            worker_threads.append(threading.current_thread())
            return super().process_student_input(text)

    async def run():
        import threading

        main_thread.append(threading.current_thread())
        session.set("system", ThreadProbeSystem())
        await module.on_message(FakeMessage("hi"))

    asyncio.run(run())

    assert worker_threads and worker_threads[0] is not main_thread[0]
