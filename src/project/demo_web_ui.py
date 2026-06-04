"""Chainlit web UI for the MELE live demo.

Wraps the core SocraticTeachingSystem with a browser-based chat interface.
Terminal receives all raw debug output (classifier state, evaluation, action);
the browser shows the polished MELE ↔ student conversation.

Launch via serve_demo_top_performer.sh (WEBUI=1) or directly:
    EXPERIMENT=gemma4-31b-online BERT_CKPT=results/.../final \\
        uv run chainlit run src/project/demo_web_ui.py --host 0.0.0.0 --port 8000
"""

import asyncio
import os

import chainlit as cl

from src.project.kele import create_system


@cl.on_chat_start
async def on_chat_start() -> None:
    experiment = os.environ.get("EXPERIMENT", "gemma4-31b-online")
    bert_ckpt = os.environ.get("BERT_CKPT", "results/state-clf-qwen3.5-0.8b-lora-wandb/final")

    system = await asyncio.to_thread(
        create_system,
        debug=True,
        experiment=experiment,
        bert_consultant=bert_ckpt,
    )
    cl.user_session.set("system", system)
    await cl.Message(
        content="Hi! I'm MELE, your Socratic math tutor. Ask me a question to get started."
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    system = cl.user_session.get("system")
    response = await asyncio.to_thread(system.process_student_input, message.content)
    await cl.Message(content=response).send()
