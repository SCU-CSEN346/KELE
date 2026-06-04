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
import transformers

from src.project.kele import create_system
from src.project.socratic_teaching_system import SocraticTeachingSystem

# flash-linear-attention and causal-conv1d are CUDA/ROCm Triton kernels — not
# available on macOS or CPU-only machines. The torch fallback is fine for
# inference on a 0.8B classifier; suppress the install-nag noise.
transformers.logging.set_verbosity_error()

_system = create_system(
    debug=True,
    experiment=os.environ.get("EXPERIMENT", "gemma4-31b-online"),
    bert_consultant=os.environ.get("BERT_CKPT", "results/state-clf-qwen3.5-0.8b-lora-wandb/final"),
)


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("system", _system)
    await cl.Message(
        content="Hi! I'm MELE, your personal tutor, and I use the Socratic teaching method. Ask me a question to get started."
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    system: SocraticTeachingSystem = cl.user_session.get("system")  # type: ignore[assignment]
    response = await asyncio.to_thread(system.process_student_input, message.content)
    await cl.Message(content=response).send()
