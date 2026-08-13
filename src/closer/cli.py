"""CLOSER CLI entry points."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
import uvicorn

from closer.benchmark.pgllm_parser import parse_pgllm_prompt
from closer.config import load_config
from closer.llm import build_llm_client
from closer.logging import build_run_manifest, create_run_store, dump_json, setup_stdlib_logging
from closer.orchestrator.engine import CloserEngine

run_app = typer.Typer(add_completion=False)


@run_app.callback(invoke_without_command=True)
def closer_run(
    input_path: Path = typer.Option(..., "--input"),
    config_path: Path | None = typer.Option(None, "--config"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    setup_stdlib_logging()
    cfg = load_config(config_path)
    store = create_run_store(cfg)
    store.write_manifest(build_run_manifest(cfg))
    raw = input_path.read_text(encoding="utf-8")
    if input_path.suffix == ".json":
        from closer.schemas.episode import ProteinEpisode

        episode = ProteinEpisode.model_validate_json(raw)
    else:
        episode = parse_pgllm_prompt(raw)

    async def _go():
        client = build_llm_client(cfg)
        try:
            engine = CloserEngine(cfg, client, store=store)
            return await engine.run_episode(episode)
        finally:
            await client.aclose()

    result = asyncio.run(_go())
    payload = result.model_dump(mode="json")
    if output:
        dump_json(output, payload)
    typer.echo(payload["ranking"])


def run_main() -> None:
    run_app()


server_app = typer.Typer(add_completion=False)


@server_app.callback(invoke_without_command=True)
def closer_server(
    config: Path | None = typer.Option(None, "--config"),
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
) -> None:
    setup_stdlib_logging()
    cfg = load_config(config)
    cfg.assert_not_recursive(host=host, port=port)
    bind_host = host or cfg.server_host()
    bind_port = port or cfg.server_port()
    from closer.server.app import create_app

    uvicorn.run(create_app(cfg), host=bind_host, port=bind_port)


def server_main() -> None:
    server_app()


eval_app = typer.Typer(add_completion=False)


@eval_app.callback(invoke_without_command=True)
def closer_eval_dev(
    dataset: Path = typer.Option(..., "--dataset"),
    config: Path | None = typer.Option(None, "--config"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    setup_stdlib_logging()
    cfg = load_config(config)
    store = create_run_store(cfg)
    store.write_manifest(build_run_manifest(cfg, extra={"mode": "dev-eval"}))

    async def _go():
        from closer.eval.dev import evaluate_dev

        client = build_llm_client(cfg)
        try:
            engine = CloserEngine(cfg, client, store=store)
            return await evaluate_dev(engine, dataset, output)
        finally:
            await client.aclose()

    summary = asyncio.run(_go())
    typer.echo(summary)


def eval_dev_main() -> None:
    eval_app()
