"""CLOSER OpenAI-compatible HTTP server."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from closer.config import CloserConfig, load_config
from closer.errors import BenchmarkParseError, CloserError, ConfigError
from closer.llm import build_llm_client
from closer.logging import build_run_manifest, create_run_store, setup_stdlib_logging
from closer.orchestrator.engine import CloserEngine
from closer.server.chat import handle_chat
from closer.server.responses import handle_responses


class LazyEngine:
    """Engine proxy so connectivity probes do not require the inner LLM."""

    def __init__(self, config: CloserConfig, store) -> None:
        self.config = config
        self.store = store
        self._engine: CloserEngine | None = None
        self._client = None

    async def run_episode(self, episode):
        if self._engine is None:
            self._client = build_llm_client(self.config)
            self._engine = CloserEngine(self.config, self._client, store=self.store)
        return await self._engine.run_episode(episode)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


def create_app(config: CloserConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    setup_stdlib_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg.assert_not_recursive()
        store = create_run_store(cfg)
        store.write_manifest(build_run_manifest(cfg))
        engine = LazyEngine(cfg, store)
        app.state.config = cfg
        app.state.store = store
        app.state.engine = engine
        try:
            yield
        finally:
            await engine.aclose()

    app = FastAPI(title="MechRank-Harness", version=cfg.project.version, lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": cfg.server_model_id()}

    @app.post("/v1/responses")
    async def responses_endpoint(request: Request) -> JSONResponse:
        _authorize(request, cfg)
        body = await request.json()
        try:
            payload = await handle_responses(request.app.state.engine, body)
        except BenchmarkParseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ConfigError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except CloserError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return JSONResponse(payload)

    @app.post("/v1/chat/completions")
    async def chat_endpoint(request: Request) -> JSONResponse:
        _authorize(request, cfg)
        body = await request.json()
        try:
            payload = await handle_chat(request.app.state.engine, body)
        except BenchmarkParseError as extra:
            raise HTTPException(status_code=400, detail=str(extra)) from extra
        except ConfigError as extra:
            raise HTTPException(status_code=500, detail=str(extra)) from extra
        except CloserError as extra:
            raise HTTPException(status_code=500, detail=str(extra)) from extra
        return JSONResponse(payload)

    return app


def _authorize(request: Request, cfg: CloserConfig) -> None:
    expected = cfg.server_api_key()
    if cfg.server.allow_dummy_api_key and expected in {"", "dummy-local-key"}:
        return
    header = request.headers.get("authorization", "")
    token = header.split(" ", 1)[1] if header.lower().startswith("bearer ") else ""
    if token != expected:
        raise HTTPException(status_code=401, detail="invalid api key")
