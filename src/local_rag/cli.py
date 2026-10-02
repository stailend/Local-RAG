from __future__ import annotations

import argparse
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import httpx

from .config import Settings
from .evaluation import evaluate, load_cases
from .providers import provider_from_settings
from .store import Store


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="local-rag", description="RAG over local files")
    result.add_argument("--qdrant-url")
    result.add_argument("--collection")
    result.add_argument("--provider", choices=["ollama", "openai-compatible"])
    result.add_argument("--base-url")
    result.add_argument("--api-key")
    result.add_argument("--chat-model")
    result.add_argument("--embedding-model")
    commands = result.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="index a file or directory")
    ingest.add_argument("path", type=Path)
    sync = commands.add_parser("sync", help="mirror a directory into the collection")
    sync.add_argument("path", type=Path)
    for command in (ingest, sync):
        command.add_argument("--chunk-size", type=int, default=1200)
        command.add_argument("--overlap", type=int, default=200)

    ask = commands.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    chat = commands.add_parser("chat", help="interactive question loop")
    search = commands.add_parser("search", help="inspect retrieved chunks")
    search.add_argument("question")
    search.add_argument("--explain", action="store_true")
    for command in (ask, chat, search):
        command.add_argument("--top-k", type=int, default=5)
        command.add_argument("--candidates", type=int, default=20)
        command.add_argument("--no-rerank", action="store_true")
        command.add_argument(
            "--rerank-model",
            default=os.getenv("LOCAL_RAG_RERANK_MODEL", "ms-marco-TinyBERT-L-2-v2"),
        )

    evaluation = commands.add_parser("evaluate", help="benchmark retrieval")
    evaluation.add_argument("dataset", type=Path)
    evaluation.add_argument("--top-k", type=int, default=5)
    evaluation.add_argument("--candidates", type=int, default=20)
    evaluation.add_argument("--json", action="store_true")
    evaluation.add_argument(
        "--rerank-model",
        default=os.getenv("LOCAL_RAG_RERANK_MODEL", "ms-marco-TinyBERT-L-2-v2"),
    )

    commands.add_parser("status", help="show collection status")
    serve = commands.add_parser("serve", help="start the local web UI and API")
    serve.add_argument("--host", default=os.getenv("LOCAL_RAG_HOST", "127.0.0.1"))
    serve.add_argument("--port", type=int, default=int(os.getenv("LOCAL_RAG_PORT", "8000")))
    delete = commands.add_parser("delete", help="delete the collection")
    delete.add_argument("--yes", action="store_true")
    commands.add_parser("doctor", help="check database and model provider")
    return result


def _settings(args: argparse.Namespace) -> Settings:
    return Settings.from_env().override(
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        chat_model=args.chat_model,
        embedding_model=args.embedding_model,
    )


def _merge_matches(matches: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for match in matches:
        key = (str(match.get("source")), str(match.get("locator")))
        groups.setdefault(key, []).append(match)

    merged = []
    for group in groups.values():
        group.sort(key=lambda item: int(item.get("chunk", 0)))
        for item in group:
            current = dict(item)
            if merged and (
                merged[-1].get("source") == current.get("source")
                and merged[-1].get("locator") == current.get("locator")
                and int(current.get("chunk", 0)) == int(merged[-1].get("last_chunk", 0)) + 1
            ):
                previous = str(merged[-1]["text"])
                following = str(current["text"])
                overlap = next(
                    (
                        size
                        for size in range(min(len(previous), len(following)), 0, -1)
                        if previous.endswith(following[:size])
                    ),
                    0,
                )
                merged[-1]["text"] = previous + following[overlap:]
                merged[-1]["last_chunk"] = current.get("chunk", 0)
            else:
                current["last_chunk"] = current.get("chunk", 0)
                merged.append(current)
    return merged


@lru_cache(maxsize=2)
def _ranker(model_name: str):
    from flashrank import Ranker

    cache = os.getenv(
        "LOCAL_RAG_MODEL_CACHE", str(Path.home() / ".cache" / "local-rag")
    )
    return Ranker(model_name=model_name, cache_dir=cache, log_level="WARNING")


def _rerank(
    question: str,
    matches: list[dict],
    limit: int,
    model_name: str,
    ranker=None,
) -> list[dict]:
    from flashrank import RerankRequest

    if not matches:
        return []
    passages = [
        {"id": number, "text": match["text"], "meta": match}
        for number, match in enumerate(matches)
    ]
    ranked = (ranker or _ranker(model_name)).rerank(
        RerankRequest(query=question, passages=passages)
    )
    return [
        {**item["meta"], "_rerank_score": float(item["score"])}
        for item in ranked[:limit]
    ]


def _retrieve(question: str, args, store: Store, provider) -> list[dict]:
    vector = provider.embed([question])[0]
    matches = store.search(
        question,
        vector,
        args.candidates,
        args.command == "search" and args.explain,
    )
    if not args.no_rerank:
        matches = _rerank(
            question, matches, args.top_k, args.rerank_model
        )
    else:
        matches = matches[: args.top_k]
    return matches


def _answer(question: str, args, store: Store, provider) -> None:
    matches = _merge_matches(_retrieve(question, args, store, provider))
    if not matches:
        print("No indexed context found.")
        return
    context = "\n\n".join(
        f"[{number}] {item.get('source')} ({item.get('locator')})\n{item.get('text')}"
        for number, item in enumerate(matches, 1)
    )
    print(provider.answer(question, context))
    print("\nSources:")
    for number, item in enumerate(matches, 1):
        print(f"[{number}] {item.get('source')} ({item.get('locator')})")


def _search(question: str, args, store: Store, provider) -> None:
    matches = _retrieve(question, args, store, provider)
    for number, item in enumerate(matches, 1):
        print(f"{number}. {item.get('source')} ({item.get('locator')})")
        if args.explain:
            print(
                f"   dense_rank={item.get('_dense_rank')} "
                f"sparse_rank={item.get('_sparse_rank')} "
                f"fused_score={item.get('_fused_score', 0):.4f} "
                f"rerank_score={item.get('_rerank_score', 0):.4f}"
            )
        print(f"   {str(item.get('text'))[:300]}\n")


def _evaluate(args, store: Store, provider) -> None:
    cases = load_cases(args.dataset)
    vectors = {
        str(case["question"]): provider.embed([str(case["question"])])[0]
        for case in cases
    }

    def retrieve(profile: str, question: str) -> list[dict]:
        vector = vectors[question]
        if profile == "dense":
            return store.search(question, vector, args.top_k, mode="dense")
        if profile == "hybrid":
            return store.search(
                question,
                vector,
                args.top_k,
                mode="hybrid",
                candidates=args.candidates,
            )
        matches = store.search(
            question,
            vector,
            args.candidates,
            mode="hybrid",
            candidates=args.candidates,
        )
        return _rerank(
            question, matches, args.top_k, args.rerank_model
        )

    rows = evaluate(
        cases, ["dense", "hybrid", "hybrid+reranker"], retrieve
    )
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(f"{'profile':<20} {'cases':>5} {'recall@k':>10} {'mrr':>8} {'median_ms':>11}")
    for row in rows:
        print(
            f"{row['profile']:<20} {row['cases']:>5} "
            f"{row['recall']:>9.1%} {row['mrr']:>8.3f} "
            f"{row['median_ms']:>11.1f}"
        )


def run(args: argparse.Namespace) -> None:
    settings = _settings(args)
    store = Store(settings.qdrant_url, settings.collection)
    if args.command in {"ask", "chat", "search", "evaluate"} and (
        args.top_k < 1 or args.candidates < args.top_k
    ):
        raise ValueError("candidates must be greater than or equal to top-k >= 1")
    if args.command == "status":
        count, status = store.status()
        print(f"collection={settings.collection} points={count} status={status}")
        return
    if args.command == "delete":
        if not args.yes:
            raise ValueError("refusing to delete without --yes")
        store.delete()
        print(f"deleted collection {settings.collection}")
        return

    provider = provider_from_settings(settings)
    if args.command == "doctor":
        count, status = store.status()
        provider.healthcheck()
        print(f"ok: qdrant={status} points={count}, provider={settings.provider}")
    elif args.command == "serve":
        from .api import serve

        serve(settings, args.host, args.port)
    elif args.command == "ingest":
        if not args.path.exists():
            raise ValueError(f"path does not exist: {args.path}")
        result = store.ingest(args.path, provider, args.chunk_size, args.overlap)
        print(f"indexed {result.files} files / {result.chunks} chunks; skipped {result.skipped}")
    elif args.command == "sync":
        if not args.path.is_dir():
            raise ValueError(f"sync path must be a directory: {args.path}")
        result = store.sync(args.path, provider, args.chunk_size, args.overlap)
        print(
            f"added={result.added} updated={result.updated} "
            f"deleted={result.deleted} unchanged={result.unchanged} "
            f"skipped={result.skipped} chunks={result.chunks}"
        )
    elif args.command == "ask":
        _answer(args.question, args, store, provider)
    elif args.command == "search":
        _search(args.question, args, store, provider)
    elif args.command == "evaluate":
        if not args.dataset.is_file():
            raise ValueError(f"dataset does not exist: {args.dataset}")
        _evaluate(args, store, provider)
    elif args.command == "chat":
        while True:
            try:
                question = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if question in {"exit", "quit"}:
                return
            if question:
                _answer(question, args, store, provider)


def main() -> None:
    try:
        run(parser().parse_args())
    except (ValueError, RuntimeError, httpx.HTTPError, ConnectionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
