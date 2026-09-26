from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

from .config import Settings
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
    ingest.add_argument("--chunk-size", type=int, default=1200)
    ingest.add_argument("--overlap", type=int, default=200)

    ask = commands.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)

    chat = commands.add_parser("chat", help="interactive question loop")
    chat.add_argument("--top-k", type=int, default=5)

    commands.add_parser("status", help="show collection status")
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


def _answer(question: str, top_k: int, store: Store, provider) -> None:
    vector = provider.embed([question])[0]
    matches = store.search(vector, top_k)
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


def run(args: argparse.Namespace) -> None:
    settings = _settings(args)
    store = Store(settings.qdrant_url, settings.collection)
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
    elif args.command == "ingest":
        if not args.path.exists():
            raise ValueError(f"path does not exist: {args.path}")
        result = store.ingest(args.path, provider, args.chunk_size, args.overlap)
        print(f"indexed {result.files} files / {result.chunks} chunks; skipped {result.skipped}")
    elif args.command == "ask":
        _answer(args.question, args.top_k, store, provider)
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
                _answer(question, args.top_k, store, provider)


def main() -> None:
    try:
        run(parser().parse_args())
    except (ValueError, RuntimeError, httpx.HTTPError, ConnectionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

