"""Command-line entry point: run any pipeline stage or ask the agent a question without the API."""

from __future__ import annotations

import argparse
import json
import logging

from recon_rag import services
from recon_rag.db import init_db


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="recon-rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="generate synthetic multi-source data")
    g.add_argument("--orders", type=int, default=2000)
    g.add_argument("--seed", type=int, default=42)
    i = sub.add_parser("ingest", help="ingest raw batches, detect drift, normalize")
    i.add_argument("--llm-resolver", action="store_true")
    sub.add_parser("reconcile", help="detect cross-source discrepancies")
    sub.add_parser("index", help="chunk + embed into pgvector")
    pl = sub.add_parser("pipeline", help="generate (optional) + ingest + reconcile + index")
    pl.add_argument("--generate", action="store_true")
    pl.add_argument("--orders", type=int, default=2000)
    q = sub.add_parser("ask", help="ask the agent a question")
    q.add_argument("question")
    e = sub.add_parser("eval", help="run the LLM-as-a-judge benchmark")
    e.add_argument("--sample", type=int, default=None)
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if a.cmd != "generate":
        init_db()

    if a.cmd == "generate":
        out = services.generate_data(a.orders, a.seed)
    elif a.cmd == "ingest":
        out = services.ingest(a.llm_resolver)
    elif a.cmd == "reconcile":
        out = services.reconcile()
    elif a.cmd == "index":
        out = services.index()
    elif a.cmd == "pipeline":
        out = services.run_pipeline(a.generate, a.orders)
    elif a.cmd == "ask":
        from recon_rag.agent.loop import Agent
        from recon_rag.agent.tools import ToolExecutor
        from recon_rag.config import get_settings
        from recon_rag.db import session_scope
        from recon_rag.index.embedder import get_embedder
        from recon_rag.llm.client import get_llm

        with session_scope() as s:
            agent = Agent(get_llm("agent"), ToolExecutor(s, get_embedder()), get_settings().agent_max_steps)
            out = agent.answer(a.question).to_dict(include_evidence=False)
    else:
        from recon_rag.eval import runner

        run_id = runner.start_run()
        out = {"run_id": run_id, "metrics": runner.execute_run(run_id, sample_size=a.sample)}
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
