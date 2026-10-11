"""Command-line auditor.  Works on a log file alone; no engine required."""

from __future__ import annotations

import argparse
import json
import sys

from .audit import AuditError, Auditor
from .log import AuditLog


def _print(obj) -> None:
    json.dump(obj, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="stare-decisis", description="Audit a decision reasoning log.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("verify", help="check the hash chain")
    s.add_argument("db")

    s = sub.add_parser("history", help="every record for one decision")
    s.add_argument("db")
    s.add_argument("decision")

    s = sub.add_parser("as-made", help="reconstruct an evaluation as it was made")
    s.add_argument("db")
    s.add_argument("decision")
    s.add_argument("eval_id")

    s = sub.add_parser("trail", help="reasoning forms behind a selected option")
    s.add_argument("db")
    s.add_argument("decision")
    s.add_argument("eval_id")

    s = sub.add_parser("impact", help="what a node fed")
    s.add_argument("db")
    s.add_argument("node")

    s = sub.add_parser("clusters", help="group retractions by provenance")
    s.add_argument("db")
    s.add_argument("--reason", default="model_hallucination")
    s.add_argument("--key", default="prompt_hash", help="prompt_hash | model | source")

    s = sub.add_parser("exposure", help="what reversing a decision would break, and how that grew")
    s.add_argument("db")
    s.add_argument("decision")
    s.add_argument("--as-of", type=int, default=None, help="log sequence number")

    s = sub.add_parser("amortization", help="regret vs reversal cost over the decision's life")
    s.add_argument("db")
    s.add_argument("decision")

    s = sub.add_parser("skipped", help="re-evaluations skipped and why")
    s.add_argument("db")
    s.add_argument("decision")

    args = p.parse_args(argv)
    log = AuditLog(args.db)
    a = Auditor(log)
    try:
        if args.cmd == "verify":
            c = a.verify()
            _print({"ok": c.ok, "checked": c.checked, "first_bad_seq": c.first_bad_seq, "reason": c.reason})
            return 0 if c.ok else 2
        if args.cmd == "history":
            _print([{"seq": r.seq, "event_type": r.event_type, "tx_time": r.tx_time, "payload": r.payload}
                    for r in a.history(args.decision)])
        elif args.cmd == "as-made":
            _print(a.reconstruct_as_made(args.decision, args.eval_id))
        elif args.cmd == "trail":
            _print(a.reasoning_trail(args.decision, args.eval_id))
        elif args.cmd == "impact":
            _print(a.retraction_impact(args.node))
        elif args.cmd == "clusters":
            _print(a.retraction_clusters(args.reason, args.key))
        elif args.cmd == "exposure":
            _print(a.reversal_exposure(args.decision, as_of_seq=args.as_of))
        elif args.cmd == "amortization":
            _print(a.amortization_schedule(args.decision))
        elif args.cmd == "skipped":
            _print(a.skipped_reevaluations(args.decision))
    except AuditError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
