"""Score each commit in commits.jsonl with an LLM judge.

For every commit we build a self-contained prompt from the commit-level and
per-file features produced by `get_contributor_stats.py`, hand it to
`llm_judge` (Gemini, see llm_judge.py), and collect a 1-10 quality/impact
score.

Output: a CSV file with columns committer_datetime, author_name, score,
reason (default filename: initial-gemini-flash3.1.csv).

Usage:
    python evaluate.py
    python evaluate.py --path commits.jsonl --max-patch-chars 2000
    python evaluate.py --output-csv my-results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from llm_judge import llm_judge


DEFAULT_MAX_PATCH_CHARS = 2000
DEFAULT_MAX_FILES_IN_PROMPT = 50


def _truncate(text: str, limit: int) -> str:
    if text is None:
        return ""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more chars]"


def build_prompt(
    commit: dict[str, Any],
    *,
    max_patch_chars: int = DEFAULT_MAX_PATCH_CHARS,
    max_files: int = DEFAULT_MAX_FILES_IN_PROMPT,
) -> str:
    """Render a single commit into an evaluator prompt.

    Only features relevant to judging code quality and impact are included
    (we deliberately omit author identity so the LLM judges the change, not
    the person).
    """
    message = (commit.get("message") or "").strip()
    is_merge = bool(commit.get("is_merge"))
    files_changed = commit.get("files_changed")
    additions = commit.get("additions")
    deletions = commit.get("deletions")
    total_changes = commit.get("total_changes")

    files = [f for f in (commit.get("files") or []) if isinstance(f, dict)]
    total_files = len(files)

    file_blocks: list[str] = []
    for idx, f in enumerate(files[:max_files], start=1):
        block_lines = [
            f"[file {idx}/{total_files}] {f.get('filename')}",
            f"  status: {f.get('status')}",
            f"  additions: {f.get('additions')}  deletions: {f.get('deletions')}  changes: {f.get('changes')}",
        ]
        prev = f.get("previous_filename")
        if prev:
            block_lines.append(f"  previous_filename: {prev}")
        patch = f.get("patch")
        if patch:
            block_lines.append("  patch:")
            block_lines.append(_truncate(patch, max_patch_chars))
        file_blocks.append("\n".join(block_lines))

    if total_files > max_files:
        file_blocks.append(f"... [{total_files - max_files} more files omitted]")

    files_section = "\n\n".join(file_blocks) if file_blocks else "(no per-file diff data)"

    return (
        "You are a senior code reviewer. Evaluate the QUALITY and IMPACT of a single git commit\n"
        "based ONLY on the change itself. Do not consider author identity, timing, or repo politics.\n\n"
        "=== EVALUATION DIMENSIONS ===\n"
        "Weigh these roughly equally:\n"
        "  1. Impact & criticality - does it touch core logic, public APIs, security, data\n"
        "     correctness, or performance hot paths? Or is it peripheral (docs, configs, tests-only)?\n"
        "  2. Engineering quality - clarity, structure, naming, error handling, edge-case\n"
        "     coverage, and whether the diff looks correct and minimal for its goal.\n"
        "  3. Test coverage - are tests added/updated alongside non-trivial logic changes?\n"
        "  4. Risk & scope discipline - focused, well-scoped change vs. sprawling/unrelated edits.\n"
        "  5. Commit message - does it clearly explain WHAT changed and WHY?\n\n"
        "=== SCORING RUBRIC (1-10) ===\n"
        "  1  Noise. Whitespace-only, reverted same-session, accidental commit, or empty change.\n"
        "  2  Trivial. Typo fix, comment tweak, version bump, formatting, generated-file churn.\n"
        "  3  Very minor. Tiny config/doc edit, single-line cosmetic change, no logic impact.\n"
        "  4  Minor. Small localized refactor or cleanup; correct but low-stakes; weak message.\n"
        "  5  Routine. Standard small feature, bug fix, or refactor in non-critical code;\n"
        "     adequate quality; tests optional or partial.\n"
        "  6  Solid. Well-scoped feature/fix with reasonable quality and a clear message;\n"
        "     touches meaningful code; some testing or obviously safe.\n"
        "  7  Strong. Non-trivial feature, fix, or refactor in important code; good structure,\n"
        "     tests present where they matter, message explains intent.\n"
        "  8  High impact. Substantial change to core functionality OR important bug fix with\n"
        "     thoughtful design, solid tests, and clear rationale.\n"
        "  9  Critical & excellent. Architectural improvement, major feature, security/correctness\n"
        "     fix in core paths; clean diff, thorough tests, excellent commit message.\n"
        "  10 Exemplary. Rare. Major architectural or correctness work executed at very high\n"
        "     quality, broad positive impact, with tests and rationale that set a standard.\n\n"
        "=== SPECIAL CASES ===\n"
        "- Pure merge commits with no conflict resolution: score 1-2.\n"
        "- Auto-generated files, lockfiles, or vendored dependencies dominating the diff: cap at 3\n"
        "  unless the change clearly required substantive engineering judgment.\n"
        "- Large LOC counts alone do NOT imply high score; mass renames or formatting stay low.\n"
        "- Small LOC counts can still score high if the change is critical (e.g., a one-line fix\n"
        "  to a serious bug in core logic).\n\n"
        "=== OUTPUT FORMAT ===\n"
        "Respond with a single line of valid JSON and nothing else - no prose, no\n"
        "markdown, no code fences. Schema:\n"
        '  {"score": <int 1-10>, "reason": "<one short sentence, <=160 chars>"}\n'
        "The reason must briefly justify the score (e.g. what carried it up or held it down).\n\n"
        "=== COMMIT SUMMARY ===\n"
        f"is_merge: {is_merge}\n"
        f"files_changed: {files_changed}\n"
        f"additions: {additions}  deletions: {deletions}  total_changes: {total_changes}\n\n"
        "=== COMMIT MESSAGE ===\n"
        f"{message}\n\n"
        f"=== PER-FILE CHANGES ({total_files} file(s)) ===\n"
        f"{files_section}\n"
    )


def _coerce_score(raw: Any) -> int:
    try:
        s = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(1, min(10, s))


_REASON_MAX_CHARS = 160


def _coerce_reason(raw: Any) -> str:
    if raw is None:
        return ""
    text = str(raw).strip().replace("\n", " ").replace("\r", " ")
    if len(text) > _REASON_MAX_CHARS:
        text = text[: _REASON_MAX_CHARS - 1].rstrip() + "..."
    return text


def parse_judge_output(raw: Any) -> tuple[int, str]:
    """Normalize whatever the judge returned into (score, reason).

    Accepts either a dict with "score"/"reason" keys, a JSON string with the
    same shape, or a bare integer/string score for backward compatibility.
    """
    if isinstance(raw, dict):
        return _coerce_score(raw.get("score")), _coerce_reason(raw.get("reason"))

    if isinstance(raw, str):
        text = raw.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return _coerce_score(text), ""
        if isinstance(parsed, dict):
            return _coerce_score(parsed.get("score")), _coerce_reason(parsed.get("reason"))
        return _coerce_score(parsed), ""

    return _coerce_score(raw), ""


def iter_commits(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_results_csv(rows: list[tuple[str, str, int, str]], path: Path) -> None:
    headers = ("committer_datetime", "author_name", "score", "reason")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--path",
        type=Path,
        default=Path("commits.jsonl"),
        help="JSONL from get_contributor_stats.py (default: commits.jsonl)",
    )
    p.add_argument("--max-patch-chars", type=int, default=DEFAULT_MAX_PATCH_CHARS)
    p.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES_IN_PROMPT)
    p.add_argument("--limit", type=int, default=0, help="Only score the first N commits (0 = all)")
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("initial-gemini-flash3.1.csv"),
        help="where to write the results table (default: initial-gemini-flash3.1.csv)",
    )
    args = p.parse_args()

    if not args.path.is_file():
        print(f"error: not a file: {args.path}", file=sys.stderr)
        sys.exit(1)

    rows: list[tuple[str, str, int, str]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    for i, commit in enumerate(iter_commits(args.path)):
        if args.limit and i >= args.limit:
            break
        print("got commit : ", commit.get("committer_datetime"))
        prompt = build_prompt(
            commit,
            max_patch_chars=args.max_patch_chars,
            max_files=args.max_files,
        )
        result, usage = llm_judge(prompt)
        score, reason = parse_judge_output(result)
        total_input_tokens += int(usage.get("input_tokens", 0))
        total_output_tokens += int(usage.get("output_tokens", 0))
        rows.append(
            (
                commit.get("committer_datetime") or "",
                commit.get("author_name") or "",
                score,
                reason,
            )
        )
        print("got score : ", score, "-", reason)
        print(
            "tokens: input=", usage.get("input_tokens", 0),
            " output=", usage.get("output_tokens", 0),
        )
        print("--------------------------------")

    write_results_csv(rows, args.output_csv)
    print(f"Wrote {len(rows)} row(s) to {args.output_csv.resolve()}", file=sys.stderr)
    print(
        f"Total tokens - input: {total_input_tokens}, output: {total_output_tokens}, "
        f"combined: {total_input_tokens + total_output_tokens}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
