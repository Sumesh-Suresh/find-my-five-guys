"""Download per-commit work records from a GitHub repo for the last N days.

Each output record is one commit, indexed by commit timestamp, and includes the
actual patch content of every file the contributor touched. The patch text is
intended to be used as a feature for downstream quality evaluation
(e.g. message-vs-diff coherence, churn/refactor ratios, test coverage in diff,
LLM-as-a-judge scoring, etc.).

Defaults to PostHog/posthog and the last 90 days.

Output format: JSON Lines (one commit per line). Load with:
    import pandas as pd
    df = pd.read_json("commits.jsonl", lines=True, convert_dates=["datetime"])
    df = df.set_index("datetime").sort_index()

By default only commits with a green CI signal (combined commit-status and all
check-runs successful) are kept. Pass --include-unsuccessful to disable that.

Usage:
    export GITHUB_TOKEN=ghp_xxx        # strongly recommended (5000/hour vs 60/hour)
    python get_contributor_stats.py
    python get_contributor_stats.py --owner PostHog --repo posthog --days 90
    python get_contributor_stats.py --branch-name release-1.0  # walk a specific branch
                                                                # (default: repo's default branch, auto-detected)
    python get_contributor_stats.py --include-unsuccessful   # don't filter by CI status
    python get_contributor_stats.py --no-patch               # skip per-file patch text (smaller output)
    python get_contributor_stats.py --skip-merges            # exclude merge commits
    python get_contributor_stats.py --max-patch-bytes 20000  # truncate huge patches
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import requests

GITHUB_API = "https://api.github.com"


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "contributor-work-extractor",
        }
    )
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    else:
        print(
            "WARNING: GITHUB_TOKEN not set. Unauthenticated requests are limited "
            "to 60/hour and will fail quickly on a busy repo.",
            file=sys.stderr,
        )
    return s


def respect_rate_limit(resp: requests.Response) -> None:
    """If we're close to the rate limit, sleep until reset."""
    remaining = resp.headers.get("X-RateLimit-Remaining")
    reset = resp.headers.get("X-RateLimit-Reset")
    if remaining is None or reset is None:
        return
    try:
        remaining_i = int(remaining)
        reset_i = int(reset)
    except ValueError:
        return
    if remaining_i <= 1:
        sleep_for = max(0, reset_i - int(time.time())) + 2
        print(
            f"Rate limit nearly exhausted. Sleeping {sleep_for}s until reset...",
            file=sys.stderr,
        )
        time.sleep(sleep_for)


def get(session: requests.Session, url: str, params: dict | None = None) -> requests.Response:
    """GET with simple retry on 403/429 (secondary rate limit) and rate-limit awareness."""
    # time.sleep(0.5)
    for attempt in range(5):
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code in (403, 429):
            retry_after = resp.headers.get("Retry-After")
            wait = int(retry_after) if retry_after else 2 ** attempt
            print(f"Got {resp.status_code}. Sleeping {wait}s and retrying...", file=sys.stderr)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        respect_rate_limit(resp)
        return resp
    resp.raise_for_status()
    return resp


def iter_commit_summaries(
    session: requests.Session,
    owner: str,
    repo: str,
    since_iso: str,
    branch: str,
) -> Iterator[dict[str, Any]]:
    """Yield commit summaries from the list-commits endpoint, following pagination.

    The list endpoint does NOT include stats or file patches; for that we need
    a separate call per commit (see fetch_commit_detail).

    The `sha` query param accepts a branch name (or a commit SHA) and restricts
    the walk to that branch's history.
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/commits"
    params: dict[str, Any] | None = {
        "since": since_iso,
        "per_page": 100,
        "sha": branch,
    }
    while url:
        resp = get(session, url, params=params)
        for commit in resp.json():
            yield commit
        params = None  # the Link header's `next` URL already has params baked in
        url = resp.links.get("next", {}).get("url")  # type: ignore[assignment]


def fetch_commit_detail(
    session: requests.Session, owner: str, repo: str, sha: str
) -> dict[str, Any]:
    """Fetch one commit with its `stats` and `files` (which contain the patch text)."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}"
    return get(session, url).json()


def get_default_branch(session: requests.Session, owner: str, repo: str) -> str:
    """Return the repo's default branch (e.g. 'master' for PostHog/posthog)."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}"
    data = get(session, url).json()
    branch = data.get("default_branch")
    if not branch:
        raise RuntimeError(f"Could not determine default branch for {owner}/{repo}")
    return branch


# Check-run conclusions that mean the run did NOT succeed. "neutral" and "skipped"
# are treated as non-failing because they're commonly used for opt-out/no-op checks.
_FAILING_CONCLUSIONS = {"failure", "timed_out", "cancelled", "action_required", "stale"}


def commit_is_successful(
    session: requests.Session, owner: str, repo: str, sha: str
) -> bool:
    """Return True iff `sha` has a green CI signal on GitHub.

    "Successful" here means:
      * the legacy combined commit-status rollup is `success` (or there are no
        legacy statuses on this commit), AND
      * every GitHub Actions check-run is `completed` with a non-failing
        conclusion, AND
      * at least one CI signal (legacy status OR check-run) exists.

    Commits with no CI signal at all are treated as not-successful, since we
    can't verify them.

    Costs up to 2 extra API calls per commit (one for `/status`, one for
    `/check-runs`).
    """
    status_url = f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}/status"
    status = get(session, status_url).json()
    legacy_state = status.get("state")
    legacy_count = status.get("total_count") or len(status.get("statuses") or [])

    runs_url = f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}/check-runs"
    runs_resp = get(session, runs_url, params={"per_page": 100}).json()
    runs = runs_resp.get("check_runs") or []

    if legacy_count == 0 and not runs:
        return False
    if legacy_count > 0 and legacy_state != "success":
        return False
    for r in runs:
        if r.get("status") != "completed":
            return False
        if r.get("conclusion") in _FAILING_CONCLUSIONS:
            return False
    return True


def truncate(text: str | None, limit: int | None) -> str | None:
    if text is None or limit is None or limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more chars]"


def build_record(
    summary: dict[str, Any],
    detail: dict[str, Any],
    include_patch: bool,
    max_patch_bytes: int | None,
) -> dict[str, Any]:
    commit_meta = detail.get("commit") or {}
    git_author = commit_meta.get("author") or {}
    git_committer = commit_meta.get("committer") or {}
    gh_author = detail.get("author") or {}
    gh_committer = detail.get("committer") or {}
    stats = detail.get("stats") or {}
    files = detail.get("files") or []
    parents = summary.get("parents") or detail.get("parents") or []

    file_records: list[dict[str, Any]] = []
    for f in files:
        rec = {
            "filename": f.get("filename"),
            "status": f.get("status"),
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "changes": f.get("changes", 0),
            "previous_filename": f.get("previous_filename"),
        }
        if include_patch:
            rec["patch"] = truncate(f.get("patch"), max_patch_bytes)
        file_records.append(rec)

    return {
        "datetime": git_author.get("date"),
        "committer_datetime": git_committer.get("date"),
        "sha": detail.get("sha") or summary.get("sha"),
        "author_login": gh_author.get("login"),
        "author_name": git_author.get("name"),
        "author_email": git_author.get("email"),
        "committer_login": gh_committer.get("login"),
        "committer_name": git_committer.get("name"),
        "message": commit_meta.get("message"),
        "is_merge": len(parents) > 1,
        "parent_shas": [p.get("sha") for p in parents],
        "additions": stats.get("additions", 0),
        "deletions": stats.get("deletions", 0),
        "total_changes": stats.get("total", 0),
        "files_changed": len(files),
        "filenames": [f.get("filename") for f in files],
        "url": detail.get("html_url") or summary.get("html_url"),
        "files": file_records,  # the "contents of the work": per-file patch text
    }


def fetch_records(
    session: requests.Session,
    owner: str,
    repo: str,
    days: int,
    branch: str,
    include_patch: bool,
    skip_merges: bool,
    only_successful: bool,
    max_patch_bytes: int | None,
    output_path: str,
) -> int:
    since_dt = datetime.now(tz=timezone.utc) - timedelta(days=days)
    since_iso = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(
        f"Fetching commits for {owner}/{repo} on branch '{branch}' since {since_iso} "
        f"(only_successful={only_successful})...",
        file=sys.stderr,
    )

    count = 0
    seen = 0
    skipped_merge = 0
    skipped_unsuccessful = 0
    # Stream straight to disk so memory stays flat even if patches are huge.
    # Records are written in the order the API returns them (newest-first); we
    # do a final sort pass below so the file ends up ascending by datetime.
    tmp_path = output_path + ".unsorted"
    with open(tmp_path, "w", encoding="utf-8") as out:
        for summary in iter_commit_summaries(session, owner, repo, since_iso, branch):
            seen += 1
            parents = summary.get("parents") or []
            if skip_merges and len(parents) > 1:
                skipped_merge += 1
                continue
            # Check CI success BEFORE the expensive detail fetch so filtered-out
            # commits cost 2 API calls instead of 3.
            if only_successful and not commit_is_successful(
                session, owner, repo, summary["sha"]
            ):
                skipped_unsuccessful += 1
                continue
            detail = fetch_commit_detail(session, owner, repo, summary["sha"])
            record = build_record(summary, detail, include_patch, max_patch_bytes)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if seen % 50 == 0:
                print(
                    f"  seen {seen}, kept {count}, "
                    f"skipped {skipped_merge} merges, {skipped_unsuccessful} unsuccessful...",
                    file=sys.stderr,
                )

    # Sort the file ascending by `datetime` so it's properly time-indexed on disk.
    print("Sorting records by commit datetime...", file=sys.stderr)
    with open(tmp_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    records.sort(key=lambda r: r.get("datetime") or "")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.remove(tmp_path)

    print(
        f"Done. Saw {seen} commits, kept {count}, "
        f"skipped {skipped_merge} merges and {skipped_unsuccessful} unsuccessful. "
        f"Wrote to {output_path}",
        file=sys.stderr,
    )
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", default="PostHog")
    parser.add_argument("--repo", default="posthog")
    parser.add_argument("--days", type=int, default=90, help="how many days back to include")
    parser.add_argument(
        "--branch-name",
        dest="branch_name",
        default=None,
        help="branch (or commit SHA) whose history to walk. Defaults to the "
        "repo's default branch (auto-detected via the GitHub API).",
    )
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="omit per-file patch text (much smaller output, but loses the 'contents of work' feature)",
    )
    parser.add_argument(
        "--skip-merges",
        action="store_true",
        help="exclude merge commits (commits with >1 parent)",
    )
    parser.add_argument(
        "--include-unsuccessful",
        dest="only_successful",
        action="store_false",
        help="include commits whose CI is failing/pending/missing. By default only "
        "commits with a green CI signal (combined status + check-runs) are kept.",
    )
    parser.set_defaults(only_successful=True)
    parser.add_argument(
        "--max-patch-bytes",
        type=int,
        default=20000,
        help="truncate each file's patch text to this many bytes (0 = no limit). Default 20000.",
    )
    parser.add_argument("--output", default="commits.jsonl")
    args = parser.parse_args()

    session = make_session()
    branch = args.branch_name or get_default_branch(session, args.owner, args.repo)
    if args.branch_name is None:
        print(f"Auto-detected default branch: '{branch}'", file=sys.stderr)
    fetch_records(
        session=session,
        owner=args.owner,
        repo=args.repo,
        days=args.days,
        branch=branch,
        include_patch=not args.no_patch,
        skip_merges=args.skip_merges,
        only_successful=args.only_successful,
        max_patch_bytes=args.max_patch_bytes if args.max_patch_bytes > 0 else None,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
