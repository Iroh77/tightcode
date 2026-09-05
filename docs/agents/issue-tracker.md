# Issue tracker: Local Markdown (vault)

Issues, specs, and tickets for this project live as markdown files in the **vault**, under `Second Cerveau/1 PROJETS/tightcode/TICKETS/` (reachable from the repo via the `Second Cerveau` symlink).

## Conventions

- One file per ticket: `TICKETS/issues/<NN>-<slug>.md`, numbered from `01` — never a single combined file, no per-feature subdirectories (flat scheme per the new process; Slopotia's per-feature dirs stay as-is until Antoine decides to migrate)
- Implementation tickets: `Type:` (`task`/`feature`/`refactor`/`spike`) + `Requirement(s):` (`R<NN>-<seq>`, comma-separated — the requirement IDs this ticket serves) + `Status:` (triage label, e.g. `ready-for-agent`/`claimed`/`resolved`) lines near the top
- Triage state: a `Status:` line near the top (see `docs/agents/triage-labels.md` for role strings)
- Blocking edges: a `Blocked by:` line near the top
- Comments/conversation history: appended under `## Comments`
- `TICKETS/map.md` = the one-way coverage table (requirement ID → tickets, full/partial; `superseded` state)

## When a skill says "publish to the issue tracker"

Create a new file under `TICKETS/issues/` (creating the directory if needed).

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path (normally passed by the user as the path or number).

## Claiming a ticket

- **Claim**: set `Status: claimed` and save before any work (used by `dev-implement`).