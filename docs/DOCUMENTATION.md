# Documentation system

What gets written down, where, and by what rules — so that work survives
context loss (human forgetting, agent compaction, machine changes) without
re-derivation.

## Where things live

| artifact | location | audience | in git? |
|---|---|---|---|
| Design narratives: why, evidence, decisions, dead ends | `docs/specs/<topic>.md` | humans + agents | yes |
| Raw measurements + reproduce commands | `docs/benchmarks/<topic>.md` | humans + agents | yes |
| User-facing quickstart (build/run/serve) | `README.md` | users | yes |
| Agent session entry point (conventions, commands, pointers) | `CLAUDE.md` | agents | **yes** — it is the project's CONTRIBUTING.md for agents; any clone, any agent, benefits |
| Personal/behavioral preferences (language, push policy, scope) | the agent's local memory (`~/.claude/.../memory/`) | one user's agent | **no** — user-personal, not project truth |
| Throwaway probes, bench harnesses mid-investigation | session scratchpad | nobody later | no — PROMOTE the results (not the script) into `docs/benchmarks/` before the session ends |

The git test: *would a stranger cloning this repo (or an agent with zero
session memory) need it to continue the work?* Yes → git. Is it about how
one particular user likes to be worked with? → memory, not git.

## Rules

1. **Numbers or it didn't happen.** Every performance or quality claim
   carries: the number, the command/method, and the machine state that
   makes it valid (for this repo: wired vs unwired, what else was resident,
   warm vs cold). An unqualified number is a future trap — see the
   0.53 tok/s incident in `docs/benchmarks/deepseek-v4.md`.
2. **Negative results are first-class.** A refuted hypothesis or a reverted
   optimization gets recorded WITH the failed reasoning (what was predicted,
   what was measured). Each one is a day someone else doesn't lose.
   Examples that already paid for this rule: the launch-tax refutations,
   AWQ folding, the HC hand kernel.
3. **Write at the moment of measurement**, in the same session — not
   "later". Compaction and forgetting are the default, not the exception.
4. **Spec vs benchmark split**: the spec tells the story (why, decisions,
   current status at the top); the benchmark ledger holds the tables and
   reproduce steps. They link to each other. Update the spec's Status
   block whenever the state of the world changes.
5. **Commit messages are documentation.** The why and the key numbers go in
   the message body; the docs hold the full tables. A reader of
   `git log` alone should be able to follow the arc.
6. Dated entries for time-sensitive facts (upstream versions, issue
   states): write "as of YYYY-MM-DD", because upstream moves.
7. Repo documents are English (matching code/commits); user-facing
   conversation stays in the user's language.

## Continuation protocol (agent, after compaction or a fresh session)

1. `CLAUDE.md` loads automatically — it points here and to the live specs.
2. Read the relevant spec's **Status** block first; it must always say
   what is done, what is broken, and what is next.
3. Trust recorded measurements over re-derivation; if a fact is dated and
   old, re-verify before building on it.
