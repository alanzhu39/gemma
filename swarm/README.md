# Optimization swarm

A small team of coding agents improving Gemma 3 inference performance, coordinating
over [SwarmChat](https://github.com/Pranjal2041/swarmchat).

The team contract (targets, measurement rules, definition of done) is
[`bench/SWARM.md`](../bench/SWARM.md). Read that first; this file only covers
how to run the swarm.

## Layout

| Path | Purpose |
| --- | --- |
| `bench/` | The harness, correctness gate, and regression gate agents are judged by |
| `swarm/setup-swarm.sh` | Creates SwarmChat identities and the `#perf` / `#results` channels |
| `swarm/launch.mjs` | Creates a git worktree per agent and writes their prompts |
| `swarm/.env.local` | Per-agent SwarmChat tokens — **gitignored, do not commit** |
| `swarm/prompts/` | Generated per-agent prompts (gitignored) |

## Starting the swarm

```sh
# 1. Start the chat server (once)
swarmchat serve --data /tmp/swarm-data        # http://127.0.0.1:8484

# 2. Create identities + channels
./swarm/setup-swarm.sh

# 3. Create a worktree and prompt for each agent
node swarm/launch.mjs

# 4. Watch the team work
swarmchat view --data /tmp/swarm-data
```

`node swarm/launch.mjs --dry-run profiler` prints a single prompt without
creating anything, which is handy when editing roles.

## Roles

| Agent | Owns |
| --- | --- |
| **orchestrator** | The target, the baseline, `#results`, conflict resolution. Does not write kernels. |
| **profiler** | Finding where time actually goes; produces a ranked bottleneck list with numbers. |
| **kernel** | Custom WebGPU kernels and jax-js patches. |
| **integrator** | Rewiring `src/lib/gemma3/*` to do less work per token and use the fast paths. |

## Coordination

- `#perf` — hypotheses, claims on files/functions, questions.
- `#results` — accepted numbers and verdicts only.

Agents claim work before starting it and check that nobody else owns the same
file. `swarmchat wait --from <agent>` blocks without spending tokens, so agents
can wait on each other instead of polling.

## Prerequisites

- `swarmchat` on `PATH` (`uv tool install swarmchat` or `pip install swarmchat`).
- A WebGPU-capable machine. Run `node bench/preflight.mjs` first: if it reports
  a software renderer, no baseline can be recorded and the swarm should stay on
  analysis-only work until GPU access is fixed.
