#!/usr/bin/env node
/**
 * Launches the optimization swarm.
 *
 *   node swarm/launch.mjs                 # start all agents
 *   node swarm/launch.mjs profiler        # start just one
 *   node swarm/launch.mjs --dry-run       # print the prompts, start nothing
 *
 * Each agent gets its own SwarmChat token, its own worktree on the perf branch,
 * and a role-specific prompt. Worktrees keep agents from clobbering each other's
 * working tree while they iterate on measurements.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const ENV_FILE = path.join(HERE, ".env.local");

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const PROVIDER = process.env.SWARM_PROVIDER ?? "claude";
const MODEL = process.env.SWARM_MODEL ?? "claude-opus-5-5";
const MODE = process.env.SWARM_MODE ?? "auto";

/** Shared context every agent needs. Kept here so roles stay short. */
function preamble(role, token) {
	return `You are the ${role} on a 4-agent team optimizing Gemma 3 WebGPU inference.

Repository: ${ROOT}
Branch: perf (already checked out; work in the worktree you were given)

FIRST, before anything else:
1. Read bench/SWARM.md in full. It is the team contract: targets, measurement
   protocol, what is forbidden, and the definition of done.
2. Read bench/README.md for how the harness works.
3. Run \`node bench/preflight.mjs\` and note the renderer and precision.

SWARM CHAT (this is how the team communicates; use it, do not work silently):
  export SWARMCHAT_SERVER=${process.env.SWARMCHAT_SERVER ?? "http://127.0.0.1:8484"}
  export SWARMCHAT_TOKEN=${token}
  swarmchat chats                      # what is unread
  swarmchat read                       # read new messages
  swarmchat send "#perf" "..."         # design discussion / claim work
  swarmchat send "#results" "..."      # accepted numbers only
  swarmchat wait --from orchestrator   # block until the orchestrator replies
  swarmchat send orchestrator "..."    # direct message
Every command accepts --json. Claim a task in #perf before starting it, and
check that nobody else has claimed the same file or function.

TARGET (enforced by bench/compare.mjs):
  decode-ctx1024: 2.0x baseline tok/s
  prefill-1024:   1.5x baseline tok/s
  no workload below 0.9x baseline

HARD RULES:
- WebGPU / web APIs only. No native code, no server-side inference.
- Never weaken the correctness gate, edit a golden file, or change benchmark
  measurement logic or thresholds to make a number look better.
- Report numbers only from a real run, with the result JSON attached. Say
  "CPU fallback, relative only" if the renderer is software.
- Small commits. Say the before/after number in the commit message.

`;
}

const ROLES = {
	orchestrator: {
		title: "Orchestrator",
		prompt: (token) =>
			preamble("orchestrator", token) +
			`YOUR JOB
You own the target, the baseline, and #results. You do not write kernels.

Do this now:
1. Read bench/SWARM.md and bench/README.md.
2. Confirm the current state of bench/baseline.json (it may not exist yet). If
   it does not exist and this machine has a real GPU, run the full suite and
   record it:
     npm run bench -- --label baseline --repeats 5
     cp "$(ls -t bench/results/baseline-*.json | head -1)" bench/baseline.json
   If the machine is on the CPU fallback, do NOT record a baseline: post in
   #perf that the baseline is blocked on GPU access and keep the team on
   analysis-only work until it is resolved.
3. Post a short plan in #perf: who is doing what, and in what order.
4. Introduce yourself to the other agents and tell the profiler to begin.

Then stay in the loop: read #perf regularly, resolve conflicts over files,
and only accept a result into #results when it satisfies the definition of done
in bench/SWARM.md. Keep run-to-run noise in mind: anything under ~5% is noise
unless min/p90 disagree.

Finally, report back to me (the human) with a single concise message: what the
baseline is, what the plan is, and any blockers.`,
	},
	profiler: {
		title: "Profiler",
		prompt: (token) =>
			preamble("profiler", token) +
			`YOUR JOB
Find where the time actually goes. Do not optimize yet.

1. Build a reproducible profile of decode at ctx-1024 and prefill at 1024.
   jax-js has \`profiler.startTrace()\` / \`profiler.stopTrace()\` and the WebGPU
   backend supports timestamp queries; use whatever gives per-kernel numbers.
   Add helpers under bench/scripts/ and document how to reproduce them.
2. Identify the top cost centres with numbers: which kernels, how much of the
   step each accounts for, how many dispatches per token, how much time is
   host<->device sync, how much is kernel launch overhead.
3. Check for obviously wasted work: recomputed masks/RoPE/slot tables per step,
   KV-cache repadding, redundant casts, the per-token logits readback.
4. Post a ranked bottleneck list with numbers in #perf, and hand the top two
   items to the kernel and integrator agents with enough detail that they can
   start immediately.
5. Attach any trace or result files to your message.

Report back to me (the human) with the ranked list and your recommendation for
what to attack first.`,
	},
	kernel: {
		title: "Kernel Dev",
		prompt: (token) =>
			preamble("kernel", token) +
			`YOUR JOB
Write fast GPU kernels for the hot paths the profiler identifies. Likely
candidates: fused attention (never materialise the [S,4,N] score tensor), a
tiled matmul for the projections, and fusing norm+projection.

1. Start by reading src/lib/gemma3/model.ts carefully, especially runAttention.
   Also look at the existing experiments on origin/webgpu-kernels
   (git show origin/webgpu-kernels:src/lib/webgpu/flashattention.ts etc.) : that
   is prior art for a flash-attention kernel here. Treat it as a starting
   point, not as correct or fast.
2. Prefer expressing fusion in jax-js primitives. Only drop to a custom WGSL
   kernel when that genuinely cannot work.
3. If you patch @jax-js/jax, keep the patch in patches/ and apply it from
   postinstall so a clean \`npm install\` reproduces the tree. Document what it
   changes and why at the top of the patch. Announce that you are doing this in
   #perf first, because it affects everyone.
4. After every change: npm run bench:check (must pass) then a benchmark run.
   Post accepted results in #results with the JSON attached.

Coordinate with the integrator before changing src/lib/gemma3/* so you are not
both editing the same functions.

Report back to me (the human) with what you built, the measured effect, and
whether it is ready to ship.`,
	},
	integrator: {
		title: "Integrator",
		prompt: (token) =>
			preamble("integrator", token) +
			`YOUR JOB
Make the inference loop itself do less work per token, and wire in the kernel
agent's fast paths as they land.

Start with the cheap, high-certainty wins in src/lib/gemma3/model.ts:
1. Per-step recomputation. precomputeSlotsAndMasks and precomputeRoPECache run
   every step. For decode (S=1) these are tiny inputs producing small outputs;
   check whether they are actually costing anything, and whether they can be
   computed once per position or folded into the layer.
2. The KV-cache update uses np.where over the whole cache buffer each layer.
   That is O(capacity) work per token. Consider a direct write/scatter path.
3. KV-cache growth repads the entire cache. Check the cost of the
   CACHE_EXPANSION_SIZE policy.
4. The sampling path forces a host readback every token (sampleLogits awaits
   .data()). That serialises the GPU pipeline: consider keeping more on device,
   or overlapping the readback with the next step.
5. Redundant .astype() casts and any [S,4,N] materialisation that can be
   avoided.

Work incrementally: measure, change one thing, run npm run bench:check, then
benchmark. Do not batch many changes into one measurement.

Coordinate with the kernel agent before editing the same functions.

Report back to me (the human) with what you changed, the measured effect of
each change, and what is still on the table.`,
	},
};

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

if (!fs.existsSync(ENV_FILE)) {
	console.error(`Missing ${ENV_FILE}. Run: ./swarm/setup-swarm.sh`);
	process.exit(1);
}

const env = Object.fromEntries(
	fs
		.readFileSync(ENV_FILE, "utf8")
		.split("\n")
		.filter((l) => l && !l.startsWith("#") && l.includes("="))
		.map((l) => {
			const i = l.indexOf("=");
			return [l.slice(0, i), l.slice(i + 1)];
		}),
);

const argv = process.argv.slice(2);
const dryRun = argv.includes("--dry-run");
const wanted = argv.filter((a) => !a.startsWith("--"));
const roles = wanted.length ? wanted : Object.keys(ROLES);

for (const r of roles) {
	if (!ROLES[r]) {
		console.error(`Unknown role "${r}". Known: ${Object.keys(ROLES).join(", ")}`);
		process.exit(1);
	}
}

if (dryRun) {
	for (const r of roles) {
		const token = env[`SWARMCHAT_TOKEN_${r.toUpperCase()}`] ?? "<no token>";
		console.log(`\n${"=".repeat(70)}\n${r.toUpperCase()} (${ROLES[r].title})\n${"=".repeat(70)}\n`);
		console.log(ROLES[r].prompt(token));
	}
	process.exit(0);
}

// ---------------------------------------------------------------------------
// Git worktrees: one per agent so their working trees do not collide
// ---------------------------------------------------------------------------

const baseBranch = process.env.SWARM_BRANCH ?? "perf";

function ensureWorktree(role) {
	const dir = path.join("/tmp", "gemma-swarm", role);
	if (fs.existsSync(dir)) return dir;
	fs.mkdirSync(path.dirname(dir), { recursive: true });
	const branch = `swarm/${role}`;
	try {
		execFileSync("git", ["worktree", "add", "-B", branch, dir, baseBranch], {
			cwd: ROOT,
			stdio: "pipe",
		});
	} catch (e) {
		console.error(`failed to create worktree for ${role}:`, String(e.stderr ?? e).slice(0, 500));
		process.exit(1);
	}
	// Share the downloaded weights and unpacked Chrome libs: they are large and
	// identical across agents.
	for (const sub of [".cache"]) {
		const from = path.join(ROOT, "bench", sub);
		const to = path.join(dir, "bench", sub);
		if (fs.existsSync(from) && !fs.existsSync(to)) {
			fs.mkdirSync(path.dirname(to), { recursive: true });
			fs.symlinkSync(from, to, "dir");
		}
	}
	// Give each agent its own identity without committing tokens.
	fs.writeFileSync(
		path.join(dir, "swarm", ".env.local"),
		fs.readFileSync(ENV_FILE, "utf8"),
	);
	return dir;
}

console.log(`provider=${PROVIDER} model=${MODEL} mode=${MODE} branch=${baseBranch}`);
console.log(`agents: ${roles.join(", ")}\n`);

const launched = [];
for (const role of roles) {
	const token = env[`SWARMCHAT_TOKEN_${role.toUpperCase()}`];
	if (!token) {
		console.error(`No SwarmChat token for ${role}; re-run ./swarm/setup-swarm.sh`);
		process.exit(1);
	}
	const dir = ensureWorktree(role);
	console.log(`${ROLES[role].title.padEnd(14)} worktree ${dir}`);
	launched.push({ role, dir, prompt: ROLES[role].prompt(token) });
}

// Write the prompts to disk so a launcher (or a human) can start the agents
// without re-deriving them. The Paseo session itself is created by the caller,
// which knows which provider/host to use.
fs.mkdirSync(path.join(HERE, "prompts"), { recursive: true });
for (const { role, prompt } of launched) {
	fs.writeFileSync(path.join(HERE, "prompts", `${role}.md`), prompt);
}

console.log(`\nwrote prompts to swarm/prompts/`);
console.log(`\nStart each agent with provider "${PROVIDER}", model "${MODEL}",`);
console.log(`working directory set to its worktree, and the prompt from`);
console.log(`swarm/prompts/<role>.md as the initial message.`);
console.log(`\nSwarmChat tokens are in swarm/.env.local (per-agent, gitignored).`);
