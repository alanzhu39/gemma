#!/usr/bin/env node
/**
 * Compares a benchmark result against the recorded baseline and enforces the
 * project's performance target.
 *
 *   node bench/compare.mjs                       # latest vs bench/baseline.json
 *   node bench/compare.mjs --against bench/results/foo.json
 *   node bench/compare.mjs --gate                 # exit 1 unless target met
 *
 * Exit codes: 0 = ok, 1 = gate failed or regression, 2 = unusable comparison.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));

function parse(argv) {
	const o = { against: path.join(HERE, "baseline.json"), latest: path.join(HERE, "results", "latest.json"), gate: false, json: false };
	for (let i = 0; i < argv.length; i++) {
		if (argv[i] === "--against") o.against = argv[++i];
		else if (argv[i] === "--latest") o.latest = argv[++i];
		else if (argv[i] === "--gate") o.gate = true;
		else if (argv[i] === "--json") o.json = true;
	}
	return o;
}

/**
 * The target. Expressed as required multipliers on the baseline, per workload.
 * `floor` is the regression allowance: an optimization may not lose more than
 * this fraction of throughput on a workload it is not targeting.
 */
const TARGET = {
	decode: 2.0, // decode-ctx1024
	prefill: 1.5, // prefill-1024
	floor: 0.9, // no workload may drop below 90% of baseline
	workloads: {
		decode: "decode-ctx1024",
		prefill: "prefill-1024",
	},
};

const args = parse(process.argv.slice(2));

function load(p) {
	if (!fs.existsSync(p)) return null;
	return JSON.parse(fs.readFileSync(p, "utf8"));
}

const baseDoc = load(args.against);
const latestDoc = load(args.latest);

if (!baseDoc || !latestDoc) {
	console.error(`Missing ${!baseDoc ? args.against : args.latest}`);
	console.error("Record a baseline with: npm run bench -- --label baseline  (then copy to bench/baseline.json)");
	process.exit(2);
}

// A comparison across different renderers/precisions is meaningless.
const baseSys = baseDoc.sysInfo ?? {};
const newSys = latestDoc.sysInfo ?? {};
if (baseSys.precision !== newSys.precision) {
	console.error(
		`Cannot compare: baseline precision ${baseSys.precision} vs ${newSys.precision}. ` +
			`Re-record the baseline on the same hardware.`,
	);
	process.exit(2);
}

const byName = (doc) => new Map((doc.samples ?? []).map((s) => [s.workload, s]));
const base = byName(baseDoc);
const latest = byName(latestDoc);

const rows = [];
for (const [name, now] of latest) {
	const before = base.get(name);
	if (!before) continue;
	rows.push({
		name,
		kind: now.kind,
		before: before.tps,
		after: now.tps,
		ratio: now.tps / before.tps,
	});
}

const get = (n) => rows.find((r) => r.name === n);
const decodeRow = get(TARGET.workloads.decode);
const prefillRow = get(TARGET.workloads.prefill);

if (!args.json) {
	console.log(`baseline: ${baseDoc.sysInfo?.precision ?? "?"} on ${describe(baseSys)}`);
	console.log(`current:  ${newSys.precision ?? "?"} on ${describe(newSys)}`);
	if (newSys.isSoftwareRenderer) {
		console.log("!! software renderer: relative comparison only\n");
	} else {
		console.log("");
	}

	console.log("workload             baseline      current     change");
	console.log("--------------------------------------------------------");
	for (const r of rows) {
		const pct = (r.ratio - 1) * 100;
		const sign = pct >= 0 ? "+" : "";
		console.log(
			`${r.name.padEnd(20)} ${r.before.toFixed(2).padStart(8)} ${r.after.toFixed(2).padStart(12)} ${(sign + pct.toFixed(1) + "%").padStart(11)}`,
		);
	}
	console.log("");
}

const problems = [];

// Regressions on anything we are not explicitly trading away.
for (const r of rows) {
	const isTarget = r.name === TARGET.workloads.decode || r.name === TARGET.workloads.prefill;
	if (!isTarget && r.ratio < TARGET.floor) {
		problems.push(`${r.name} regressed to ${(r.ratio * 100).toFixed(1)}% of baseline`);
	}
}

if (decodeRow) {
	const ok = decodeRow.ratio >= TARGET.decode;
	if (!args.json) {
		console.log(
			`${ok ? "PASS" : "FAIL"}  decode target ${TARGET.decode}x  ` +
				`(${decodeRow.ratio.toFixed(2)}x vs baseline)`,
		);
	}
	if (!ok) problems.push(`decode ${decodeRow.ratio.toFixed(2)}x < ${TARGET.decode}x`);
}

if (prefillRow) {
	const ok = prefillRow.ratio >= TARGET.prefill;
	if (!args.json) {
		console.log(
			`${ok ? "PASS" : "FAIL"}  prefill target ${TARGET.prefill}x  ` +
				`(${prefillRow.ratio.toFixed(2)}x vs baseline)`,
		);
	}
	if (!ok) problems.push(`prefill ${prefillRow.ratio.toFixed(2)}x < ${TARGET.prefill}x`);
}

if (args.json) {
	console.log(JSON.stringify({ rows, problems, target: TARGET }, null, 2));
}

function describe(sys) {
	const info = sys.adapterInfo ?? {};
	return info.description || info.device || (sys.isSoftwareRenderer ? "software renderer" : "unknown adapter");
}

if (problems.length) {
	if (!args.json) {
		console.log("\nBlocking issues:");
		for (const p of problems) console.log(`  - ${p}`);
	}
	if (args.gate) process.exit(1);
}
