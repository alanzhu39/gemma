/**
 * Records whether the machine was quiet enough for timing to mean anything.
 *
 * A benchmark is only comparable to another benchmark if both ran without
 * competing for CPU/GPU. This box has other workloads (and other agents), so a
 * run can easily be 3x slower for reasons that have nothing to do with the code
 * under test.
 *
 * Crucially this measures *other* processes, not total load. Load average
 * includes the benchmark's own threads, so a perfectly clean single run on a
 * 4-core box still shows a load around 1.5 and would always look "busy" — a
 * check that fires on every run is worse than no check at all.
 */
import fs from "node:fs";
import os from "node:os";

/** Total busy CPU time of the whole machine, in clock ticks. */
function readTotalBusy() {
	const stat = fs.readFileSync("/proc/stat", "utf8");
	const line = stat.split("\n")[0]; // "cpu  user nice system idle iowait irq softirq steal ..."
	const parts = line.trim().split(/\s+/).slice(1).map(Number);
	const idle = parts[3] + (parts[4] ?? 0); // idle + iowait
	const total = parts.reduce((a, b) => a + b, 0);
	return { busy: total - idle, total };
}

/** CPU time consumed by this process and its descendants, in clock ticks. */
function readOwnCpu(rootPid = process.pid) {
	// Walk /proc once to build child links, then sum utime+stime for the subtree.
	const children = new Map();
	const cpu = new Map();
	for (const entry of fs.readdirSync("/proc")) {
		if (!/^\d+$/.test(entry)) continue;
		try {
			const stat = fs.readFileSync(`/proc/${entry}/stat`, "utf8");
			// `comm` may contain spaces and parens, so parse after the last ')'.
			const rest = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
			const ppid = Number(rest[1]);
			const utime = Number(rest[11]);
			const stime = Number(rest[12]);
			cpu.set(Number(entry), utime + stime);
			if (!children.has(ppid)) children.set(ppid, []);
			children.get(ppid).push(Number(entry));
		} catch {
			// Process vanished while reading; ignore.
		}
	}

	let total = 0;
	const stack = [rootPid];
	while (stack.length) {
		const pid = stack.pop();
		total += cpu.get(pid) ?? 0;
		for (const c of children.get(pid) ?? []) stack.push(c);
	}
	return total;
}

const TICKS_PER_SEC = 100; // Linux USER_HZ

/**
 * Begin sampling. Call `finish()` after the measured region; the returned
 * report says how much of the machine other processes used while we ran.
 */
export function startContentionSample() {
	const cores = os.cpus().length;
	const t0 = Date.now();
	let total0 = null;
	let own0 = 0;
	try {
		total0 = readTotalBusy();
		own0 = readOwnCpu();
	} catch {
		// /proc not available (unlikely on Linux); fall back to load average.
	}

	return function finish() {
		const elapsedSec = (Date.now() - t0) / 1000;
		let report;

		if (total0) {
			try {
				const total1 = readTotalBusy();
				const own1 = readOwnCpu();
				const totalDeltaSec = (total1.total - total0.total) / TICKS_PER_SEC;
				const ownDeltaSec = (own1 - own0) / TICKS_PER_SEC;
				// Everything the machine did that was not us.
				const otherSec = Math.max(0, totalDeltaSec - ownDeltaSec);
				const capacitySec = elapsedSec * cores;
				report = {
					method: "proc-cpu",
					cores,
					elapsedSec: Number(elapsedSec.toFixed(1)),
					ownCpuSec: Number(ownDeltaSec.toFixed(1)),
					otherCpuSec: Number(otherSec.toFixed(1)),
					// Fraction of total machine capacity consumed by other work.
					otherUtilisation: Number((otherSec / capacitySec).toFixed(2)),
				};
				// 0.5 means other processes used half the machine for the whole run.
				// Below that, timing is still broadly usable.
				report.quiet = report.otherUtilisation < 0.5;
				return report;
			} catch {
				// Fall through to load average.
			}
		}

		const load = os.loadavg();
		report = {
			method: "loadavg",
			cores,
			elapsedSec: Number(elapsedSec.toFixed(1)),
			load: { one: load[0], five: load[1], fifteen: load[2] },
			otherUtilisation: Number((load[0] / cores).toFixed(2)),
		};
		report.quiet = report.otherUtilisation < 1.0;
		return report;
	};
}
