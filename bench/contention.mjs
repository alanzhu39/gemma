/**
 * Records whether the machine was quiet enough for timing to mean anything.
 *
 * A benchmark is only comparable to another benchmark if both ran without
 * competing for CPU/GPU. This box has other workloads (and other agents), so a
 * run can easily be 3x slower for reasons that have nothing to do with the code
 * under test. Recording the load makes that visible in the result file instead
 * of turning into a mysterious regression later.
 */
import os from "node:os";

function readLoadAvg() {
	const [a, b, c] = os.loadavg();
	return { one: a, five: b, fifteen: c };
}

/**
 * Sample system load around a measured region and report whether it was clean.
 * `cores` is the machine's core count; load is normalised against it.
 */
export function sampleContention(cores = os.cpus().length) {
	const load = readLoadAvg();
	const busy = load.one / cores;
	return {
		cores,
		load,
		/** load average / cores: > 1 means more runnable work than cores. */
		utilisation: Number(busy.toFixed(2)),
		/**
		 * A run is "quiet" when utilisation stays under 1: the benchmark can
		 * actually get a core when it wants one.
		 */
		quiet: busy < 1.0,
	};
}
