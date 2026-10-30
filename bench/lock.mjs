/**
 * Serialises benchmark runs across processes.
 *
 * Several agents share this machine. Two concurrent benchmark runs contend for
 * the same CPU/GPU and make both sets of numbers meaningless, and on a CPU
 * fallback renderer they also contend for the same cores. A coarse exclusive
 * lock is the simplest thing that is actually correct.
 *
 * Stale locks (a crashed run) are broken after STALE_MS so a dead agent cannot
 * wedge the team.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const LOCK = path.join(HERE, ".cache", "bench.lock");
const STALE_MS = 45 * 60 * 1000;

function pidAlive(pid) {
	try {
		process.kill(pid, 0);
		return true;
	} catch {
		return false;
	}
}

/**
 * Acquire the lock. Returns a release function.
 *
 * @param {{ wait?: boolean, onWait?: (holder: object) => void }} opts
 */
export function acquireLock({ wait = true, onWait } = {}) {
	fs.mkdirSync(path.dirname(LOCK), { recursive: true });

	for (;;) {
		try {
			// 'wx' fails if the file exists, which is the atomicity we need.
			const fd = fs.openSync(LOCK, "wx");
			fs.writeSync(
				fd,
				JSON.stringify({ pid: process.pid, startedAt: Date.now(), argv: process.argv.slice(2) }),
			);
			fs.closeSync(fd);
			return () => {
				try {
					fs.unlinkSync(LOCK);
				} catch {
					// Already gone; nothing to do.
				}
			};
		} catch (e) {
			if (e.code !== "EEXIST") throw e;

			let holder = {};
			try {
				holder = JSON.parse(fs.readFileSync(LOCK, "utf8"));
			} catch {
				// Mid-write or corrupt; treat as stale below.
			}

			const age = Date.now() - (holder.startedAt ?? 0);
			const dead = holder.pid ? !pidAlive(holder.pid) : true;
			if (dead || age > STALE_MS) {
				console.warn(
					`[bench] breaking stale lock (pid ${holder.pid ?? "?"}, age ${Math.round(age / 60000)}m)`,
				);
				try {
					fs.unlinkSync(LOCK);
				} catch {
					// Someone else broke it first; loop and retry.
				}
				continue;
			}

			if (!wait) {
				throw new Error(
					`another benchmark is running (pid ${holder.pid}). ` +
						`Benchmarks are serialised so concurrent runs cannot corrupt each other.`,
				);
			}

			onWait?.(holder);
			// Busy-wait with a short sleep. Kept dependency-free on purpose.
			const until = Date.now() + 5000;
			while (Date.now() < until) {
				// no-op
			}
		}
	}
}
