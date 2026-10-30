/**
 * Serialises benchmark runs across processes.
 *
 * Several agents share this machine. Two concurrent benchmark runs contend for
 * the same CPU/GPU and make both sets of numbers meaningless, and on a CPU
 * fallback renderer they also contend for the same cores. A coarse exclusive
 * lock is the simplest thing that is actually correct.
 *
 * Design notes:
 *  - The lock is only broken when the holder process is gone. A long benchmark
 *    (a full suite on a software renderer takes hours) holds it legitimately,
 *    so age alone must never be treated as staleness.
 *  - Waiting sleeps instead of spinning. A busy-wait burns a core, which slows
 *    down the run holding the lock and perturbs the very timings we want.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const LOCK = path.join(HERE, ".cache", "bench.lock");

function pidAlive(pid) {
	if (!Number.isInteger(pid) || pid <= 0) return false;
	try {
		process.kill(pid, 0);
		return true;
	} catch (e) {
		// EPERM means the process exists but belongs to someone else.
		return e.code === "EPERM";
	}
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function readHolder() {
	try {
		return JSON.parse(fs.readFileSync(LOCK, "utf8"));
	} catch {
		// Mid-write, empty, or corrupt.
		return null;
	}
}

/**
 * Acquire the lock. Returns an async release function.
 *
 * @param {{ wait?: boolean, onWait?: (holder: object) => void }} opts
 */
export async function acquireLock({ wait = true, onWait } = {}) {
	fs.mkdirSync(path.dirname(LOCK), { recursive: true });
	let announced = false;

	for (;;) {
		try {
			// 'wx' fails if the file exists, which is the atomicity we need.
			const fd = fs.openSync(LOCK, "wx");
			fs.writeSync(
				fd,
				JSON.stringify({ pid: process.pid, startedAt: Date.now(), argv: process.argv.slice(2) }),
			);
			fs.closeSync(fd);
			return async () => {
				try {
					fs.unlinkSync(LOCK);
				} catch {
					// Already gone; nothing to do.
				}
			};
		} catch (e) {
			if (e.code !== "EEXIST") throw e;

			const holder = readHolder();

			// A lock file we cannot parse is either being written right now or
			// was left behind by a crash. Give it a moment before judging.
			if (!holder) {
				await sleep(250);
				const again = readHolder();
				if (!again) {
					// Still unparseable and nobody claims it: safe to clear. There is
					// a small race here between two waiters; unlinking a file that is
					// already gone is harmless, and the 'wx' create below re-serialises
					// whoever wins.
					try {
						fs.unlinkSync(LOCK);
					} catch {
						// Someone else cleared it.
					}
				}
				continue;
			}

			if (!pidAlive(holder.pid)) {
				console.warn(`[bench] clearing lock left by dead process ${holder.pid}`);
				try {
					fs.unlinkSync(LOCK);
				} catch {
					// Another waiter got there first.
				}
				continue;
			}

			if (!wait) {
				throw new Error(
					`another benchmark is running (pid ${holder.pid}). ` +
						`Benchmarks are serialised so concurrent runs cannot corrupt each other.`,
				);
			}

			if (!announced) {
				onWait?.(holder);
				announced = true;
			}
			await sleep(1000);
		}
	}
}
