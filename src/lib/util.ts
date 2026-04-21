export function stringify(s: string) {
	return JSON.stringify(s).slice(1, -1);
}

export function clamp(n: number, { min, max }: { min?: number; max?: number }) {
	if (min !== undefined) {
		n = Math.max(n, min);
	}
	if (max !== undefined) {
		n = Math.min(n, max);
	}
	return n;
}
