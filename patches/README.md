# Maintaining a patch to @jax-js/jax

Custom WebGPU kernels sometimes need to reach inside the library. When that is
genuinely necessary — after fusion and restructuring in our own code have been
ruled out — we keep the change as a patch in this directory rather than
vendoring the package or editing `node_modules` by hand.

## Why not edit node_modules

`node_modules` is generated. Hand edits are invisible to review, disappear on a
fresh install, and silently make every benchmark unreproducible. A patch file is
reviewable, diffable, and reproducible.

## Layout

```
patches/
  README.md                 # this file
  jax-js-<topic>.patch      # one patch per concern
scripts/
  apply-patches.mjs         # applies every patch in patches/
```

`package.json` runs `apply-patches.mjs` from `postinstall`, so a clean
`npm install` reproduces the patched tree.

## Creating a patch

1. Pin the version first. `@jax-js/jax` is pinned in `package.json`; bumping it
   invalidates the patch, so do that deliberately.
2. Copy the pristine package aside so you have something to diff against:

   ```sh
   cp -r node_modules/@jax-js/jax /tmp/jax-js-pristine
   ```

3. Make your change in `node_modules/@jax-js/jax`.
4. Generate a patch:

   ```sh
   diff -ruN /tmp/jax-js-pristine node_modules/@jax-js/jax > patches/jax-js-<topic>.patch
   ```

   Fix the paths inside the patch so they apply from the repo root (they should
   read `node_modules/@jax-js/jax/...`).
5. Verify it applies to a clean tree:

   ```sh
   rm -rf node_modules/@jax-js/jax
   npm install     # or restore from cache
   node scripts/apply-patches.mjs
   ```

## Rules

- **One concern per patch.** A patch that adds a kernel *and* changes scheduling
  cannot be evaluated or reverted independently.
- **Document it at the top of the patch** — or in a matching `.md` next to it —
  with what it changes, why, and what it is expected to buy.
- **Announce it in `#perf` before applying.** It affects every agent's build.
- **Keep it small.** A large patch is a fork in all but name; that decision
  needs an explicit conversation.
- **Delete it when upstream lands the equivalent change.** Patches rot.

## Verifying

```sh
node scripts/apply-patches.mjs --check   # non-zero if a patch no longer applies
npm run bench:check                      # numerics still correct
```
