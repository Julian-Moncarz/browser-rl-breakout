# Training Harness Refactor

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove browser-based training, share env.js with the viewer, and optimize the terminal training harness for maximum efficiency.

**Architecture:** Pure Node.js training with tfjs-node (auto-detects GPU). Browser is view-only. Single source of truth for game logic (env.js) used by both training and browser viewer via ES modules.

**Tech Stack:** TensorFlow.js Node (tfjs-node/tfjs-node-gpu auto-detected), ES modules, simple HTTP server for viewer.

---

## Current State Analysis

**Files to DELETE:**
- `train.html` - Browser-based training (slow, unnecessary)
- `game.html` - Human-playable game (duplicates env.js)
- `test-env.html` - Browser tests (redundant with test-env.mjs)

**Files to MODIFY:**
- `watch.html` - Currently duplicates game logic; refactor to import env.js
- `train-cli.mjs` - Add efficiency improvements
- `package.json` - Clean up dependencies

**Files to KEEP AS-IS:**
- `env.js` - Headless environment (already good)
- `play.mjs` - HTTP server (minor tweaks)
- `test-env.mjs` - Tests (already good)

---

## Task 1: Delete Browser Training Files

**Files:**
- Delete: `train.html`
- Delete: `game.html`
- Delete: `test-env.html`

**Step 1: Remove the files**

```bash
rm train.html game.html test-env.html
```

**Step 2: Verify removal**

Run: `ls *.html`
Expected: Only `watch.html` remains

**Step 3: Commit**

```bash
git add -A && git commit -m "chore: remove browser-based training files

Deleted train.html, game.html, test-env.html - training now terminal-only"
```

---

## Task 2: Refactor watch.html to Use Shared env.js

**Files:**
- Modify: `watch.html`
- Modify: `play.mjs` (serve env.js with correct MIME type)

**Step 1: Write the failing test**

Open `watch.html` in browser after changes, verify model loads and plays correctly.
Manual test - no automated test file needed.

**Step 2: Update play.mjs to serve env.js**

The server already serves static files, but ensure `.mjs` extension is handled:

```javascript
const MIME_TYPES = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.mjs': 'application/javascript',  // Add this
    '.json': 'application/json',
    '.css': 'text/css',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.bin': 'application/octet-stream'
};
```

**Step 3: Rewrite watch.html to import env.js**

Replace the duplicated game logic with an ES module import. The key changes:
1. Import `createEnv` from `/env.js`
2. Remove all duplicated game state and physics
3. Use `env.getState()` for rendering

```html
<script type="module">
    import { createEnv } from '/env.js';

    // ... rest of viewer code uses createEnv() instead of duplicated logic
</script>
```

Full implementation: Replace lines 107-501 with modular version that:
- Creates env via `createEnv()`
- Uses `env.getState()` for ball/paddle/brick positions
- Renders based on env state
- Removes ~200 lines of duplicated game logic

**Step 4: Verify viewer still works**

Run: `npm run play`
Expected: Browser opens, model loads, AI plays correctly

**Step 5: Commit**

```bash
git add watch.html play.mjs && git commit -m "refactor: watch.html now imports shared env.js

- Removes ~200 lines of duplicated game logic
- Single source of truth for Breakout physics
- play.mjs serves .mjs files with correct MIME type"
```

---

## Task 3: Optimize Training Harness

**Files:**
- Modify: `train-cli.mjs`
- Modify: `package.json`

**Step 1: Update package.json dependencies**

Remove browser tfjs (not needed for training), ensure tfjs-node is explicit:

```json
{
  "name": "breakout-rl",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "train": "node train-cli.mjs",
    "play": "node play.mjs",
    "test": "node test-env.mjs"
  },
  "dependencies": {
    "@tensorflow/tfjs-node": "^4.17.0",
    "asciichart": "^1.5.25"
  }
}
```

Note: Removed `@tensorflow/tfjs` (browser version), `blessed`, `blessed-contrib` (unused).

**Step 2: Add configurable hyperparameters via CLI**

Add argparse for configuration:

```javascript
// At top of train-cli.mjs
const args = process.argv.slice(2);
const config = {
    resume: args.includes('--resume'),
    lr: parseFloat(args.find(a => a.startsWith('--lr='))?.split('=')[1] || '0.001'),
    batchSize: parseInt(args.find(a => a.startsWith('--batch='))?.split('=')[1] || '64'),
    gamma: parseFloat(args.find(a => a.startsWith('--gamma='))?.split('=')[1] || '0.99'),
};
```

**Step 3: Add Double DQN (reduces overestimation)**

In the `replay()` function, change target computation:

```javascript
// Current (standard DQN):
// targetValues = reward + gamma * max(targetNetwork.predict(nextState))

// Double DQN:
// 1. Use qNetwork to SELECT the best action
// 2. Use targetNetwork to EVALUATE that action
const nextQsOnline = qNetwork.predict(nextStatesTensor);
const bestActions = nextQsOnline.argMax(1);
const nextQsTarget = targetNetwork.predict(nextStatesTensor);
// Gather Q-values for best actions from target network
```

**Step 4: Add learning rate decay**

```javascript
// After each episode, decay learning rate
if (episode % 1000 === 0 && episode > 0) {
    const newLr = LEARNING_RATE * Math.pow(0.95, episode / 1000);
    qNetwork.optimizer.learningRate = Math.max(newLr, 0.0001);
}
```

**Step 5: Run tests**

Run: `npm test`
Expected: All tests pass

**Step 6: Run training briefly to verify**

Run: `timeout 30 npm run train || true`
Expected: Training starts, shows stats, no errors

**Step 7: Commit**

```bash
git add train-cli.mjs package.json && git commit -m "feat: optimize training harness

- Simplified dependencies (removed browser tfjs, unused blessed)
- Double DQN for better Q-value estimates
- Learning rate decay
- CLI config options (--lr=, --batch=, --gamma=)"
```

---

## Task 4: Clean Up and Final Verification

**Files:**
- Modify: `CLAUDE.md` (update if needed)

**Step 1: Reinstall dependencies**

```bash
rm -rf node_modules package-lock.json && npm install
```

**Step 2: Run all tests**

Run: `npm test`
Expected: All 10+ tests pass

**Step 3: Verify training works**

Run: `timeout 60 npm run train || true`
Expected: Training runs, displays stats (steps/s, reward, epsilon)

**Step 4: Verify viewer works**

Run: `npm run play &` then check browser
Expected: Viewer loads, can load model if one exists

**Step 5: Update CLAUDE.md if commands changed**

Verify the documented commands still work.

**Step 6: Final commit**

```bash
git add -A && git commit -m "chore: clean up after refactor

- Reinstalled dependencies
- Verified all commands work"
```

---

## Summary of Changes

| Before | After |
|--------|-------|
| Browser training (train.html) | Terminal-only training |
| Duplicated game logic in 3 files | Single env.js shared everywhere |
| Standard DQN | Double DQN |
| Fixed hyperparameters | CLI-configurable |
| 6 dependencies | 2 dependencies |

**Performance expectations:**
- Training should run at 1000+ steps/sec on CPU
- GPU (if available) should be auto-detected by tfjs-node
- Memory usage should stay stable (ring buffer)
