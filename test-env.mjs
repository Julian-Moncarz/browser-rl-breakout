/**
 * Node.js test runner for the Breakout environment
 */

import { createEnv } from './env.js';

let passed = 0;
let failed = 0;

function assert(condition, message) {
    if (condition) {
        console.log(`✓ ${message}`);
        passed++;
    } else {
        console.log(`✗ ${message}`);
        failed++;
    }
}

console.log('=== Breakout Environment Tests ===\n');

// Test 1: createEnv returns an object with required methods
const env = createEnv();
assert(typeof env.reset === 'function', 'env.reset is a function');
assert(typeof env.step === 'function', 'env.step is a function');
assert(typeof env.getObs === 'function', 'env.getObs is a function');

// Test 2: reset() returns an observation
const obs1 = env.reset();
assert(obs1 instanceof Float32Array, 'reset() returns Float32Array');
assert(obs1.length === 13, `observation has 13 elements (got ${obs1.length})`);

// Test 3: observations are normalized [0,1] or small range for velocities
assert(obs1[0] >= 0 && obs1[0] <= 1, `ball.x is normalized [0,1] (got ${obs1[0]})`);
assert(obs1[1] >= 0 && obs1[1] <= 1, `ball.y is normalized [0,1] (got ${obs1[1]})`);
assert(obs1[4] >= 0 && obs1[4] <= 1, `paddle.x is normalized [0,1] (got ${obs1[4]})`);

// Test 4: after reset, ball is not launched
assert(obs1[5] === 0, 'ball not launched after reset');

// Test 5: step() returns {obs, reward, done, info}
const result = env.step(1); // action=stay
assert(result.obs instanceof Float32Array, 'step().obs is Float32Array');
assert(typeof result.reward === 'number', 'step().reward is number');
assert(typeof result.done === 'boolean', 'step().done is boolean');
assert(typeof result.info === 'object', 'step().info is object');

// Test 6: actions affect paddle position
env.reset();
const obs2 = env.getObs();
const paddleXBefore = obs2[4];

// Move left multiple times
for (let i = 0; i < 10; i++) env.step(0);
const paddleXAfterLeft = env.getObs()[4];
assert(paddleXAfterLeft < paddleXBefore, `action=0 (left) moves paddle left (${paddleXBefore} -> ${paddleXAfterLeft})`);

// Move right multiple times
for (let i = 0; i < 20; i++) env.step(2);
const paddleXAfterRight = env.getObs()[4];
assert(paddleXAfterRight > paddleXAfterLeft, `action=2 (right) moves paddle right (${paddleXAfterLeft} -> ${paddleXAfterRight})`);

// Test 7: ball launches and moves after step with launched=true
env.reset();
env.launch();
const obsBeforeMove = env.getObs();
env.step(1);
const obsAfterMove = env.getObs();
assert(obsAfterMove[5] === 1, 'ball is launched');
assert(obsAfterMove[1] !== obsBeforeMove[1], `ball.y changes after step when launched (${obsBeforeMove[1]} -> ${obsAfterMove[1]})`);

// Test 8: hitting bottom loses a life and gives negative reward
env.reset();
env.launch();
let totalReward = 0;
let lifeLost = false;
for (let i = 0; i < 2000 && !lifeLost; i++) {
    const r = env.step(1); // stay still, ball will eventually fall
    totalReward += r.reward;
    if (r.info.lifeLost) lifeLost = true;
}
assert(lifeLost, 'life is lost when ball hits bottom');
assert(totalReward < 0, `negative reward when life lost (got ${totalReward})`);

// Test 9: breaking a brick gives positive reward
env.reset();
env.launch();
let brickBroken = false;
let brickReward = 0;
for (let i = 0; i < 500 && !brickBroken; i++) {
    // Chase the ball with paddle
    const obs = env.getObs();
    const ballX = obs[0];
    const paddleX = obs[4];
    const action = ballX < paddleX ? 0 : ballX > paddleX + 0.15 ? 2 : 1;
    const r = env.step(action);
    if (r.reward > 0) {
        brickBroken = true;
        brickReward = r.reward;
    }
}
assert(brickBroken, 'brick can be broken');
assert(brickReward > 0, `positive reward for breaking brick (got ${brickReward})`);

// Test 10: game ends (done=true) when all lives lost
env.reset();
env.launch();
let done = false;
let steps = 0;
for (let i = 0; i < 10000 && !done; i++) {
    const r = env.step(1); // stay still, lose all lives
    done = r.done;
    steps++;
}
assert(done, `game ends when all lives lost (took ${steps} steps)`);

// Summary
console.log(`\n${'='.repeat(40)}`);
console.log(`Passed: ${passed}, Failed: ${failed}`);
process.exit(failed > 0 ? 1 : 0);
