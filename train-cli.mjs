import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createEnv } from './env.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, 'model');
const BACKUP_DIR = path.join(__dirname, 'model.backup');
const BEST_DIR = path.join(__dirname, 'model.best');
const LOG_FILE = path.join(__dirname, 'train-log.csv');

// Parse CLI args
const args = process.argv.slice(2);
const config = {
    resume: args.includes('--resume'),
    lr: parseFloat(args.find(a => a.startsWith('--lr='))?.split('=')[1] || '0.001'),
    batchSize: parseInt(args.find(a => a.startsWith('--batch='))?.split('=')[1] || '64'),
    gamma: parseFloat(args.find(a => a.startsWith('--gamma='))?.split('=')[1] || '0.99'),
};

const OBS_SIZE = 13;
const NUM_ACTIONS = 3;
const MEMORY_SIZE = 50000;
const BATCH_SIZE = config.batchSize;
const GAMMA = config.gamma;
const LEARNING_RATE = config.lr;
const EPSILON_START = 1.0;
const EPSILON_END = 0.01;
const EPSILON_DECAY = 0.9995;
const TARGET_UPDATE_FREQ = 1000;
const MAX_STEPS_PER_EPISODE = 10000;

// Ring buffer for experience replay
class RingBuffer {
    constructor(size) {
        this.size = size;
        this.buffer = [];
        this.idx = 0;
    }
    
    push(item) {
        if (this.buffer.length < this.size) {
            this.buffer.push(item);
        } else {
            this.buffer[this.idx] = item;
        }
        this.idx = (this.idx + 1) % this.size;
    }
    
    sample(count) {
        const batch = [];
        for (let i = 0; i < count; i++) {
            batch.push(this.buffer[Math.floor(Math.random() * this.buffer.length)]);
        }
        return batch;
    }
    
    length() {
        return this.buffer.length;
    }
}

let qNetwork, targetNetwork;
let memory = new RingBuffer(MEMORY_SIZE);
let epsilon = EPSILON_START;
let totalSteps = 0;
let episode = 0;
let running = true;
let rewardHistory = [];
let lossHistory = [];
let bestAvgReward = -Infinity;
let lastSaveStep = 0;

// Save checkpoint with backup
async function saveCheckpoint(dir, isBest = false) {
    const tempDir = dir + '.tmp';

    // Remove temp dir if exists from failed save
    if (fs.existsSync(tempDir)) {
        fs.rmSync(tempDir, { recursive: true });
    }

    // Save to temp directory first
    await qNetwork.save(`file://${tempDir}`);

    // Save training state
    const state = { epsilon, totalSteps, episode, bestAvgReward };
    fs.writeFileSync(path.join(tempDir, 'state.json'), JSON.stringify(state, null, 2));

    // Atomic swap: backup old, rename temp to target
    if (fs.existsSync(dir)) {
        if (!isBest) {
            // For regular saves, keep one backup
            if (fs.existsSync(BACKUP_DIR)) {
                fs.rmSync(BACKUP_DIR, { recursive: true });
            }
            fs.renameSync(dir, BACKUP_DIR);
        } else {
            fs.rmSync(dir, { recursive: true });
        }
    }
    fs.renameSync(tempDir, dir);
}

// Load checkpoint if exists
async function loadCheckpoint() {
    const modelPath = path.join(MODEL_DIR, 'model.json');
    const statePath = path.join(MODEL_DIR, 'state.json');

    if (!fs.existsSync(modelPath)) {
        console.log('No saved model found, starting fresh.\n');
        return false;
    }

    try {
        // Load model weights
        const loadedModel = await tf.loadLayersModel(`file://${modelPath}`);
        qNetwork.setWeights(loadedModel.getWeights());
        copyWeights(qNetwork, targetNetwork);
        loadedModel.dispose();

        // Load training state
        if (fs.existsSync(statePath)) {
            const state = JSON.parse(fs.readFileSync(statePath, 'utf-8'));
            epsilon = state.epsilon;
            totalSteps = state.totalSteps;
            episode = state.episode;
            bestAvgReward = state.bestAvgReward ?? -Infinity;
            lastSaveStep = totalSteps;
        }

        console.log(`Resumed from checkpoint:`);
        console.log(`  Episode: ${episode}, Steps: ${totalSteps}, Epsilon: ${epsilon.toFixed(4)}, Best Avg: ${bestAvgReward.toFixed(1)}\n`);
        return true;
    } catch (err) {
        console.error('Failed to load checkpoint:', err.message);
        console.log('Starting fresh.\n');
        return false;
    }
}

function createModel() {
    const model = tf.sequential({
        layers: [
            tf.layers.dense({ inputShape: [OBS_SIZE], units: 128, activation: 'relu' }),
            tf.layers.dense({ units: 128, activation: 'relu' }),
            tf.layers.dense({ units: 64, activation: 'relu' }),
            tf.layers.dense({ units: NUM_ACTIONS, activation: 'linear' })
        ]
    });
    model.compile({ optimizer: tf.train.adam(LEARNING_RATE), loss: 'meanSquaredError' });
    return model;
}

function copyWeights(source, target) {
    target.setWeights(source.getWeights());
}

function selectAction(state) {
    if (Math.random() < epsilon) {
        return Math.floor(Math.random() * NUM_ACTIONS);
    }
    return tf.tidy(() => {
        const stateTensor = tf.tensor2d([state]);
        const qValues = qNetwork.predict(stateTensor);
        return qValues.argMax(1).dataSync()[0];
    });
}

async function replay() {
    if (memory.length() < BATCH_SIZE) return 0;

    const batch = memory.sample(BATCH_SIZE);
    const states = batch.map(e => Array.from(e.state));
    const nextStates = batch.map(e => Array.from(e.nextState));
    const actions = batch.map(e => e.action);
    const rewards = batch.map(e => e.reward);
    const dones = batch.map(e => e.done ? 0 : 1);

    const statesTensor = tf.tensor2d(states);
    const nextStatesTensor = tf.tensor2d(nextStates);

    // Current Q-values
    const currentQs = qNetwork.predict(statesTensor);

    // Next Q-values from target network
    const nextQs = targetNetwork.predict(nextStatesTensor);
    const maxNextQs = nextQs.max(1);

    // Compute targets: reward + gamma * Q(s', best_action) * (1 - done)
    const rewardsTensor = tf.tensor1d(rewards);
    const donesTensor = tf.tensor1d(dones);
    const targetValues = rewardsTensor.add(tf.mul(maxNextQs, tf.mul(donesTensor, GAMMA)));

    // Build target Q-values: copy current, replace action col with target
    const currentQsData = currentQs.arraySync();
    const targetQsData = currentQsData.map(row => [...row]);
    const targetValuesData = targetValues.dataSync();
    for (let i = 0; i < BATCH_SIZE; i++) {
        targetQsData[i][actions[i]] = targetValuesData[i];
    }
    const targetTensor = tf.tensor2d(targetQsData);

    // Train - await the Promise to get actual loss value
    const loss = await qNetwork.trainOnBatch(statesTensor, targetTensor);

    // Clean up
    statesTensor.dispose();
    nextStatesTensor.dispose();
    currentQs.dispose();
    nextQs.dispose();
    maxNextQs.dispose();
    rewardsTensor.dispose();
    donesTensor.dispose();
    targetValues.dispose();
    targetTensor.dispose();

    return loss;
}

async function runEpisode(env) {
    let obs = env.reset();
    env.launch();
    let totalReward = 0;
    let steps = 0;

    while (steps < MAX_STEPS_PER_EPISODE && running) {
        const action = selectAction(obs);
        const { obs: nextObs, reward, done, info } = env.step(action);

        memory.push({ state: obs, action, reward, nextState: nextObs, done });
        totalReward += reward;
        obs = nextObs;
        steps++;
        totalSteps++;

        if (totalSteps % 4 === 0) {
            const loss = await replay();
            if (loss) lossHistory.push(loss);
        }

        if (totalSteps % TARGET_UPDATE_FREQ === 0) {
            copyWeights(qNetwork, targetNetwork);
        }

        if (done) break;
    }

    epsilon = Math.max(EPSILON_END, epsilon * EPSILON_DECAY);
    
    // Learning rate decay
    if (episode % 1000 === 0 && episode > 0) {
        const newLr = LEARNING_RATE * Math.pow(0.95, episode / 1000);
        qNetwork.optimizer.learningRate = Math.max(newLr, 0.0001);
    }
    
    return totalReward;
}

async function init() {
     console.log('Initializing TensorFlow.js...');
     // Force CPU backend (tfjs-node native backend incompatible with Node 24+)
     await tf.setBackend('cpu');
     await tf.ready();
     console.log(`Backend: ${tf.getBackend()}`);

     qNetwork = createModel();
     targetNetwork = createModel();
     copyWeights(qNetwork, targetNetwork);
     console.log('Models created\n');

     // Initialize CSV log file
     if (!fs.existsSync(LOG_FILE)) {
         fs.writeFileSync(LOG_FILE, 'episode,reward,avg100,best_avg,epsilon,steps_per_sec,loss,memory_usage,elapsed_sec\n');
     }

     // Check for --resume flag
     if (config.resume) {
         await loadCheckpoint();
     }

     return true;
 }

async function main() {
    await init();
    
    const env = createEnv();
    let startTime = Date.now();
    let lastUpdateTime = startTime;
    let lastUpdateSteps = 0;

    console.log('Training started. Press Ctrl+C to stop.\n');

    while (running) {
        episode++;
        const reward = await runEpisode(env);
        rewardHistory.push(reward);

        const now = Date.now();
        if (now - lastUpdateTime >= 1000 || episode % 10 === 0) {
            const elapsed = (now - startTime) / 1000;
            const stepsPerSec = (totalSteps - lastUpdateSteps) / ((now - lastUpdateTime) / 1000);
            
            const last100 = rewardHistory.slice(-100);
            const avgReward = last100.length > 0 ? last100.reduce((a, b) => a + b, 0) / last100.length : 0;
            
            const lossEma = lossHistory.length > 0 ? lossHistory.slice(-20).reduce((a, b) => a + b, 0) / 20 : 0;

            // Save every 1000 steps
            if (totalSteps - lastSaveStep >= 1000) {
                await saveCheckpoint(MODEL_DIR);
                lastSaveStep = totalSteps;

                // Save best model if avg reward improved
                if (avgReward > bestAvgReward && rewardHistory.length >= 100) {
                    bestAvgReward = avgReward;
                    await saveCheckpoint(BEST_DIR, true);
                }
            }

            console.log(
                `[${episode.toString().padStart(5)}] ` +
                `reward=${reward.toFixed(0).padStart(6)} ` +
                `avg100=${avgReward.toFixed(1).padStart(7)} ` +
                `best=${bestAvgReward === -Infinity ? '   N/A' : bestAvgReward.toFixed(1).padStart(6)} ` +
                `ε=${epsilon.toFixed(4)} ` +
                `steps/s=${stepsPerSec.toFixed(0).padStart(5)} ` +
                `loss=${lossEma.toFixed(4).padStart(7)} ` +
                `mem=${memory.length()}/${MEMORY_SIZE} ` +
                `elapsed=${elapsed.toFixed(1)}s`
            );

            // Log to CSV
            const bestAvgValue = bestAvgReward === -Infinity ? '' : bestAvgReward.toFixed(1);
            const logLine = `${episode},${reward.toFixed(0)},${avgReward.toFixed(1)},${bestAvgValue},${epsilon.toFixed(4)},${stepsPerSec.toFixed(0)},${lossEma.toFixed(4)},${memory.length()},${elapsed.toFixed(1)}`;
            fs.appendFileSync(LOG_FILE, logLine + '\n');

            lastUpdateTime = now;
            lastUpdateSteps = totalSteps;
        }
    }

    console.log('\nTraining stopped');
    process.exit(0);
}

// Handle Ctrl+C gracefully
process.on('SIGINT', async () => {
    console.log('\nShutting down...');
    running = false;

    // Save final checkpoint
    console.log('Saving checkpoint...');
    try {
        await saveCheckpoint(MODEL_DIR);
        console.log('Checkpoint saved to ./model/');
    } catch (err) {
        console.error('Failed to save checkpoint:', err.message);
    }

    process.exit(0);
});

main().catch(e => {
    console.error(e);
    process.exit(1);
});
