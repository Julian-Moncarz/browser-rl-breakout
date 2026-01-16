# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run train             # Start RL training (runs indefinitely, Ctrl+C to stop)
npm run train -- --resume # Resume training from saved checkpoint
npm run play              # Open browser to watch latest model play
npm run play -- --best    # Open browser to watch best model play
npm test                  # Run environment tests
```

## Architecture

This is a reinforcement learning project that trains a DQN (Deep Q-Network) agent to play Breakout using TensorFlow.js.

### Core Components

**env.js** - Headless Breakout game environment with OpenAI Gym-style interface:
- `createEnv(config)` - Factory function returning environment object
- `reset()` - Returns initial observation (Float32Array of 13 elements)
- `step(action)` - Takes action (0=left, 1=stay, 2=right), returns `{obs, reward, done, info}`
- `launch()` - Launches the ball (must call after reset to start game)

**train-cli.mjs** - DQN training loop:
- Uses experience replay with ring buffer (50k capacity)
- Dual networks: Q-network and target network (updated every 1000 steps)
- Epsilon-greedy exploration with exponential decay
- Trains every 4 steps with batch size 64
- Saves checkpoint every 1000 steps to `./model/`
- Tracks best avg100 reward and saves to `./model.best/`
- Keeps one backup checkpoint at `./model.backup/`

**play.mjs** - Local HTTP server serving `watch.html` for browser-based model viewing

### Checkpoint Structure

```
./model/        # Latest checkpoint (auto-saved every 1000 steps)
./model.best/   # Best performing model (by avg100 reward)
./model.backup/ # Previous checkpoint (for recovery)
```

Each checkpoint contains:
- `model.json` + `*.bin` - TensorFlow.js model weights
- `state.json` - Training state (epsilon, totalSteps, episode, bestAvgReward)

### Observation Space

13-element Float32Array:
- [0-3]: Ball position and velocity (normalized)
- [4]: Paddle x position (normalized)
- [5]: Ball launched flag
- [6]: Lives remaining (normalized)
- [7]: Level (normalized, capped at 10)
- [8-12]: Bricks alive per row (normalized)

### Reward Structure

- Breaking brick: 10 × combo × level
- Losing life: -100
- Clearing level: +500
- Game over (all lives lost): -100
