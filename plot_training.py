#!/usr/bin/env python3
"""
Simple script to visualize training metrics from train-log.csv
Usage: python3 plot_training.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys

try:
    df = pd.read_csv('train-log.csv')
except FileNotFoundError:
    print("Error: train-log.csv not found. Run training first.")
    sys.exit(1)

if df.empty:
    print("Error: train-log.csv is empty.")
    sys.exit(1)

# Create a 2x2 subplot figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Breakout RL Training Metrics', fontsize=16, fontweight='bold')

# Plot 1: Reward over episodes
ax = axes[0, 0]
ax.plot(df['episode'], df['reward'], alpha=0.6, label='Episode Reward')
ax.plot(df['episode'], df['avg100'], linewidth=2, label='Avg100 Reward', color='orange')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_title('Rewards')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Loss over episodes
ax = axes[0, 1]
ax.plot(df['episode'], df['loss'], color='red', alpha=0.7)
ax.set_xlabel('Episode')
ax.set_ylabel('Loss (EMA)')
ax.set_title('Training Loss')
ax.grid(True, alpha=0.3)

# Plot 3: Epsilon decay
ax = axes[1, 0]
ax.plot(df['episode'], df['epsilon'], color='green')
ax.set_xlabel('Episode')
ax.set_ylabel('Epsilon')
ax.set_title('Exploration Rate Decay')
ax.grid(True, alpha=0.3)

# Plot 4: Steps per second (throughput)
ax = axes[1, 1]
ax.plot(df['episode'], df['steps_per_sec'], color='purple', alpha=0.7)
ax.set_xlabel('Episode')
ax.set_ylabel('Steps/Second')
ax.set_title('Training Throughput')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to training_metrics.png")
plt.show()
