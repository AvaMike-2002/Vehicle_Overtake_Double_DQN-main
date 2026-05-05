"""
train_run.py
------------
Generate convergence plots from trained model data with proper axis marking.

Usage
-----
    python train_run.py

This script loads training logs and generates:
- Loss convergence plot (x-axis: 0 to 3000 steps, y-axis: Loss Value)
- Reward convergence plot (x-axis: Episode Number, y-axis: Mean Reward)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from scipy.ndimage import uniform_filter1d
import matplotlib.ticker as ticker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_training_data():
    """
    Load training loss and reward data from training logs.

    Returns
    -------
    tuple
        (step_losses, episode_rewards) or (None, None) if not found
    """
    logs_dir = os.path.join("files", "training", "logs")

    step_losses = None
    episode_rewards = None

    # Load loss data from Excel
    loss_path = os.path.join(logs_dir, "training_loss.xlsx")
    try:
        if os.path.exists(loss_path):
            loss_df = pd.read_excel(loss_path)
            if 'Loss' in loss_df.columns:
                step_losses = loss_df['Loss'].values
            elif 'value' in loss_df.columns:
                step_losses = loss_df['value'].values
            logger.info(f"Loaded {len(step_losses)} loss values from {loss_path}")
    except Exception as e:
        logger.warning(f"Could not load loss data: {e}")

    # Load reward data from CSV
    rewards_path = os.path.join(logs_dir, "episodic_reward.csv")
    try:
        if os.path.exists(rewards_path):
            reward_df = pd.read_csv(rewards_path)
            if 'AvgReward' in reward_df.columns:
                episode_rewards = reward_df['AvgReward'].values
            elif 'value' in reward_df.columns:
                episode_rewards = reward_df['value'].values
            logger.info(f"Loaded {len(episode_rewards)} reward values from {rewards_path}")
    except Exception as e:
        logger.warning(f"Could not load reward data: {e}")

    # Try alternative paths
    if step_losses is None:
        alt_loss_path = os.path.join("files", "training", "my_logs", "loss.csv")
        if os.path.exists(alt_loss_path):
            try:
                loss_df = pd.read_csv(alt_loss_path)
                if 'value' in loss_df.columns:
                    step_losses = loss_df['value'].values
                    logger.info(f"Loaded {len(step_losses)} loss values from {alt_loss_path}")
            except Exception as e:
                logger.warning(f"Could not load from {alt_loss_path}: {e}")

    return step_losses, episode_rewards


def plot_loss_convergence(step_losses, save_dir="plots"):
    """
    Plot loss convergence with proper axis marking.
    - X-axis: Training steps from 0 to 3000 (0, 500, 1.0k, 1.5k, 2.0k, 2.5k, 3.0k)
    - Y-axis: Loss Value with appropriate scale
    """
    if step_losses is None or len(step_losses) == 0:
        logger.error("No loss data available for convergence plot")
        return None

    os.makedirs(save_dir, exist_ok=True)

    # Limit to first 3000 steps
    max_steps_to_show = 3000
    if len(step_losses) > max_steps_to_show:
        steps_to_plot = step_losses[:max_steps_to_show]
        logger.info(f"Limiting loss convergence plot to first {max_steps_to_show} steps")
    else:
        steps_to_plot = step_losses

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Apply smoothing
    window_size = min(50, len(steps_to_plot) // 10)
    if window_size < 5:
        window_size = 5

    smoothed_loss = uniform_filter1d(steps_to_plot, size=window_size, mode='nearest')

    # Create x-axis values
    steps = np.arange(len(steps_to_plot))

    # Plot the convergence line
    ax.plot(steps, smoothed_loss, color="#4472C4", linewidth=2, label="Training Loss")

    # Set title and labels
    ax.set_title("Training Loss Progression", fontsize=14, fontweight="bold",
                 pad=15, color="#2C3E50")
    ax.set_xlabel("Training Steps", fontsize=12, color="#34495E", fontweight="bold")
    ax.set_ylabel("Loss Value", fontsize=12, color="#34495E", fontweight="bold")

    # Set x-axis limit to exactly 3000
    ax.set_xlim([0, max_steps_to_show])

    # Format x-axis with 'k' notation
    def x_format(x, p):
        if x >= 1000:
            return f'{x/1000:.1f}k'
        return f'{int(x)}'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(x_format))

    # Set specific x-axis ticks (0, 500, 1000, 1500, 2000, 2500, 3000)
    x_ticks = [0, 500, 1000, 1500, 2000, 2500, 3000]
    ax.set_xticks(x_ticks)

    # Set x-tick labels with proper formatting
    x_labels = ['0', '500', '1.0k', '1.5k', '2.0k', '2.5k', '3.0k']
    ax.set_xticklabels(x_labels)

    # Y-axis formatting
    y_max = max(smoothed_loss) * 1.1
    y_min = min(0, min(smoothed_loss))

    # Use scientific notation if values are very small
    if y_max < 0.01:
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-3,-2))
    else:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

    # Add grid with proper styling
    ax.grid(True, linestyle="--", alpha=0.6, color="#CCCCCC", linewidth=0.8)

    # Style spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # Style ticks
    ax.tick_params(colors="#555555", labelsize=10, width=1.5, length=6)

    # Add light background grid lines
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", alpha=0.3, color="#CCCCCC", linewidth=0.8)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "loss_convergence.png")
    plt.savefig(save_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close()

    logger.info(f"Loss convergence plot saved to {save_path}")

    # Print statistics
    final_loss = smoothed_loss[-1]
    initial_loss = smoothed_loss[0]
    print(f"\nLoss Convergence:")
    print(f"  Steps shown: 0 to {max_steps_to_show}")
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Reduction: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")

    return save_path


def plot_reward_convergence(episode_rewards, save_dir="plots"):
    """
    Plot reward convergence with proper axis marking.
    - X-axis: Episode Number
    - Y-axis: Mean Reward
    """
    if episode_rewards is None or len(episode_rewards) == 0:
        logger.error("No reward data available for convergence plot")
        return None

    os.makedirs(save_dir, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Create x-axis values (episodes)
    episodes = np.arange(len(episode_rewards))

    # Calculate moving average for cleaner trend line
    avg_window = min(20, len(episode_rewards) // 10)
    if avg_window < 3:
        avg_window = 3

    moving_avg = []
    for i in range(len(episode_rewards)):
        start = max(0, i - avg_window + 1)
        moving_avg.append(np.mean(episode_rewards[start:i+1]))

    # Apply smoothing
    window_size = min(10, len(moving_avg) // 20)
    if window_size < 2:
        window_size = 2

    smoothed_rewards = uniform_filter1d(moving_avg, size=window_size, mode='nearest')

    # Plot the convergence line
    ax.plot(episodes, smoothed_rewards, color="#00B050", linewidth=2, label="Mean Reward")

    # Set title and labels
    ax.set_title("Increasing Rewards Per Episode", fontsize=14, fontweight="bold",
                 pad=15, color="#2C3E50")
    ax.set_xlabel("Episode Number", fontsize=12, color="#34495E", fontweight="bold")
    ax.set_ylabel("Mean Reward", fontsize=12, color="#34495E", fontweight="bold")

    # Format x-axis with 'k' notation if needed
    def x_format(x, p):
        if x >= 1000:
            return f'{x/1000:.1f}k'
        return f'{int(x)}'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(x_format))

    # Set x-axis ticks at reasonable intervals
    max_episodes = episodes[-1]
    num_ticks = min(8, max_episodes // 100 + 2)
    x_ticks = np.linspace(0, max_episodes, num_ticks, dtype=int)
    ax.set_xticks(x_ticks)

    # Y-axis formatting with 3 decimal places
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    # Set reasonable y-axis limits
    y_min = min(0, min(smoothed_rewards) * 0.9)
    y_max = max(smoothed_rewards) * 1.05
    ax.set_ylim([y_min, y_max])

    # Add grid with proper styling
    ax.grid(True, linestyle="--", alpha=0.6, color="#CCCCCC", linewidth=0.8)

    # Style spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # Style ticks
    ax.tick_params(colors="#555555", labelsize=10, width=1.5, length=6)

    # Add light background grid lines
    ax.set_axisbelow(True)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "reward_convergence.png")
    plt.savefig(save_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close()

    logger.info(f"Reward convergence plot saved to {save_path}")

    # Print statistics
    initial_reward = smoothed_rewards[0]
    final_reward = smoothed_rewards[-1]
    best_reward = max(episode_rewards)
    best_episode = np.argmax(episode_rewards)

    print(f"\nReward Convergence:")
    print(f"  Initial reward: {initial_reward:.3f}")
    print(f"  Final reward: {final_reward:.3f}")
    print(f"  Best reward: {best_reward:.3f} (Episode {best_episode})")
    print(f"  Improvement: {final_reward - initial_reward:+.3f}")

    return save_path


def plot_combined_convergence(step_losses, episode_rewards, save_dir="plots"):
    """
    Create a combined figure with both convergence plots side by side
    with proper axis marking like the reference images.
    """
    if step_losses is None or episode_rewards is None:
        logger.warning("Insufficient data for combined plot")
        return None

    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle("Training Convergence Analysis", fontsize=16, fontweight="bold", y=1.02)

    # ==========================================
    # Loss subplot (left)
    # ==========================================
    max_steps_to_show = 3000
    if len(step_losses) > max_steps_to_show:
        steps_to_plot = step_losses[:max_steps_to_show]
    else:
        steps_to_plot = step_losses

    steps = np.arange(len(steps_to_plot))
    window_size = min(50, len(steps_to_plot) // 10)
    if window_size < 5:
        window_size = 5
    smoothed_loss = uniform_filter1d(steps_to_plot, size=window_size, mode='nearest')

    ax1.plot(steps, smoothed_loss, color="#4472C4", linewidth=2.5, label="Loss")

    ax1.set_title("Training Loss Progression", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Training Steps", fontsize=11, fontweight="bold", color="#34495E")
    ax1.set_ylabel("Loss Value", fontsize=11, fontweight="bold", color="#34495E")

    # X-axis: 0 to 3000 with proper labels
    ax1.set_xlim([0, max_steps_to_show])

    def x_format(x, p):
        if x >= 1000:
            return f'{x/1000:.1f}k'
        return f'{int(x)}'

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(x_format))
    x_ticks = [0, 500, 1000, 1500, 2000, 2500, 3000]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(['0', '500', '1.0k', '1.5k', '2.0k', '2.5k', '3.0k'])

    # Y-axis formatting
    if max(smoothed_loss) < 0.01:
        ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(-3,-2))

    # Grid and styling
    ax1.grid(True, linestyle="--", alpha=0.4, color="#CCCCCC", linewidth=0.8)
    ax1.set_facecolor("#FAFAFA")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#333333")
    ax1.spines["bottom"].set_color("#333333")
    ax1.spines["left"].set_linewidth(1.2)
    ax1.spines["bottom"].set_linewidth(1.2)
    ax1.tick_params(colors="#555555", labelsize=10, width=1.2, length=5)

    # ==========================================
    # Reward subplot (right)
    # ==========================================
    episodes = np.arange(len(episode_rewards))
    avg_window = min(20, len(episode_rewards) // 10)
    if avg_window < 3:
        avg_window = 3

    moving_avg = []
    for i in range(len(episode_rewards)):
        start = max(0, i - avg_window + 1)
        moving_avg.append(np.mean(episode_rewards[start:i+1]))

    window_size_reward = min(10, len(moving_avg) // 20)
    if window_size_reward < 2:
        window_size_reward = 2
    smoothed_rewards = uniform_filter1d(moving_avg, size=window_size_reward, mode='nearest')

    ax2.plot(episodes, smoothed_rewards, color="#00B050", linewidth=2.5, label="Reward")

    ax2.set_title("Increasing Rewards Per Episode", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Episode Number", fontsize=11, fontweight="bold", color="#34495E")
    ax2.set_ylabel("Mean Reward", fontsize=11, fontweight="bold", color="#34495E")

    # X-axis formatting
    max_episodes = episodes[-1]
    num_ticks = min(8, max_episodes // 100 + 2)
    x_ticks_reward = np.linspace(0, max_episodes, num_ticks, dtype=int)
    ax2.set_xticks(x_ticks_reward)

    # Format x-tick labels with 'k' for large numbers
    x_labels_reward = []
    for tick in x_ticks_reward:
        if tick >= 1000:
            x_labels_reward.append(f'{tick/1000:.0f}k')
        else:
            x_labels_reward.append(str(tick))
    ax2.set_xticklabels(x_labels_reward)

    # Y-axis formatting with 3 decimal places
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    # Set reasonable y-axis limits
    y_min = min(0, min(smoothed_rewards) * 0.9)
    y_max = max(smoothed_rewards) * 1.05
    ax2.set_ylim([y_min, y_max])

    # Grid and styling
    ax2.grid(True, linestyle="--", alpha=0.4, color="#CCCCCC", linewidth=0.8)
    ax2.set_facecolor("#FAFAFA")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#333333")
    ax2.spines["bottom"].set_color("#333333")
    ax2.spines["left"].set_linewidth(1.2)
    ax2.spines["bottom"].set_linewidth(1.2)
    ax2.tick_params(colors="#555555", labelsize=10, width=1.2, length=5)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "combined_convergence.png")
    plt.savefig(save_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close()

    logger.info(f"Combined convergence plot saved to {save_path}")
    return save_path


def main():
    """Main function to generate convergence plots from training data."""
    print("\n" + "=" * 70)
    print("CONVERGENCE PLOT GENERATOR")
    print("=" * 70)
    print("\nGenerating convergence plots from training data...")
    print("\nPlot specifications:")
    print("  • Loss plot: X-axis 0 → 3.0k steps (0, 500, 1.0k, 1.5k, 2.0k, 2.5k, 3.0k)")
    print("  • Reward plot: X-axis Episode Number with 'k' notation for large values")
    print("  • Both plots: Professional styling with grid and proper axis labels")
    print("=" * 70 + "\n")

    # Create plots directory
    plots_dir = os.path.join("plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load training data
    logger.info("Loading training data...")
    step_losses, episode_rewards = load_training_data()

    # Generate plots
    plots_generated = []

    if step_losses is not None and len(step_losses) > 0:
        print("📊 Generating Loss Convergence Plot...")
        loss_path = plot_loss_convergence(step_losses, plots_dir)
        if loss_path:
            plots_generated.append(loss_path)
            print(f"   ✓ Loss plot saved")
    else:
        logger.error("No loss data found. Please run 'python run.py' first.")
        print("\n💡 Make sure training is complete and files exist in:")
        print("   files/training/logs/training_loss.xlsx")

    if episode_rewards is not None and len(episode_rewards) > 0:
        print("\n📈 Generating Reward Convergence Plot...")
        reward_path = plot_reward_convergence(episode_rewards, plots_dir)
        if reward_path:
            plots_generated.append(reward_path)
            print(f"   ✓ Reward plot saved")
    else:
        logger.error("No reward data found. Please run 'python run.py' first.")
        print("\n💡 Make sure training is complete and files exist in:")
        print("   files/training/logs/episodic_reward.csv")

    # Create combined plot if both datasets exist
    if step_losses is not None and episode_rewards is not None:
        if len(step_losses) > 0 and len(episode_rewards) > 0:
            print("\n🖼️ Generating Combined Convergence Plot...")
            combined_path = plot_combined_convergence(step_losses, episode_rewards, plots_dir)
            if combined_path:
                plots_generated.append(combined_path)
                print(f"   ✓ Combined plot saved")

    # Summary
    if plots_generated:
        print("\n" + "=" * 70)
        print("✅ CONVERGENCE PLOTS GENERATED SUCCESSFULLY")
        print("=" * 70)
        print("\nGenerated files:")
        for plot_path in plots_generated:
            print(f"  📁 {plot_path}")
        print("\n" + "=" * 70)
        print("\n🎯 Axis specifications:")
        print("   Loss Plot:")
        print("     • X-axis: 0 → 500 → 1.0k → 1.5k → 2.0k → 2.5k → 3.0k")
        print("     • Y-axis: Loss Value (scientific notation for small values)")
        print("   Reward Plot:")
        print("     • X-axis: Episode Number (0 to max episodes)")
        print("     • Y-axis: Mean Reward (3 decimal places)")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ NO PLOTS GENERATED")
        print("=" * 70)
        print("\nPossible issues:")
        print("  1. Training not completed yet")
        print("  2. Log files are empty or corrupted")
        print("  3. Wrong file paths")
        print("\nSolutions:")
        print("  • Run 'python run.py' to train the model first")
        print("  • Check if files exist in 'files/training/logs/'")
        print("=" * 70)

    print("\n")


if __name__ == "__main__":
    main()