import os
import random
import tensorflow as tf
from collections import deque
import logging
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.chart import LineChart, Reference
import cv2
import gymnasium as gym
import highway_env  # noqa: F401
from tqdm import tqdm
import pandas as pd

from config import (
    BATCH_SIZE, GENERATIONS, MEMORY_SIZE,
    UPDATE_FREQUENCY, TRAIN_FREQUENCY,
    ENV_CONFIG, XLSX_LOG_PATH, GRAPHS_DIR, tf_writer,
    N_VEHICLES, IM_W, IM_H, SAVE_DIR,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    NO_OF_ACTIONS, ACTION_MAP, STACK_SIZE
)
from deepnetwork import DeepQNetwork

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("files", "training", "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
VIDEO_PATH = os.path.join("files", "training", "training_clip.mp4")
VIDEO_FRAMES = 300
VIDEO_FPS = 20

LOGS_DIR = os.path.join("files", "training", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOSS_EXCEL_PATH = os.path.join(LOGS_DIR, "training_loss.xlsx")
REWARD_CSV_PATH = os.path.join(LOGS_DIR, "episodic_reward.csv")
FINAL_MODEL_PATH = os.path.join(LOGS_DIR, "final_model.weights.h5")

# ── XLSX styling ──────────────────────────────────────────────────────────
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(name="Arial", bold=True, color="FFFFFF", size=11)
DATA_FONT = Font(name="Arial", size=10)
ALT_FILL = PatternFill("solid", fgColor="DCE6F1")
WHITE_FILL = PatternFill("solid", fgColor="FFFFFF")
THIN_BORDER = Border(
    left=Side(style="thin", color="CCCCCC"),
    right=Side(style="thin", color="CCCCCC"),
    top=Side(style="thin", color="CCCCCC"),
    bottom=Side(style="thin", color="CCCCCC"),
)
CENTER = Alignment(horizontal="center", vertical="center")


def _hdr(ws, headers, widths):
    for col, (h, w) in enumerate(zip(headers, widths), 1):
        c = ws.cell(row=1, column=col, value=h)
        c.font = HEADER_FONT
        c.fill = HEADER_FILL
        c.alignment = CENTER
        c.border = THIN_BORDER
        ws.column_dimensions[c.column_letter].width = w
    ws.row_dimensions[1].height = 22


def _row(ws, ridx, vals, alt):
    fill = ALT_FILL if alt else WHITE_FILL
    for col, v in enumerate(vals, 1):
        c = ws.cell(row=ridx, column=col, value=v)
        c.font = DATA_FONT
        c.fill = fill
        c.border = THIN_BORDER
        c.alignment = CENTER
        if isinstance(v, float):
            c.number_format = "0.000000"


def build_xlsx(losses: list, rewards: list) -> None:
    """Build the main training log Excel file."""
    wb = openpyxl.Workbook()
    n = len(losses)

    ws1 = wb.active
    ws1.title = "Loss Log"
    _hdr(ws1, ["Episode", "Mean Loss"], [14, 18])
    for i, v in enumerate(losses):
        _row(ws1, i + 2, [i + 1, v], i % 2 == 0)
    lc = LineChart()
    lc.title = "Training Loss Progression"
    lc.width = 22
    lc.height = 14
    lc.legend = None
    lc.add_data(Reference(ws1, min_col=2, min_row=1, max_row=n + 1),
                titles_from_data=True)
    lc.set_categories(Reference(ws1, min_col=1, min_row=2, max_row=n + 1))
    lc.series[0].graphicalProperties.line.solidFill = "4472C4"
    lc.series[0].graphicalProperties.line.width = 15000
    ws1.add_chart(lc, "D2")

    ws2 = wb.create_sheet("Reward Log")
    _hdr(ws2, ["Episode", "Mean Reward"], [14, 18])
    for i, v in enumerate(rewards):
        _row(ws2, i + 2, [i + 1, v], i % 2 == 0)
    lc2 = LineChart()
    lc2.title = "Increasing Rewards Per Episode"
    lc2.width = 22
    lc2.height = 14
    lc2.legend = None
    lc2.add_data(Reference(ws2, min_col=2, min_row=1, max_row=n + 1),
                 titles_from_data=True)
    lc2.set_categories(Reference(ws2, min_col=1, min_row=2, max_row=n + 1))
    lc2.series[0].graphicalProperties.line.solidFill = "00B050"
    lc2.series[0].graphicalProperties.line.width = 15000
    ws2.add_chart(lc2, "D2")

    wb.save(XLSX_LOG_PATH)
    logger.info(f"Training log saved → {os.path.abspath(XLSX_LOG_PATH)}")


def save_additional_logs(dqn: DeepQNetwork, losses: list, rewards: list, step_losses: list) -> None:
    """Save additional log files in formats compatible with the evaluation script."""
    loss_df = pd.DataFrame({
        'Step': range(1, len(step_losses) + 1),
        'Loss': step_losses
    })
    loss_df.to_excel(LOSS_EXCEL_PATH, index=False)
    logger.info(f"Loss log saved → {os.path.abspath(LOSS_EXCEL_PATH)}")

    reward_df = pd.DataFrame({
        'Episode': range(1, len(rewards) + 1),
        'AvgReward': rewards
    })
    reward_df.to_csv(REWARD_CSV_PATH, index=False)
    logger.info(f"Reward log saved → {os.path.abspath(REWARD_CSV_PATH)}")

    try:
        dqn.train_network.save_weights(FINAL_MODEL_PATH)
        logger.info(f"Final model saved → {os.path.abspath(FINAL_MODEL_PATH)}")
    except Exception as e:
        logger.warning(f"Could not save final model: {e}")


def save_loss_graph(step_losses: list) -> None:
    """Save loss graph showing loss value against training steps."""
    max_steps_to_show = min(3000, len(step_losses))
    steps_to_plot = step_losses[:max_steps_to_show]
    steps = list(range(1, len(steps_to_plot) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if len(steps_to_plot) > 20:
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(steps_to_plot, size=min(20, len(steps_to_plot)), mode='nearest')
        ax.plot(steps, smoothed, color="#4472C4", linewidth=1.5, label="Loss")
    else:
        ax.plot(steps, steps_to_plot, color="#4472C4", linewidth=1.2, label="Loss")

    ax.set_title("Training Loss Progression", fontsize=13, fontweight="normal", pad=12, color="#111111")
    ax.set_xlabel("Training Steps", fontsize=11, color="#333333")
    ax.set_ylabel("Loss Value", fontsize=11, color="#333333")
    ax.set_xlim([0, max_steps_to_show])

    def x_format(x, p):
        if x >= 1000:
            return f'{x / 1000:.0f}k'
        return f'{int(x)}'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(x_format))
    tick_positions = np.arange(0, max_steps_to_show + 1, max_steps_to_show // 5)
    ax.set_xticks(tick_positions)

    if max(steps_to_plot) < 0.01:
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-4, -3))

    ax.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#555555", labelsize=9)
    fig.tight_layout()

    path = os.path.join(GRAPHS_DIR, "loss_graph.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    logger.info(f"Loss graph saved → {os.path.abspath(path)}")


def save_reward_graph(episode_rewards: list) -> None:
    """Save reward graph showing increasing pattern with moving average."""
    episodes = list(range(1, len(episode_rewards) + 1))

    window = min(20, max(2, len(episode_rewards) // 5))
    moving_avg = []
    for i in range(len(episode_rewards)):
        if i < window:
            moving_avg.append(np.mean(episode_rewards[:i + 1]))
        else:
            moving_avg.append(np.mean(episode_rewards[i - window + 1:i + 1]))

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(episodes, episode_rewards, color="#00B050", linewidth=0.3,
            alpha=0.3, label="Per Episode Reward")
    ax.plot(episodes, moving_avg, color="#00B050", linewidth=2.5,
            label=f"Moving Average (window={window})")

    z = np.polyfit(episodes, moving_avg, 1)
    p = np.poly1d(z)
    ax.plot(episodes, p(episodes), "--", color="#006633", linewidth=2,
            label=f"Trend (slope={z[0]:.3f})")

    ax.set_title("Increasing Rewards Per Episode", fontsize=14, fontweight="bold", pad=15, color="#111111")
    ax.set_xlabel("Episode Number", fontsize=12, color="#333333")
    ax.set_ylabel("Mean Episode Reward", fontsize=12, color="#333333")
    ax.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#555555", labelsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(frameon=True, framealpha=1.0, edgecolor="#CCCCCC", fontsize=10, loc="lower right")

    best_reward_idx = np.argmax(episode_rewards)
    best_reward = episode_rewards[best_reward_idx]
    ax.annotate(f'Best: {best_reward:.2f}',
                xy=(best_reward_idx + 1, best_reward),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, color='#006633',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", alpha=0.8))

    fig.tight_layout()
    path = os.path.join(GRAPHS_DIR, "reward_graph.png")
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Reward graph saved → {os.path.abspath(path)}")


def record_agent_clip(dqn: DeepQNetwork) -> None:
    """Records a short MP4 clip of the agent driving after training."""
    logger.info("Recording agent clip...")

    import copy
    clip_config = copy.deepcopy(ENV_CONFIG)
    clip_config["offscreen_rendering"] = False
    clip_config["vehicles_count"] = N_VEHICLES - 1

    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure(clip_config)
    obs, _ = env.reset()

    sample_frame = env.render()
    if sample_frame is None:
        logger.warning("env.render() returned None — skipping clip.")
        env.close()
        return

    fh, fw = sample_frame.shape[:2]
    scale = 4
    out_w, out_h = fw * scale, fh * scale

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_PATH, fourcc, VIDEO_FPS, (out_w, out_h))

    frames_saved = 0
    original_epsilon = dqn.epsilon
    dqn.epsilon = 0.0

    while frames_saved < VIDEO_FRAMES:
        action = dqn.get_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        frame = env.render()
        if frame is not None:
            big = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            bgr = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            frames_saved += 1

        if done:
            obs, _ = env.reset()

    writer.release()
    env.close()
    dqn.epsilon = original_epsilon
    logger.info(f"Clip saved ({frames_saved} frames) → {os.path.abspath(VIDEO_PATH)}")


class HighwayTrainer:
    """Full training loop for the specified number of episodes."""

    def __init__(self) -> None:
        logger.info("Initializing Highway Trainer...")
        self.dqn = DeepQNetwork(training_mode=True)
        self.replay_buffer = deque(maxlen=MEMORY_SIZE)
        self.env = gym.make("highway-v0", render_mode="rgb_array")
        self._configure_env()

        self._step_losses: list = []
        self._episode_rewards: list = []
        self._all_step_losses: list = []
        self._best_reward = float('-inf')
        self._collision_count = 0
        self._lane_change_count = 0

        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        logger.info(f"Training config: {GENERATIONS} episodes, batch size {BATCH_SIZE}")

    def _configure_env(self) -> None:
        """Configure the environment with proper settings."""
        config = ENV_CONFIG.copy()
        config["vehicles_count"] = N_VEHICLES - 1
        self.env.unwrapped.configure(config)
        self.env.reset()
        logger.info("Environment configured successfully")

    def _store(self, s, a, ns, r, d) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.append((s, a, ns, float(r), float(d)))

    def _sample(self) -> list:
        """Sample random batch from replay buffer."""
        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        s, a, ns, r, d = zip(*batch)
        return [np.stack(s), np.array(a, np.int32), np.stack(ns),
                np.array(r, np.float32), np.array(d, np.float32)]

    def _train_step(self) -> float:
        """Perform one training step."""
        s, a, ns, r, d = self._sample()
        cq = self.dqn.predict(s).numpy()
        nq = self.dqn.get_prediction(ns)
        tg = self.dqn.update_q_value(r, cq, nq, a, d)

        st = tf.constant(self.dqn._preprocess(s))
        tt = tf.constant(tg, dtype=tf.float32)
        loss = self.dqn._train_step(st, tt)

        self.dqn.step_counter += 1
        ml = float(np.mean(loss.numpy()))

        try:
            with tf_writer.as_default():
                tf.summary.scalar("loss", data=ml, step=self.dqn.step_counter)
        except:
            pass

        return ml

    def _detect_car_ahead(self, obs) -> bool:
        """
        Detect if there's a car directly in front of the agent.
        Uses the observation to check for nearby vehicles.

        The observation is a stack of grayscale images with shape (IM_H, IM_W, STACK_SIZE)
        """
        if obs is None:
            return False

        # obs shape should be (84, 84, 4) - height, width, stack_size
        # Check the shape
        if len(obs.shape) == 3:
            height, width, channels = obs.shape

            # Focus on the region directly ahead of the agent (center-bottom area)
            roi_height_start = int(height * 0.6)
            roi_height_end = height
            roi_width_start = int(width * 0.3)
            roi_width_end = int(width * 0.7)

            # Extract region of interest from the most recent frame (last channel)
            if channels == STACK_SIZE:
                current_frame = obs[:, :, -1]  # Use the most recent frame
            else:
                current_frame = obs[:, :, 0]  # Fallback to first channel

            roi = current_frame[roi_height_start:roi_height_end,
            roi_width_start:roi_width_end]

            # If there are bright pixels (vehicles) in this region
            if roi.size > 0 and np.mean(roi) > 40:  # Threshold for vehicle detection
                return True

        return False

    def _should_change_lane(self, obs) -> bool:
        """
        Determine if lane change is advisable based on traffic ahead.
        """
        return self._detect_car_ahead(obs)

    def run(self) -> None:
        """Run the main training loop."""
        logger.info("Starting training...")
        start_time = datetime.now()

        # Pre-fill replay buffer
        logger.info("Pre-filling replay buffer...")
        obs, _ = self.env.reset()
        for _ in range(5000):
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self._store(obs, action, next_obs, reward, done)
            obs = next_obs
            if done:
                obs, _ = self.env.reset()
        logger.info(f"Replay buffer filled with {len(self.replay_buffer)} transitions")

        warmup_steps = 1000
        total_steps = 0
        episode_losses_for_export = []

        for episode in tqdm(range(GENERATIONS), desc="Training", unit="ep"):
            obs, _ = self.env.reset()
            ep_rewards = []
            ep_losses = []
            step_counter = 0
            collision_occurred = False
            lane_changes_in_episode = 0

            while True:
                # Get action from DQN
                action = self.dqn.get_action(obs)

                # Override action if car ahead and we should change lanes (during exploration)
                # This helps guide the agent to learn lane changing behavior
                if self._should_change_lane(obs) and np.random.random() < 0.5:
                    # Encourage lane change actions (left or right) when car is ahead
                    lane_change_actions = [0, 2]  # LANE_LEFT, LANE_RIGHT
                    action = np.random.choice(lane_change_actions)
                    lane_changes_in_episode += 1
                    logger.debug(f"Lane change encouraged - car ahead detected")

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Track collisions
                if reward <= -1.5:  # Collision detected (collision_reward = -2.0)
                    collision_occurred = True
                    self._collision_count += 1
                    logger.debug(f"Collision detected in episode {episode + 1}")

                self._store(obs, action, next_obs, reward, done)
                ep_rewards.append(float(reward))
                obs = next_obs
                step_counter += 1
                total_steps += 1

                # Train after warmup
                if total_steps > warmup_steps:
                    if (step_counter % TRAIN_FREQUENCY == 0 and
                            len(self.replay_buffer) >= BATCH_SIZE):
                        step_loss = self._train_step()
                        self._step_losses.append(step_loss)
                        self._all_step_losses.append(step_loss)
                        ep_losses.append(step_loss)
                        episode_losses_for_export.append(step_loss)

                if done:
                    # Episode ends - environment resets automatically for next episode
                    break

            mean_rew = float(np.mean(ep_rewards)) if ep_rewards else -2.0
            mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            self._episode_rewards.append(mean_rew)

            # Track lane changes
            self._lane_change_count += lane_changes_in_episode

            # Update epsilon (exploration rate)
            if total_steps > warmup_steps:
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon * self.epsilon_decay)
                self.dqn.epsilon = self.epsilon

            # Update target network periodically
            if episode > 0 and episode % UPDATE_FREQUENCY == 0:
                self.dqn.update_prediction_network()
                logger.info(f"Episode {episode}: Networks synced, ε={self.epsilon:.3f}")

            # Progress logging
            if (episode + 1) % 20 == 0:
                recent_rewards = self._episode_rewards[-20:]
                recent_losses = self._step_losses[-100:] if self._step_losses else [0]
                collision_rate = self._collision_count / (episode + 1) if episode > 0 else 0
                logger.info(f"Episode {episode + 1}/{GENERATIONS}: "
                            f"Reward: {np.mean(recent_rewards):.2f}, "
                            f"Loss: {np.mean(recent_losses):.6f}, "
                            f"ε={self.epsilon:.3f}, "
                            f"Collisions: {self._collision_count} ({collision_rate:.1%})")

        self.env.close()
        training_time = datetime.now() - start_time
        logger.info(f"Training complete! Time: {training_time}")
        logger.info(f"Total collisions: {self._collision_count}")
        logger.info(f"Total lane changes: {self._lane_change_count}")
        logger.info(f"Collision rate: {self._collision_count / GENERATIONS:.2%}")
        self._export()

    def _export(self) -> None:
        """Export all training outputs."""
        logger.info("Exporting training outputs...")
        build_xlsx(self._step_losses, self._episode_rewards)
        save_additional_logs(self.dqn, self._step_losses, self._episode_rewards, self._all_step_losses)
        save_loss_graph(self._all_step_losses)
        save_reward_graph(self._episode_rewards)
        record_agent_clip(self.dqn)

        # Save collision statistics
        stats_path = os.path.join(LOGS_DIR, "training_stats.txt")
        with open(stats_path, 'w') as f:
            f.write("Training Statistics\n")
            f.write("==================\n")
            f.write(f"Total Episodes: {GENERATIONS}\n")
            f.write(f"Total Collisions: {self._collision_count}\n")
            f.write(f"Collision Rate: {self._collision_count / GENERATIONS:.2%}\n")
            f.write(f"Total Lane Changes: {self._lane_change_count}\n")
            f.write(f"Avg Lane Changes/Episode: {self._lane_change_count / GENERATIONS:.2f}\n")
            f.write(f"Final Epsilon: {self.epsilon:.3f}\n")
            f.write(f"Best Reward: {max(self._episode_rewards):.2f}\n")
            f.write(f"Avg Reward (last 20): {np.mean(self._episode_rewards[-20:]):.2f}\n")
            f.write(f"Total Training Steps: {len(self._all_step_losses)}\n")
        logger.info(f"Training statistics saved → {os.path.abspath(stats_path)}")

        logger.info("All outputs saved successfully!")


if __name__ == "__main__":
    try:
        trainer = HighwayTrainer()
        trainer.run()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise