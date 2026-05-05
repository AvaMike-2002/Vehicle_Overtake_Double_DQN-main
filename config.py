import os
import tensorflow as tf

# ---------------------------------------------------------------------------
# Replay buffer & training schedule
# ---------------------------------------------------------------------------
BATCH_SIZE        = 32          # Back to 32 for stability
N_VEHICLES        = 20
MEMORY_SIZE       = 100_000     # Much larger buffer for diverse experiences
GENERATIONS       = 500
NO_OF_ACTIONS     = 5
DISCOUNT_FACTOR   = 0.99
TRAIN_FREQUENCY   = 4
UPDATE_FREQUENCY  = 100         # Less frequent updates for stability (every 100 episodes)

# ---------------------------------------------------------------------------
# Exploration schedule - CRITICAL FIX
# ---------------------------------------------------------------------------
EPSILON_START     = 1.0
EPSILON_MIN       = 0.1         # Higher min exploration (0.1 instead of 0.05)
EPSILON_DECAY     = 0.997       # Slower decay for more exploration

# ---------------------------------------------------------------------------
# Observation image dimensions
# ---------------------------------------------------------------------------
IM_W, IM_H        = 84, 84
STACK_SIZE        = 4

# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------
EVAL_EPISODES     = 10          # Number of episodes for evaluation
EVAL_VIDEO_FPS    = 20          # FPS for evaluation video
EVAL_VIDEO_SCALE  = 4           # Scale factor for video output

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
SAVE_PATH         = os.path.join("files", "training", "model_files", "cp-{}.weights.h5")
SAVE_DIR          = os.path.dirname(SAVE_PATH)
LOG_SAVE_PATH     = os.path.join("files", "training", "my_logs")
TF_BOARD_PATH     = os.path.join("files", "training", "my_logs", "tf_board")
XLSX_LOG_PATH     = os.path.join("files", "training", "training_log.xlsx")
GRAPHS_DIR        = os.path.join("files", "training", "graphs")

LOGS_DIR          = os.path.join("files", "training", "logs")
LOSS_EXCEL_PATH   = os.path.join(LOGS_DIR, "training_loss.xlsx")
REWARD_CSV_PATH   = os.path.join(LOGS_DIR, "episodic_reward.csv")
FINAL_MODEL_PATH  = os.path.join(LOGS_DIR, "final_model.weights.h5")

for dir_path in [LOG_SAVE_PATH, SAVE_DIR, GRAPHS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ---------------------------------------------------------------------------
# TensorBoard setup
# ---------------------------------------------------------------------------
try:
    tf_writer = tf.summary.create_file_writer(TF_BOARD_PATH)
    TENSORBOARD_AVAILABLE = True
except Exception as e:
    print(f"Warning: TensorBoard not available - {e}")
    class DummyWriter:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def as_default(self): return self
        def set_as_default(self): pass
    tf_writer = DummyWriter()
    TENSORBOARD_AVAILABLE = False

# ---------------------------------------------------------------------------
# Highway-env configuration with improved reward shaping
# ---------------------------------------------------------------------------
ENV_CONFIG = {
    "offscreen_rendering": True,
    "observation": {
        "type": "GrayscaleObservation",
        "weights": [0.2989, 0.5870, 0.1140],
        "stack_size": STACK_SIZE,
        "observation_shape": (IM_W, IM_H),
    },
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": True,
        "lateral": True,
    },
    "duration": 100,
    "vehicles_count": N_VEHICLES - 1,
    "lanes_count": 4,
    "policy_frequency": 5,
    "simulation_frequency": 15,
    "vehicles_density": 0.6,
    "speed_limit": 30,
    "collision_reward": -2.0,           # Increased penalty for collisions
    "high_speed_reward": 0.3,           # Reduced to encourage safe speed
    "right_lane_reward": 0.05,          # Small positive for staying right
    "lane_change_reward": 0.1,          # Reward for successful lane changes
    "normalize_reward": True,
    "offroad_terminal": False,
    "screen_width": IM_W,
    "screen_height": IM_H,
    "scaling": 5.75,
    # Additional parameters for better behavior
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "initial_lane_id": None,
    "ego_spacing": 2,
}

# Add action mapping for clarity
ACTION_MAP = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER"
}