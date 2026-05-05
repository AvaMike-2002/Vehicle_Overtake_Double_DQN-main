import os
import csv
import random
import logging

import numpy as np
import tensorflow as tf

from config import (
    GENERATIONS, NO_OF_ACTIONS, DISCOUNT_FACTOR,
    SAVE_PATH, SAVE_DIR, LOG_SAVE_PATH,
    BATCH_SIZE, IM_W, IM_H, STACK_SIZE,
    EPSILON_START, EPSILON_MIN,
    tf_writer,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeepQNetwork:
    """
    Double DQN agent with a CNN backbone.

    Parameters
    ----------
    training_mode : bool
        When True the agent explores (epsilon-greedy) and writes logs.
        When False epsilon is set to 0 for pure exploitation (trail_run).
    """

    def __init__(self, training_mode: bool) -> None:
        self.training_mode = training_mode
        self.epsilon = EPSILON_START
        self.min_epsilon = EPSILON_MIN
        self.decay = (EPSILON_START - EPSILON_MIN) / ((GENERATIONS // 2) - 1) if GENERATIONS > 2 else 0

        # CRITICAL FIX: Lower learning rate and add clipnorm
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-4,  # Lower LR for stability
            clipnorm=1.0  # Clip gradients by norm
        )

        self.train_network = self._build_network()
        self.predict_network = self._build_network()
        self.predict_network.set_weights(self.train_network.get_weights())
        self.step_counter = 0

        logger.info(f"DeepQNetwork initialized with training_mode={training_mode}")

    def _build_network(self) -> tf.keras.Model:
        """
        Optimized CNN architecture for highway driving with better spatial awareness.
        """
        inp = tf.keras.layers.Input(shape=(IM_H, IM_W, STACK_SIZE),
                                    name="state_input")

        # Normalize input
        x = tf.keras.layers.Lambda(lambda y: y / 255.0)(inp)

        # Conv1 - larger filters for better feature extraction
        x = tf.keras.layers.Conv2D(32, (8, 8), strides=4, activation="relu",
                                   padding="valid", name="conv1")(x)

        # Conv2
        x = tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation="relu",
                                   padding="valid", name="conv2")(x)

        # Conv3
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu",
                                   padding="valid", name="conv3")(x)

        # Flatten and Dense layers
        x = tf.keras.layers.Flatten(name="flatten")(x)
        x = tf.keras.layers.Dense(512, activation="relu", name="fc1")(x)
        x = tf.keras.layers.Dropout(0.2)(x)  # Add dropout for better generalization
        x = tf.keras.layers.Dense(NO_OF_ACTIONS, activation="linear",
                                  name="q_values")(x)

        model = tf.keras.Model(inputs=inp, outputs=x)
        return model

    def _preprocess(self, states: np.ndarray) -> np.ndarray:
        """Reshape and normalise a batch of stacked-frame observations."""
        # Handle different input shapes
        if len(states.shape) == 3:  # Single state
            states = np.expand_dims(states, axis=0)
        elif len(states.shape) == 4:  # Batch of states
            pass
        else:
            raise ValueError(f"Unexpected states shape: {states.shape}")

        return (states.reshape(-1, IM_H, IM_W, STACK_SIZE)
                .astype(np.float32) / 255.0)

    def get_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        Explores with probability epsilon; otherwise takes the greedy action
        from the predict_network.
        """
        if np.random.random() > self.epsilon:
            processed = self._preprocess(state)
            q_values = self.predict_network(processed, training=False)
            return int(np.argmax(q_values[0]))
        return np.random.randint(0, NO_OF_ACTIONS)

    def get_prediction(self, states: np.ndarray) -> np.ndarray:
        """Forward pass through the *predict* network (used for next-Q values)."""
        return self.predict_network(self._preprocess(states),
                                    training=False).numpy()

    def predict(self, states: np.ndarray) -> tf.Tensor:
        """Forward pass through the *train* network (used for current-Q values)."""
        return self.train_network(self._preprocess(states), training=True)

    # ──────────────────────────────────────────────────────────────────────
    # Q-value target computation
    # ──────────────────────────────────────────────────────────────────────

    def update_q_value(
            self,
            rewards: np.ndarray,
            current_qs: np.ndarray,
            next_qs: np.ndarray,
            actions: np.ndarray,
            done: np.ndarray,
    ) -> np.ndarray:
        """
        Bellman update (Double DQN target):

            target = r  +  γ · max_a Q_pred(s', a)  ·  (1 − done)

        Only the Q-value for the taken action is updated; all others are
        left unchanged so the loss only propagates through the chosen action.
        """
        targets = current_qs.copy()
        next_max_qs = np.max(next_qs, axis=1)  # greedy next Q
        new_qs = rewards + (1.0 - done) * DISCOUNT_FACTOR * next_max_qs

        for i in range(len(targets)):
            targets[i, actions[i]] = new_qs[i]
        return targets

    # ──────────────────────────────────────────────────────────────────────
    # Training step
    # ──────────────────────────────────────────────────────────────────────

    @tf.function
    def _train_step(
            self,
            states: tf.Tensor,
            targets: tf.Tensor,
    ) -> tf.Tensor:
        """
        Single gradient-descent step using Huber loss for stability.
        """
        with tf.GradientTape() as tape:
            predictions = self.train_network(states, training=True)
            # Use Huber loss (more robust to outliers)
            loss = tf.reduce_mean(
                tf.keras.losses.Huber(delta=1.0)(targets, predictions)
            )

        gradients = tape.gradient(loss, self.train_network.trainable_variables)
        # Clip gradients to prevent explosion
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        self.optimizer.apply_gradients(
            zip(gradients, self.train_network.trainable_variables)
        )
        return loss

    def train(self, batch: list) -> None:
        """Run one training step on a sampled mini-batch."""
        self.step_counter += 1
        current_states, actions, next_states, rewards, done = batch

        # Clip rewards to [-1, 1] for stability
        rewards = np.clip(rewards, -1.0, 1.0)

        # Compute Bellman targets
        current_qs = self.predict(current_states).numpy()
        next_qs = self.get_prediction(next_states)
        targets = self.update_q_value(rewards, current_qs,
                                      next_qs, actions, done)

        states_tf = tf.constant(self._preprocess(current_states))
        targets_tf = tf.constant(targets, dtype=tf.float32)
        loss = self._train_step(states_tf, targets_tf)

        mean_loss = float(np.mean(loss.numpy()))

        try:
            with tf_writer.as_default():
                tf.summary.scalar("loss", data=mean_loss, step=self.step_counter)
        except:
            pass

        self._save_log(self.step_counter, mean_loss, "loss.csv")

    def update_prediction_network(self) -> None:
        """
        Copy train_network weights → predict_network and save a checkpoint.

        Called every UPDATE_FREQUENCY episodes. This is the core 'levelling
        up' step of Double DQN — the predict network only improves in
        discrete jumps, keeping target Q-values stable between syncs.
        """
        self.predict_network.set_weights(self.train_network.get_weights())

        # Handle different checkpoint formats
        ckpt_path = SAVE_PATH.format(self.step_counter)

        try:
            # Try the new .weights.h5 format first
            if not ckpt_path.endswith('.weights.h5'):
                ckpt_path = ckpt_path.replace('.ckpt', '.weights.h5')
            self.train_network.save_weights(ckpt_path)
            logger.info(f"Synced networks — checkpoint saved: {ckpt_path}")
        except Exception as e:
            # Fall back to legacy checkpoint format
            try:
                checkpoint = tf.train.Checkpoint(model=self.train_network)
                checkpoint.save(ckpt_path.replace('.weights.h5', ''))
                logger.info(f"Synced networks — checkpoint saved (legacy): {ckpt_path}")
            except Exception as e2:
                logger.error(f"Failed to save checkpoint: {e2}")
                raise

    def load_model(self, model_path: str = None) -> bool:
        """
        Restore the latest checkpoint into predict_network.

        Parameters
        ----------
        model_path : str, optional
            Specific model path to load. If None, loads latest checkpoint.

        Returns
        -------
        bool
            True if model loaded successfully, False otherwise.
        """
        try:
            if model_path and os.path.exists(model_path):
                # Load from specific path
                if model_path.endswith('.h5') or model_path.endswith('.weights.h5'):
                    self.predict_network.load_weights(model_path)
                else:
                    # Try legacy checkpoint
                    checkpoint = tf.train.Checkpoint(model=self.predict_network)
                    checkpoint.restore(model_path).expect_partial()
                logger.info(f"Loaded model from: {model_path}")
                return True
            else:
                # Load latest checkpoint from directory
                # Try .weights.h5 files first
                import glob
                weight_files = glob.glob(os.path.join(SAVE_DIR, "*.weights.h5"))

                if weight_files:
                    # Get the latest file by modification time
                    latest = max(weight_files, key=os.path.getmtime)
                    self.predict_network.load_weights(latest)
                    logger.info(f"Loaded weights from: {latest}")
                    return True
                else:
                    # Fall back to legacy checkpoint format
                    latest = tf.train.latest_checkpoint(SAVE_DIR)
                    if latest is None:
                        raise FileNotFoundError(
                            f"No checkpoint found in '{SAVE_DIR}'. "
                            "Run training first (python run.py)."
                        )
                    # For legacy checkpoints, we need to rebuild and load
                    checkpoint = tf.train.Checkpoint(model=self.predict_network)
                    checkpoint.restore(latest).expect_partial()
                    logger.info(f"Loaded weights from (legacy): {latest}")
                    return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────
    # CSV logging
    # ──────────────────────────────────────────────────────────────────────

    def _save_log(self, step: int, value: float, filename: str) -> None:
        """Append a (step, value) row to a CSV log file."""
        try:
            filepath = os.path.join(LOG_SAVE_PATH, filename)
            os.makedirs(LOG_SAVE_PATH, exist_ok=True)

            # Check if file exists to determine if we need to write header
            file_exists = os.path.isfile(filepath)

            with open(filepath, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["step", "value"])
                writer.writerow([step, value])
        except Exception as e:
            logger.warning(f"Failed to save log: {e}")