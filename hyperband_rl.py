import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import random
from collections import deque
import os
import pandas as pd
from multiprocessing import Pool, cpu_count

# Define the Actor network
def create_actor(state_dim, action_dim, max_action):
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(action_dim, activation="tanh")(x)
    outputs = layers.Lambda(lambda i: i * max_action)(outputs)
    return models.Model(inputs, outputs)

# Define the Critic network
def create_critic(state_dim, action_dim):
    state_input = layers.Input(shape=(state_dim,))
    action_input = layers.Input(shape=(action_dim,))
    concat = layers.Concatenate()([state_input, action_input])

    x = layers.Dense(256, activation="relu")(concat)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    return models.Model([state_input, action_input], outputs)

# Define the SAC Agent
class SAC:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4, memory_size=1000000, batch_size=64):
        self.actor = create_actor(state_dim, action_dim, max_action)
        self.actor_optimizer = optimizers.Adam(learning_rate=lr)

        self.critic_1 = create_critic(state_dim, action_dim)
        self.critic_2 = create_critic(state_dim, action_dim)
        self.critic_1_optimizer = optimizers.Adam(learning_rate=lr)
        self.critic_2_optimizer = optimizers.Adam(learning_rate=lr)

        self.target_critic_1 = create_critic(state_dim, action_dim)
        self.target_critic_2 = create_critic(state_dim, action_dim)
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

    def select_action(self, state):
        state = np.expand_dims(state, axis=0)
        return self.actor(state)[0].numpy()

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape(persistent=True) as tape:
            next_actions = self.actor(next_states)
            next_q1 = self.target_critic_1([next_states, next_actions])
            next_q2 = self.target_critic_2([next_states, next_actions])
            next_q = rewards + (1 - dones) * self.gamma * (tf.minimum(next_q1, next_q2) - self.alpha * next_actions)

            q1 = self.critic_1([states, actions])
            q2 = self.critic_2([states, actions])
            critic_1_loss = tf.reduce_mean(tf.square(q1 - next_q))
            critic_2_loss = tf.reduce_mean(tf.square(q2 - next_q))

        critic_1_grads = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grads = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))

        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            actor_loss = tf.reduce_mean(self.alpha * new_actions - tf.minimum(self.critic_1([states, new_actions]), self.critic_2([states, new_actions])))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        for target_param, param in zip(self.target_critic_1.trainable_variables, self.critic_1.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)

        for target_param, param in zip(self.target_critic_2.trainable_variables, self.critic_2.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array(actions), dtype=tf.float32)
        rewards = tf.convert_to_tensor(np.array(rewards).reshape(-1, 1), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.array(next_states), dtype=tf.float32)
        dones = tf.convert_to_tensor(np.array(dones).reshape(-1, 1), dtype=tf.float32)

        self.train_step(states, actions, rewards, next_states, dones)

    def save_models(self, path=""):
        self.actor.save(os.path.join(path, "best_actor_model.keras"))  # Use .keras or .h5 extension
        self.critic_1.save(os.path.join(path, "best_critic_1_model.keras"))
        self.critic_2.save(os.path.join(path, "best_critic_2_model.keras"))
        self.target_critic_1.save(os.path.join(path, "best_target_critic_1_model.keras"))
        self.target_critic_2.save(os.path.join(path, "best_target_critic_2_model.keras"))

def train_sac(env, agent, num_episodes, render_every=10):
    all_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            if episode % render_every == 0:
                env.render()

            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.update_memory(state, action, reward, next_state, done or truncated)
            agent.train()
            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        all_rewards.append(episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward}")
    
    env.close()
    return all_rewards

def run_trial(env_name, sampled_params, num_episodes, render_every):
    print(f"Running Trial with hyperparameters: {sampled_params}")

    # Ensure that each process creates its own TensorFlow session
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SAC(
        state_dim, action_dim, max_action,
        gamma=sampled_params['gamma'],
        tau=sampled_params['tau'],
        alpha=sampled_params['alpha'],
        lr=sampled_params['lr'],
        memory_size=sampled_params['memory_size'],
        batch_size=sampled_params['batch_size']
    )

    rewards = train_sac(env, agent, num_episodes, render_every)
    avg_reward = np.mean(rewards[-100:])  # Average reward over last 100 episodes
    env.close()

    return {
        **sampled_params,
        "Average Reward": avg_reward,
        "agent": agent if avg_reward > -np.inf else None  # Return agent only if it has some performance
    }

def get_hyperparameter_space():
    return {
        'gamma': [0.98, 0.99, 0.995],
        'tau': [0.005, 0.01, 0.02],
        'alpha': [0.2, 0.3, 0.4],
        'lr': [1e-4, 3e-4, 1e-3],
        'memory_size': [1000000],
        'batch_size': [64, 128]
    }

def hyperband(env_name, max_iter, eta, num_trials, render_every):
    hyperparameter_space = get_hyperparameter_space()
    results = []

    s_max = int(np.log(max_iter) / np.log(eta))
    B = (s_max + 1) * max_iter

    for s in reversed(range(s_max + 1)):
        n = int(np.ceil(B / max_iter / (s + 1) * eta**s))
        r = max_iter * eta**(-s)

        # Sample random configurations
        configurations = [
            {key: random.choice(values) for key, values in hyperparameter_space.items()}
            for _ in range(n)
        ]

        for i in range(s + 1):
            ni = int(n * eta**(-i))
            ri = int(r * eta**(i))
            print(f"\n--- Running {ni} configurations with {ri} episodes each ---")

            with Pool(processes=cpu_count()) as pool:
                trial_args = [(env_name, config, ri, render_every) for config in configurations]
                trial_results = pool.starmap(run_trial, trial_args)

            # Sort configurations by reward and select top configurations
            trial_results.sort(key=lambda x: x["Average Reward"], reverse=True)
            configurations = [result for result in trial_results[:int(ni / eta)]]
            results.extend(trial_results)

    best_result = max(results, key=lambda x: x["Average Reward"])
    
    if best_result["agent"] is not None:
        model_save_path = "best_models"
        os.makedirs(model_save_path, exist_ok=True)
        best_result["agent"].save_models(model_save_path)

        with open(os.path.join(model_save_path, 'best_hyperparameters.txt'), 'w') as f:
            for key, value in best_result.items():
                if key != "agent":
                    f.write(f"{key}: {value}\n")

    # Save all results
    results_df = pd.DataFrame(results).drop(columns=["agent"])
    results_df.to_csv("hyperband_results.csv", index=False)

    print("Best average reward:", best_result["Average Reward"])
    print("Best hyperparameters:", best_result)
    return best_result

if __name__ == "__main__":
    env_name = "LunarLanderContinuous-v2"
    max_iter = 200  # Maximum iterations (episodes)
    eta = 3  # Controls the proportion of configurations to discard
    num_trials = 60  # Number of trials (parallel runs)
    render_every = max_iter  # Render every max_iter

    best_params = hyperband(
        env_name=env_name,
        max_iter=max_iter,
        eta=eta,
        num_trials=num_trials,
        render_every=render_every
    )

    print("Hyperband Search Completed.")
