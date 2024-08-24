import random
import numpy as np
import tensorflow as tf
import gymnasium as gym
from collections import deque
from multiprocessing import Pool, cpu_count
import os
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import traceback

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define the Actor network
def create_actor(state_dim, action_dim, max_action):
    inputs = tf.keras.layers.Input(shape=(state_dim,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(action_dim, activation="tanh")(x)
    outputs = tf.keras.layers.Lambda(lambda i: i * max_action)(outputs)
    return tf.keras.models.Model(inputs, outputs)

# Define the Critic network
def create_critic(state_dim, action_dim):
    state_input = tf.keras.layers.Input(shape=(state_dim,))
    action_input = tf.keras.layers.Input(shape=(action_dim,))
    concat = tf.keras.layers.Concatenate()([state_input, action_input])

    x = tf.keras.layers.Dense(256, activation="relu")(concat)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)

    return tf.keras.models.Model([state_input, action_input], outputs)

# Define the SAC Agent
class SAC:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, alpha_initial=0.2, target_entropy=-1.0, actor_lr=3e-4, critic_lr=3e-4, memory_size=1000000, batch_size=64, replay_init=1000):
        self.actor = create_actor(state_dim, action_dim, max_action)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

        self.critic_1 = create_critic(state_dim, action_dim)
        self.critic_2 = create_critic(state_dim, action_dim)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        self.target_critic_1 = create_critic(state_dim, action_dim)
        self.target_critic_2 = create_critic(state_dim, action_dim)
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = tf.Variable(alpha_initial, trainable=True)
        self.target_entropy = target_entropy
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.replay_init = replay_init

        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

    def select_action(self, state):
        # Ensure the input state has the correct shape (1, state_dim)
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        return self.actor(state)[0].numpy()  # Get the action and remove the batch dimension

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

        # Alpha loss and optimization
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            log_probs = tf.reduce_sum(-tf.square(actions), axis=1, keepdims=True)  # Approximation for log probabilities
            alpha_loss = -tf.reduce_mean(self.alpha * (log_probs + self.target_entropy))

        alpha_grads = tape.gradient(alpha_loss, [self.alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.alpha]))

        for target_param, param in zip(self.target_critic_1.trainable_variables, self.critic_1.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)

        for target_param, param in zip(self.target_critic_2.trainable_variables, self.critic_2.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)

    def train(self):
        if len(self.memory) < max(self.batch_size, self.replay_init):
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
        self.actor.save(os.path.join(path, "best_actor_model.keras"))
        self.critic_1.save(os.path.join(path, "best_critic_1_model.keras"))
        self.critic_2.save(os.path.join(path, "best_critic_2_model.keras"))
        self.target_critic_1.save(os.path.join(path, "best_target_critic_1_model.keras"))
        self.target_critic_2.save(os.path.join(path, "best_target_critic_2_model.keras"))

def train_sac(env, agent, num_episodes, render_every=10, save_renders=False, save_path=None):
    all_rewards = []
    all_entropies = []
    all_alphas = []
    all_losses = []

    last_5_renders = []  # To store the renders of the last 5 episodes
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        frames = []

        while not done:
            if episode % render_every == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.update_memory(state, action, reward, next_state, done or truncated)
            agent.train()
            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        all_rewards.append(episode_reward)
        all_entropies.append(-np.mean(agent.actor(np.expand_dims(state, axis=0)).numpy()))  # Ensure correct input shape
        all_alphas.append(agent.alpha.numpy())
        all_losses.append(agent.train())  # Save loss if needed

        # Save the last 5 episode renders
        if save_renders and episode >= num_episodes - 5:
            last_5_renders.append(frames)

        print(f"Episode {episode}, Reward: {episode_reward}")

    # Save the renders as GIFs
    if save_renders and save_path is not None:
        for i, frames in enumerate(last_5_renders):
            if frames:  # Check if frames list is not empty
                gif_path = os.path.join(save_path, f"episode_{num_episodes - 5 + i}.gif")
                imageio.mimsave(gif_path, frames, fps=30)

    env.close()

    # Save the entropy and alpha metrics as plots
    if save_path is not None:
        plt.figure()
        plt.plot(all_entropies, label='Entropy')
        plt.plot(all_alphas, label='Alpha')
        plt.xlabel('Episode')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(os.path.join(save_path, "metrics.png"))
        plt.close()

    return all_rewards, all_entropies, all_alphas, all_losses

def run_trial(env_name, sampled_params, num_episodes, render_every):
    try:
        print(f"Running Trial with hyperparameters: {sampled_params}")

        # Exclude complex objects like SAC from the directory name
        simple_params = {k: v for k, v in sampled_params.items() if not isinstance(v, SAC)}

        # Simplify directory name: truncate float values for readability and validity
        dir_name = "_".join([f"{key}={str(simple_params[key])[:6]}" for key in simple_params.keys()])
        dir_name = dir_name.replace("/", "").replace("\\", "")  # Ensure the directory name is valid

        # Create a directory for this trial's results
        save_path = os.path.join("trials", dir_name)
        os.makedirs(save_path, exist_ok=True)

        # Ensure that each process creates its own TensorFlow session
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        agent = SAC(
            state_dim, action_dim, max_action,
            gamma=sampled_params['gamma'],
            tau=sampled_params['tau'],
            alpha_initial=sampled_params['alpha_initial'],
            target_entropy=sampled_params['target_entropy'],
            actor_lr=sampled_params['actor_lr'],
            critic_lr=sampled_params['critic_lr'],
            memory_size=sampled_params['memory_size'],
            batch_size=sampled_params['batch_size'],
            replay_init=sampled_params['replay_init']
        )

        # Warm-up phase: fill the replay buffer before training
        state, _ = env.reset()
        for _ in range(sampled_params['replay_init']):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.update_memory(state, action, reward, next_state, done or truncated)
            if done or truncated:
                state, _ = env.reset()
            else:
                state = next_state

        # Debug: Check replay buffer size
        print(f"Replay buffer size after warm-up: {len(agent.memory)}")

        # Now proceed to training
        rewards, entropies, alphas, losses = train_sac(env, agent, num_episodes, render_every, save_renders=True, save_path=save_path)
        avg_reward = np.mean(rewards[-100:])  # Average reward over last 100 episodes
        env.close()

        # Return sampled parameters, average reward, and the SAC agent
        return {
            **simple_params,
            "Average Reward": avg_reward,
            "agent": agent  # Ensure the SAC agent is returned
        }

    except Exception as e:
        print(f"An error occurred in run_trial: {e}")
        traceback.print_exc()
        raise

# Generate a new hyperparameter set by combining two parents
def crossover(parent1, parent2):
    child = {}
    for key in parent1.keys():
        if random.random() > 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

# Introduce mutations into a hyperparameter set
def mutate(hyperparameters, mutation_rate=0.1):
    for key in hyperparameters.keys():
        if random.random() < mutation_rate:
            if isinstance(hyperparameters[key], float):
                # Apply a small percentage-based mutation for float hyperparameters
                hyperparameters[key] += hyperparameters[key] * random.uniform(-0.1, 0.1)
            else:
                if key == 'batch_size':
                    hyperparameters[key] = random.choice([32, 64, 128, 256])
                elif key == 'replay_init':
                    hyperparameters[key] = random.choice([500, 1000, 5000])
    return hyperparameters

# Initialize a random population of hyperparameter sets
def initialize_population(pop_size, action_dim):
    population = []
    for _ in range(pop_size):
        hyperparameters = {
            'gamma': random.uniform(0.98, 0.999),  # Random float for gamma
            'tau': random.uniform(0.005, 0.02),  # Random float for tau
            'alpha_initial': random.uniform(0.1, 0.3),  # Random float for initial alpha
            'target_entropy': random.uniform(-action_dim, -action_dim/2),  # Random float for target entropy
            'actor_lr': random.uniform(1e-5, 1e-3),  # Random float for actor learning rate
            'critic_lr': random.uniform(1e-5, 1e-3),  # Random float for critic learning rate
            'memory_size': 1000000,
            'batch_size': random.choice([32, 64, 128, 256]),
            'replay_init': random.choice([500, 1000, 5000])
        }
        population.append(hyperparameters)
    return population

# Evolve the population over multiple generations
def evolve_population(env_name, population, num_generations, num_episodes, render_every):
    best_params = None
    best_avg_reward = -np.inf

    for generation in range(num_generations):
        print(f"\n--- Generation {generation + 1} ---")
        with Pool(processes=cpu_count()) as pool:
            trial_results = pool.starmap(run_trial, [(env_name, params, num_episodes, render_every) for params in population])

        # Filter out any trials that failed or returned None
        trial_results = [result for result in trial_results if result is not None]

        # Sort results by average reward (descending)
        trial_results.sort(key=lambda x: x['Average Reward'], reverse=True)

        # Debug: Print all rewards for this generation
        for result in trial_results:
            print(f"Reward: {result['Average Reward']} for parameters: {result}")

        # Handle the case where no valid trials are produced
        if not trial_results:
            print("No valid trials produced in this generation.")
            continue

        # Keep the top 50% as the parents for the next generation
        num_parents = max(len(trial_results) // 2, 1)  # Ensure at least one parent
        parents = trial_results[:num_parents]
        
        # Update best parameters and best reward
        if parents and parents[0]["Average Reward"] > best_avg_reward:
            best_avg_reward = parents[0]["Average Reward"]
            best_params = parents[0]

        print(f"Best reward in this generation: {best_avg_reward}")

        # Generate new population through crossover and mutation
        new_population = []
        if len(parents) >= 2:
            for _ in range(len(population) - num_parents):
                parent1, parent2 = random.sample(parents, 2)
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
        else:
            print("Not enough parents to sample from, carrying over existing parents.")
            new_population = [mutate(parent.copy()) for parent in parents]  # If only one parent or less, mutate and carry them over

        # Combine the parents and new offspring to form the new population
        population = new_population + parents  # Keep the new_population first to avoid SAC objects

    return best_params, best_avg_reward

if __name__ == "__main__":
    env_name = "LunarLanderContinuous-v2"
    population_size = 10
    num_generations = 10
    num_episodes = 100
    render_every = 20

    env = gym.make(env_name)
    action_dim = env.action_space.shape[0]

    population = initialize_population(population_size, action_dim)
    best_params, best_avg_reward = evolve_population(
        env_name=env_name,
        population=population,
        num_generations=num_generations,
        num_episodes=num_episodes,
        render_every=render_every
    )

    print(f"Best hyperparameters found: {best_params}")
    print(f"Best average reward: {best_avg_reward}")
