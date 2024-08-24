import random
import numpy as np
import tensorflow as tf
import gymnasium as gym
from multiprocessing import Pool, cpu_count
import os
import pandas as pd
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

# Efficient Replay Buffer using NumPy arrays
class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state_buffer = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_state_buffer = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((max_size, 1), dtype=np.float32)
        self.done_buffer = np.zeros((max_size, 1), dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        self.state_buffer[self.ptr] = state
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.next_state_buffer[self.ptr] = next_state
        self.done_buffer[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return dict(state=self.state_buffer[indices],
                    next_state=self.next_state_buffer[indices],
                    action=self.action_buffer[indices],
                    reward=self.reward_buffer[indices],
                    done=self.done_buffer[indices])

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
        self.alpha = tf.Variable(alpha_initial, trainable=True, dtype=tf.float32)
        self.target_entropy = target_entropy

        self.memory = ReplayBuffer(memory_size, state_dim, action_dim)
        self.batch_size = batch_size
        self.replay_init = replay_init

        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

    def select_action(self, state):
        state = np.expand_dims(state, axis=0)  # Ensure correct shape
        return self.actor(state)[0].numpy()

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

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

        # Apply gradient clipping
        critic_1_grads = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grads = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        clipped_critic_1_grads = [tf.clip_by_norm(g, 1.0) for g in critic_1_grads]
        clipped_critic_2_grads = [tf.clip_by_norm(g, 1.0) for g in critic_2_grads]
        self.critic_1_optimizer.apply_gradients(zip(clipped_critic_1_grads, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(clipped_critic_2_grads, self.critic_2.trainable_variables))

        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            actor_loss = tf.reduce_mean(self.alpha * new_actions - tf.minimum(self.critic_1([states, new_actions]), self.critic_2([states, new_actions])))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        clipped_actor_grads = [tf.clip_by_norm(g, 1.0) for g in actor_grads]
        self.actor_optimizer.apply_gradients(zip(clipped_actor_grads, self.actor.trainable_variables))

        # Alpha loss and optimization
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            log_probs = -tf.reduce_sum(tf.square(actions), axis=1, keepdims=True)  # Approximation for log probabilities
            alpha_loss = -tf.reduce_mean(self.alpha * (log_probs + self.target_entropy))

        alpha_grads = tape.gradient(alpha_loss, [self.alpha])
        clipped_alpha_grads = [tf.clip_by_norm(g, 1.0) for g in alpha_grads]
        self.alpha_optimizer.apply_gradients(zip(clipped_alpha_grads, [self.alpha]))

        for target_param, param in zip(self.target_critic_1.trainable_variables, self.critic_1.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)

        for target_param, param in zip(self.target_critic_2.trainable_variables, self.critic_2.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)

    def train(self):
        if self.memory.size < max(self.batch_size, self.replay_init):
            return

        batch = self.memory.sample_batch(self.batch_size)
        states = tf.convert_to_tensor(batch['state'], dtype=tf.float32)
        actions = tf.convert_to_tensor(batch['action'], dtype=tf.float32)
        rewards = tf.convert_to_tensor(batch['reward'], dtype=tf.float32)
        next_states = tf.convert_to_tensor(batch['next_state'], dtype=tf.float32)
        dones = tf.convert_to_tensor(batch['done'], dtype=tf.float32)

        self.train_step(states, actions, rewards, next_states, dones)

    def save_models(self, path=""):
        self.actor.save(os.path.join(path, "best_actor_model.keras"))
        self.critic_1.save(os.path.join(path, "best_critic_1_model.keras"))
        self.critic_2.save(os.path.join(path, "best_critic_2_model.keras"))
        self.target_critic_1.save(os.path.join(path, "best_target_critic_1_model.keras"))
        self.target_critic_2.save(os.path.join(path, "best_target_critic_2_model.keras"))

def train_sac(env, agent, num_episodes, render_every=10):
    all_rewards = []
    all_entropies = []
    all_alphas = []
    all_losses = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
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

        print(f"Episode {episode}, Reward: {episode_reward}")

    env.close()

    return all_rewards, all_entropies, all_alphas, all_losses

def run_trial(env_name, sampled_params, num_episodes, render_every):
    try:
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
        # print(f"Replay buffer size after warm-up: {agent.memory.size}")

        # Now proceed to training
        rewards, entropies, alphas, losses = train_sac(env, agent, num_episodes, render_every)
        avg_reward = np.mean(rewards[-100:])  # Average reward over last 100 episodes
        env.close()

        # Return sampled parameters, average reward, and the SAC agent
        return {
            **sampled_params,
            "Average Reward": avg_reward
        }

    except Exception as e:
        print(f"An error occurred in run_trial: {e}")
        traceback.print_exc()
        raise

# Generate a new hyperparameter set by combining two parents with fitness sharing
def crossover(parent1, parent2):
    child = {}
    for key in parent1.keys():
        if random.random() > 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

# Introduce mutations into a hyperparameter set with adaptive mutation rates
def mutate(hyperparameters, mutation_rate=0.1, diversity_factor=0.1):
    for key in hyperparameters.keys():
        if random.random() < mutation_rate:
            if isinstance(hyperparameters[key], float):
                # Apply a small percentage-based mutation for float hyperparameters
                hyperparameters[key] += hyperparameters[key] * random.uniform(-diversity_factor, diversity_factor)
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

# Evolve the population over multiple generations with enhanced strategies
def evolve_population(env_name, population, num_generations, num_episodes, render_every):
    best_params = None
    best_avg_reward = -np.inf
    all_results = []  # To store results of all trials

    for generation in range(num_generations):
        print(f"\n--- Generation {generation + 1} ---")
        with Pool(processes=NUM_OF_PROCESSES) as pool:
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

        # Apply fitness sharing and crossover for the new population
        new_population = []
        if len(parents) >= 2:
            for _ in range(len(population) - num_parents):
                parent1, parent2 = random.sample(parents, 2)
                child = crossover(parent1, parent2)
                # Adjust mutation rate based on diversity
                diversity_factor = 1.0 - (best_avg_reward / (np.mean([p["Average Reward"] for p in parents]) + 1e-8))
                child = mutate(child, mutation_rate=0.1, diversity_factor=diversity_factor)
                new_population.append(child)
        else:
            print("Not enough parents to sample from, carrying over existing parents.")
            new_population = [mutate(parent.copy()) for parent in parents]  # If only one parent or less, mutate and carry them over

        # Combine the parents and new offspring to form the new population
        population = new_population + parents  # Keep the new_population first to avoid SAC objects

        # Add results of this generation to the all_results list
        all_results.extend(trial_results)

    # Save all results to a CSV file
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results.csv", index=False)

    return best_params, best_avg_reward

if __name__ == "__main__":
    env_name = "LunarLanderContinuous-v2"
    # print(cpu_count())
    population_size = 20
    num_generations = 50
    num_episodes = 100
    render_every = 20
    NUM_OF_PROCESSES = 10

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
