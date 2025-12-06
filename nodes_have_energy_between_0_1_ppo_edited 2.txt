# =====================================================
# الگوریتم PPO برای مسئله‌ی انتخاب مجموعه‌ی غالب (Dominating Set)
# نسخه‌ی بهبود یافته برای مقاله:
#   ✅ Masked PPO (هماهنگی act و evaluate)
#   ✅ نرخ یادگیری جداگانه Actor/Critic
#   ✅ گرادیان‌کلیپ برای پایداری
#   ✅ مینی‌بچ در آپدیت PPO
#   ✅ تنظیمات پایدارتر PPO
# =====================================================

import os, random, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_FIXED_SEED = False
GLOBAL_RANDOM_SEED = 42
SAVE_DIR = "ppo_models"
os.makedirs(SAVE_DIR, exist_ok=True)

def set_seeds(seed=GLOBAL_RANDOM_SEED):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if USE_FIXED_SEED:
    set_seeds()

# =====================================================
# ۱️⃣ ساخت شبکه حسگر
# =====================================================
class Sensor():
    def __init__(self, model, id ,degree, neighbors):
        self.xd = self.yd = 0
        self.E  = 1.0
        self.degree = degree
        self.Type = 'Node'
        self.color = 'white'
        if id == model.nodes:
            self.xd = model.Sinkx
            self.yd = model.Sinky
            self.E = 100
            self.Type = 'BS'
            self.color = 'black'
        self.ID = id
        self.neighbors = np.array(neighbors, dtype=int)

class Model():
    def __init__(self):
        self.init_energy = 1.0
        self.Area = 100
        self.nodes = 50
        self.Sinkx = self.Sinky = self.Area/2
        self.ep = 0.1
        self.delta = 0.001

def network_builder_definitedegree(count_nodes, count_edges):
    matrix = np.zeros((count_nodes + 1, count_nodes + 1))
    nodes_degree = np.zeros(count_nodes + 1, dtype=int)
    neighbors_nodes = []
    for _ in range(count_edges):
        n1 = random.randint(0, count_nodes - 1)
        n2 = random.randint(0, count_nodes - 1)
        while n1 == n2 or matrix[n1,n2] == 1:
            n1 = random.randint(0, count_nodes - 1)
            n2 = random.randint(0, count_nodes - 1)
        matrix[n1,n2] = matrix[n2,n1] = 1

    matrix[:, count_nodes] = 1
    matrix[count_nodes,:] = 1

    for i in range(nodes_degree.shape[0]):
        nodes_degree[i] = int(np.sum(matrix[i, :]))
        neighbors_nodes.append([k for k in range(matrix.shape[0]) if matrix[i,k]==1])

    return nodes_degree, neighbors_nodes, matrix

def networkbuilder():
    model = Model()
    count_edges = 1205
    degree, nodes_neighbors, matrix = network_builder_definitedegree(model.nodes, count_edges)
    sensors = [Sensor(model, i, degree[i], nodes_neighbors[i]) for i in range(model.nodes + 1)]
    for i in range(model.nodes):
        sensors[i].E = random.uniform(0.1, 1.0)
    return sensors, model, matrix

# =====================================================
# ۲️⃣ محیط یادگیری تقویتی
# =====================================================
class DominatingSetEnv:
    def __init__(self, sensors, model, matrix):
        self.sensors = sensors
        self.model   = model
        self.matrix  = matrix
        self.N       = model.nodes
        self.ep      = model.ep
        self.energy_threshold = model.ep
        self.active = np.zeros(self.N, dtype=int)
        self.dominated = np.zeros(self.N+1, dtype=int)
        self.max_steps = self.N * 2
        self.step_count = 0
        self.deg = np.array([s.degree for s in sensors[:self.N]], dtype=float)
        self._update_dominated()
        self.total_energy_before = sum([s.E for s in self.sensors[:self.N]])

    def reset(self):
        self.active[:] = 0
        self.dominated[:] = 0
        for s in self.sensors[:self.N]:
            s.E = random.uniform(0.1, 1.0)
        self.step_count = 0
        self._update_dominated()
        self.total_energy_before = sum([s.E for s in self.sensors[:self.N]])
        return self._state()

    def _update_dominated(self):
        self.dominated[:] = 0
        for i in range(self.N):
            if self.active[i] == 1:
                self.dominated[i] = 1
                for nb in self.sensors[i].neighbors:
                    self.dominated[nb] = 1
        self.dominated[-1] = 1

    def _state(self):
        energies = np.array([s.E for s in self.sensors[:self.N]], dtype=float)
        dominated_nodes = self.dominated[:self.N].astype(float)
        frac_active = float(self.active.mean())
        mean_en = float(energies.mean())
        var_en  = float(energies.var())
        rem = float(self.N - dominated_nodes.sum())
        return np.concatenate(
            [energies, self.deg, dominated_nodes,
             np.array([frac_active, mean_en, var_en, rem], dtype=float)],
            axis=0
        )

    def valid_actions(self):
        valids = np.ones(self.N+1, dtype=bool)
        for i in range(self.N):
            if self.active[i]==1 or self.sensors[i].E < self.energy_threshold:
                valids[i] = False
        valids[self.N] = self._is_all_dominated()
        return valids

    def _is_all_dominated(self):
        return bool(np.all(self.dominated[:self.N] == 1))

    def step(self, action):
        self.step_count += 1
        prev_dom = self.dominated[:self.N].sum()
        reward = 0.0
        done = False

        # --- STOP action ---
        if action == self.N:
            if self._is_all_dominated():
                cost = self.active.sum()
                total_energy_after = sum([s.E for s in self.sensors[:self.N]])
                lifetime_loss = (self.total_energy_before - total_energy_after)
                reward = +10.0 - 0.5 * cost - 2.0 * lifetime_loss
            else:
                reward = -5.0
            done = True

        # --- activate node ---
        else:
            if self.active[action]==1 or self.sensors[action].E < self.energy_threshold:
                reward = -2.0
            else:
                self.active[action] = 1

                if self.sensors[action].E > 0.7:
                    reward += 0.3
                if self.sensors[action].E < 0.2:
                    reward -= 0.5

                self._update_dominated()
                now_dom = self.dominated[:self.N].sum()

                # reward for new coverage
                reward += 0.05 * (now_dom - prev_dom)

                # energy decay (no negative)
                self.sensors[action].E = max(0, self.sensors[action].E - self.model.ep)

        # max steps condition
        if self.step_count >= self.max_steps:
            done = True
            if not self._is_all_dominated():
                reward -= 3.0

        # low mean energy condition
        if np.mean([s.E for s in self.sensors[:self.N]]) < self.energy_threshold:
            done = True
            reward -= 3.0

        return self._state(), float(reward), done, {}

# =====================================================
# ۳️⃣ Actor–Critic
# =====================================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def _masked_probs(self, logits, valid_mask):
        probs = torch.softmax(logits, dim=-1)
        probs = probs * valid_mask
        if probs.sum() <= 0:
            probs = torch.ones_like(probs) / probs.numel()
        else:
            probs = probs / probs.sum()
        return probs

    def act(self, state, valid_mask):
        logits = self.actor(state)
        probs = self._masked_probs(logits, valid_mask)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def evaluate(self, states, actions, valid_masks):
        logits = self.actor(states)
        probs = torch.softmax(logits, dim=-1)
        probs = probs * valid_masks
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(probs_sum > 0, probs / probs_sum, torch.ones_like(probs)/probs.shape[-1])

        dist = Categorical(probs)
        logprobs = dist.log_prob(actions)
        entropy  = dist.entropy()
        values   = self.critic(states).squeeze(-1)
        return logprobs, values, entropy

# =====================================================
# ۴️⃣ PPO (بهبود یافته)
# =====================================================
class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_lr=1e-4,
        critic_lr=5e-5,
        gamma=0.99,
        eps_clip=0.1,
        K_epochs=8,
        entropy_coef=0.02,
        batch_size=256,
        grad_clip=0.5
    ):
        self.policy = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.policy_old = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam([
            {"params": self.policy.actor.parameters(), "lr": actor_lr},
            {"params": self.policy.critic.parameters(), "lr": critic_lr},
        ])

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # ---- compute discounted returns ----
        returns = []
        discounted = 0
        for r, done in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if done:
                discounted = 0
            discounted = r + self.gamma * discounted
            returns.insert(0, discounted)

        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        old_states   = torch.stack(memory.states).detach()
        old_actions  = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()
        old_masks    = torch.stack(memory.valid_masks).detach()

        n = old_states.size(0)
        idxs = torch.arange(n)

        for _ in range(self.K_epochs):
            perm = idxs[torch.randperm(n)]
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                batch_idx = perm[start:end]

                states_b   = old_states[batch_idx]
                actions_b  = old_actions[batch_idx]
                logprobs_b = old_logprobs[batch_idx]
                returns_b  = returns[batch_idx]
                masks_b    = old_masks[batch_idx]

                logprobs, values, entropy = self.policy.evaluate(states_b, actions_b, masks_b)

                ratios = torch.exp(logprobs - logprobs_b)

                advantages = returns_b - values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = self.MseLoss(values, returns_b)
                entropy_loss = -entropy.mean()

                loss = actor_loss + 0.5*critic_loss + self.entropy_coef*entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

# =====================================================
# حافظه تجربه
# =====================================================
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.valid_masks = []

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.valid_masks.clear()

# =====================================================
# ۵️⃣ آموزش PPO
# =====================================================
def train_ppo(episodes=1000, seed=None):
    if seed is not None:
        set_seeds(seed)

    sensors, model, matrix = networkbuilder()
    env = DominatingSetEnv(sensors, model, matrix)

    state_dim = env._state().shape[0]
    action_dim = env.N + 1

    memory = Memory()
    ppo = PPO(state_dim, action_dim)

    episode_rewards = []
    episode_lifetimes = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        lifetime = 0

        for t in range(env.max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)

            valid_mask_np = env.valid_actions()
            valid_mask = torch.tensor(valid_mask_np, dtype=torch.float32, device=DEVICE)

            action, logprob, _ = ppo.policy_old.act(state_tensor, valid_mask)

            new_state, reward, done, _ = env.step(action)

            # store rollout
            memory.states.append(state_tensor)
            memory.actions.append(torch.tensor(action, device=DEVICE))
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            memory.valid_masks.append(valid_mask)

            ep_reward += reward
            lifetime += 1
            state = new_state

            if done:
                break

        ppo.update(memory)
        memory.clear()

        episode_rewards.append(ep_reward)
        episode_lifetimes.append(lifetime)

        if ep % 50 == 0:
            avg_r = np.mean(episode_rewards[-50:])
            avg_l = np.mean(episode_lifetimes[-50:])
            print(f"[PPO] Episode {ep} | Avg Reward(50)={avg_r:.3f} | Avg Lifetime(50)={avg_l:.1f}")

    torch.save(ppo.policy.state_dict(), os.path.join(SAVE_DIR, "ppo_final.pt"))

    print("\nTraining complete.")
    print("Mean Final Reward:", np.mean(episode_rewards[-100:]))
    print("Mean Network Lifetime:", np.mean(episode_lifetimes[-100:]))
    return episode_rewards, episode_lifetimes


# اجرای مستقیم
if __name__ == "__main__":
    train_ppo(episodes=1000)
