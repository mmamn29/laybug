# compare_ppo_vs_lbo_lifetime.py
import os, random, math
import numpy as np
import torch
import matplotlib.pyplot as plt

# ============================================================
# 1) Import your PPO environment + training code
#    !!! اگر اسم فایل PPO شما متفاوت است این خط را تغییر دهید
# ============================================================
from nodes_have_energy_between_0_1_ppo_edited import (
    networkbuilder, DominatingSetEnv, PPO, train_ppo
)

DEVICE = torch.device(cuda if torch.cuda.is_available() else cpu)
os.makedirs(results, exist_ok=True)


# ============================================================
# 2) LBO (Ladybug  Kafshdozak) as a function
#    This is your original algorithm but WITHOUT globals.
#    It returns a Dominating-Set position (01 list) or None.
# ============================================================
def lbo_find_dominating_set(sensors, model, matrix, params)
    energy_threshold = model.ep
    nvar = model.nodes

    # ---------- helper functions (same logic as your code) ----------
    def calc_mean()
        energys = sum([s.E for s in sensors[model.nodes]])
        return energys  model.nodes

    def is_dominating_set(solution)
        dominated = np.zeros(len(sensors), dtype=int)
        for i in range(len(sensors)-1)
            if solution[i] == 1
                dominated[i] = 1
                for nb in sensors[i].neighbors
                    dominated[nb] = 1
        dominated[-1] = 1  # BS dominated
        return np.all(dominated == 1)

    def fitness(solution)
        if not is_dominating_set(solution)
            return float('inf')

        cost = np.sum(solution)
        meanE = calc_mean()

        for i in range(len(solution))
            if solution[i] == 1
                if sensors[i].E  meanE
                    cost -= 0.5
                if sensors[i].E = meanE - 0.1
                    cost += 10
                if sensors[i].E  0.2
                    cost += 5
        return cost

    def tournament_selection(pop, k=3)
        inds = random.sample(range(len(pop)), k)
        best = min(inds, key=lambda idx pop[idx]['cost'])
        return best

    def mutate(x, mu, sigma)
        y = x.copy()
        flag = np.random.rand(len(y)) = mu
        ind = np.argwhere(flag)
        for idx in ind
            y[idx[0]] += sigmanp.random.randn()
        return y

    # ---------- unpack params ----------
    max_NFE = params['max_NFE']
    npop = params['npop']
    npop_init = npop
    beta = params['beta']
    sigma = params['sigma']

    # ---------- initialize population ----------
    bestsol = {position [0]nvar, cost np.inf}
    pop = []
    NFE = 0

    for _ in range(npop)
        p = {position []}
        for m in range(nvar)
            if sensors[m].E = energy_threshold
                p[position].append(np.random.randint(0, 2))
            else
                p[position].append(0)
        p[cost] = fitness(p[position])
        pop.append(p)
        NFE += 1
        if p[cost]  bestsol[cost]
            bestsol = p.copy()

    # if all invalid - network dead
    if all(p['cost'] == float('inf') for p in pop)
        return None

    # ---------- main LBO loop ----------
    it = 0
    while NFE  max_NFE
        it += 1
        costs = np.array([x[cost] for x in pop], dtype=float)
        SoC = sum(costs)
        avg_cost = np.mean(costs)
        if avg_cost != 0
            costs = costs  avg_cost
        probs = np.exp(-betacosts)

        newSol = []
        for i in range(npop)
            new = {cost float('inf')}
            j = 0
            while (j  2 or j  npop)
                j = tournament_selection(pop)

            if random.random()  0.2
                Rnd = np.random.random() - 0.5
                new_pos = (
                    np.array(pop[j][position])
                    + np.random.random(size=nvar)  (np.array(pop[j][position]) - np.array(pop[i][position]))
                    + np.random.random(size=nvar)  (np.array(pop[j-1][position]) - np.array(pop[j][position]))
                )
                new[position] = new_pos.tolist()
            else
                new[position] = mutate(pop[i][position], 0.05 * nvar, sigma)

            # binarize
            new[position] = [1 if v  0.5 else 0 for v in new[position]]

            # energy constraint
            for k in range(len(new[position]))
                if sensors[k].E  energy_threshold
                    new[position][k] = 0

            new[cost] = fitness(new[position])
            newSol.append(new)
            NFE += 1

            if new[cost]  bestsol[cost]
                bestsol = new.copy()

        # population reduction (same as your code)
        npop = npop - 0.1np.random.random()npop(NFEmax_NFE)
        npop = round(npop)
        npop = max(math.floor(npop_init4), npop)

        pop += newSol
        pop = sorted(pop, key=lambda x x[cost])
        pop = pop[npop]

    return bestsol[position]


# ============================================================
# 3) Outer-loop lifetime evaluation for LBO
#    EXACTLY like your original lifetime loop.
# ============================================================
def eval_lbo_lifetime(runs=30)
    params = {'max_NFE' 4000, 'npop' 50, 'beta' 8, 'sigma' 0.05}
    lifetimes = []

    for _ in range(runs)
        sensors, model, matrix = networkbuilder()
        energy_threshold = model.ep
        life_time = 0

        def is_finished()
            for s in sensors[model.nodes]
                if s.E  energy_threshold
                    return False
            return True

        while not is_finished()
            best_pos = lbo_find_dominating_set(sensors, model, matrix, params)
            if best_pos is None
                break

            active = [i for i, v in enumerate(best_pos) if v == 1]
            inactive = [i for i, v in enumerate(best_pos) if v == 0]

            for i in active
                sensors[i].E -= model.ep
            for i in inactive
                sensors[i].E += model.delta

            life_time += 1

        lifetimes.append(life_time)

    return np.array(lifetimes)


# ============================================================
# 4) PPO find dominating set using trained policy (one episode)
# ============================================================
def ppo_find_dominating_set(env, ppo)
    state = env._state()

    for _ in range(env.max_steps)
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        valid_mask = torch.tensor(env.valid_actions(), dtype=torch.float32, device=DEVICE)

        with torch.no_grad()
            action, _, _ = ppo.policy_old.act(state_t, valid_mask)

        state, r, done, _ = env.step(int(action))
        if done
            break

    if env._is_all_dominated()
        return env.active.copy()
    return None


# ============================================================
# 5) Outer-loop lifetime evaluation for PPO (same as LBO loop)
# ============================================================
def eval_ppo_lifetime(ppo, runs=30)
    lifetimes = []

    for _ in range(runs)
        sensors, model, matrix = networkbuilder()
        env = DominatingSetEnv(sensors, model, matrix)

        energy_threshold = model.ep
        life_time = 0

        def is_finished()
            for s in sensors[model.nodes]
                if s.E  energy_threshold
                    return False
            return True

        while not is_finished()
            # IMPORTANT reset only activedominated; energies stay as current sensors energies
            env.reset()

            best_active = ppo_find_dominating_set(env, ppo)
            if best_active is None
                break

            active = [i for i, v in enumerate(best_active) if v == 1]
            inactive = [i for i, v in enumerate(best_active) if v == 0]

            for i in active
                sensors[i].E -= model.ep
            for i in inactive
                sensors[i].E += model.delta

            life_time += 1

        lifetimes.append(life_time)

    return np.array(lifetimes)


# ============================================================
# 6) Main comparison
# ============================================================
def main()
    # ---- Train PPO once ----
    ppo_train_rewards, ppo_train_lifetimes = train_ppo(episodes=500)

    # ---- Build temp env to get dimensions ----
    sensors, model, matrix = networkbuilder()
    env_tmp = DominatingSetEnv(sensors, model, matrix)
    state_dim = env_tmp._state().shape[0]
    action_dim = env_tmp.N + 1

    # ---- Load trained PPO model ----
    ppo = PPO(state_dim=state_dim, action_dim=action_dim)
    ppo.policy.load_state_dict(torch.load(ppo_modelsppo_final.pt, map_location=DEVICE))
    ppo.policy_old.load_state_dict(ppo.policy.state_dict())
    ppo.policy.eval()
    ppo.policy_old.eval()

    # ---- Evaluate lifetimes ----
    lbo_life = eval_lbo_lifetime(runs=30)
    ppo_life = eval_ppo_lifetime(ppo, runs=30)

    print(===================================================)
    print(Mean lifetime LBO, lbo_life.mean(),  +- , lbo_life.std())
    print(Mean lifetime PPO, ppo_life.mean(),  +- , ppo_life.std())
    print(===================================================)

    # ---- Plot learning curves PPO ----
    plt.figure()
    plt.plot(ppo_train_rewards)
    plt.xlabel(Episode)
    plt.ylabel(Reward)
    plt.title(PPO Training Reward)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(resultsppo_learning_reward.png, dpi=200)

    plt.figure()
    plt.plot(ppo_train_lifetimes)
    plt.xlabel(Episode)
    plt.ylabel(Lifetime per episode)
    plt.title(PPO Training Lifetime)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(resultsppo_learning_lifetime.png, dpi=200)

    # ---- Boxplot lifetime ----
    plt.figure()
    plt.boxplot([lbo_life, ppo_life], labels=[LBO, PPO])
    plt.ylabel(Outer-loop Lifetime)
    plt.title(Lifetime Comparison (Outer Loop))
    plt.grid(True, axis=y, alpha=0.3)
    plt.tight_layout()
    plt.savefig(resultscompare_lifetime_box.png, dpi=200)

    # ---- ECDF lifetime ----
    def ecdf(x)
        x = np.sort(x)
        y = np.arange(1, len(x)+1)  len(x)
        return x, y

    x1, y1 = ecdf(lbo_life)
    x2, y2 = ecdf(ppo_life)

    plt.figure()
    plt.plot(x1, y1, label=LBO)
    plt.plot(x2, y2, label=PPO)
    plt.xlabel(Outer-loop Lifetime)
    plt.ylabel(ECDF)
    plt.title(Lifetime ECDF)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(resultscompare_lifetime_cdf.png, dpi=200)

    # ---- Save numeric results ----
    np.savez(resultssummary_lifetime.npz,
             lbo_lifetime=lbo_life,
             ppo_lifetime=ppo_life,
             ppo_train_rewards=np.array(ppo_train_rewards),
             ppo_train_lifetimes=np.array(ppo_train_lifetimes))

    print(Saved all figures and summary to .results)


if __name__ == __main__
    main()

