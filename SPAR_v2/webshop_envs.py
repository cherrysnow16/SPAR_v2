"""WebShop vector env with several envs per Ray actor.

Same semantics as agent_system's WebshopMultiProcessEnv (per-env seeds, goal split, reset sampling),
but each actor hosts envs_per_actor WebshopWorker instances instead of one: every actor process
carries ~37 threads (Ray core + pyserini JVM), so 256 single-env actors exceed this container's
pids.max (11776).
"""
import numpy as np
import ray

from agent_system.environments.env_package.webshop.envs import WebshopWorker


class _WebshopGroupWorker:
    def __init__(self, seeds, env_kwargs):
        self.envs = [WebshopWorker(seed, dict(env_kwargs)) for seed in seeds]

    def step(self, actions):
        return [env.step(a) for env, a in zip(self.envs, actions)]

    def reset(self, idxs):
        return [env.reset(i) for env, i in zip(self.envs, idxs)]

    def get_goals(self):
        return self.envs[0].get_goals()

    def close(self):
        for env in self.envs:
            env.close()


class WebshopGroupedEnv:
    def __init__(self, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs=None,
                 envs_per_actor=8):
        if not ray.is_initialized():
            ray.init()
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train:
            assert group_n == 1
        self._rng = np.random.RandomState(seed)
        env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        seeds = [seed + (i // group_n) for i in range(self.num_processes)]
        self._chunks = [list(range(s, min(s + envs_per_actor, self.num_processes)))
                        for s in range(0, self.num_processes, envs_per_actor)]
        actor_cls = ray.remote(**resources_per_worker)(_WebshopGroupWorker)
        self._workers = [actor_cls.remote([seeds[i] for i in chunk], env_kwargs) for chunk in self._chunks]

        goals = ray.get(self._workers[0].get_goals.remote())
        self.goal_idxs = range(500) if not is_train else range(500, len(goals))

    def _gather(self, futures):
        return [r for chunk_results in ray.get(futures) for r in chunk_results]

    def step(self, actions):
        if len(actions) != self.num_processes:
            raise ValueError(f'Expected {self.num_processes} actions, got {len(actions)}')
        results = self._gather([w.step.remote([actions[i] for i in chunk])
                                for w, chunk in zip(self._workers, self._chunks)])
        obs, rewards, dones, infos = map(list, zip(*results))
        return obs, rewards, dones, infos

    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()
        results = self._gather([w.reset.remote([idx[i] for i in chunk])
                                for w, chunk in zip(self._workers, self._chunks)])
        obs, infos = map(list, zip(*results))
        return obs, infos

    def close(self):
        if getattr(self, '_closed', False):
            return
        ray.get([w.close.remote() for w in self._workers])
        for w in self._workers:
            ray.kill(w)
        self._closed = True

    def __del__(self):
        self.close()
