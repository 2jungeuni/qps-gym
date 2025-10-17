import numpy as np

import pygame
import gymnasium as gym
from gymnasium import spaces

from sim.qps_sim import *
from render.qps_render import *


class LogisticsQPSEnv(gym.Env):
    """
    물류창고의 정적 할당과 동적 주문 순서 결정을 결합한 Gymnasium 환경
    """
    metadata = {"render_nodes": ["human"], "render_fps": 30}

    def __init__(self, static_optimizer, sim_params, render_mode=None):
        super().__init__()

        # ⚠️ 렌더링 여부와 관계 없이 Pygame 모듈을 사용하므로 항상 초기화함
        pygame.init()

        self.static_optimizer = static_optimizer
        self.sim_params = sim_params
        self.simulator = LogisticsQPSSimulator(sim_params)

        # 환경 정보
        MAX_STATIONS = 15  # 최대 스테이션 개수 # TODO: 알맞게 바꾸어야 함!
        MAX_CLUSTERS = 1000  # 최대 택배 상자 개수 # TODO: 알맞게 바꾸어야 함!
        self.action_space = spaces.Discrete(MAX_CLUSTERS)
        self.observation_space = spaces.Dict({
            "time_elapsed": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "station_status": spaces.Box(low=0, high=1, shape=(MAX_STATIONS,), dtype=np.int8),
            "station_loads": spaces.Box(low=0, high=np.inf, shape=(MAX_STATIONS,), dtype=np.float32),
            "queue_lengths": spaces.Box(low=0, high=sim_params["queue_limit"], shape=(MAX_STATIONS,), dtype=np.int8),
            "pending_mask": spaces.Box(low=0, high=1, shape=(MAX_CLUSTERS,), dtype=np.int8)
        })

        self.render_mode = render_mode
        self.renderer = LogisticsQPSRenderer(self.sim_params) if self.render_mode == "human" else None

    def _get_obs(self):
        state = self.simulator._get_state_snapshot()
        station_status = np.array([1 if s['status'] == 'busy' else 0 for s in state['station_status']], dtype=np.int8)
        station_loads = state['station_loads'].astype(np.float32)
        queue_lengths = np.array([len(q) for q in state['queues']], dtype=np.int8)
        pending_mask = np.zeros(self.action_space.n, dtype=np.int8)
        if state['pending_cluster_ids']:
            pending_mask[list(state['pending_cluster_ids'])] = 1

        obs = {
            "time_elapsed": np.array([state['time']], dtype=np.float32),
            "station_status": np.pad(station_status,
                                     (0, self.observation_space['station_status'].shape[0] - len(station_status))),
            "station_loads": np.pad(station_loads,
                                    (0, self.observation_space['station_loads'].shape[0] - len(station_loads))),
            "queue_lengths": np.pad(queue_lengths,
                                    (0, self.observation_space['queue_lengths'].shape[0] - len(queue_lengths))),
            "pending_mask": pending_mask
        }
        return obs

    def reset(self, static_plan=None, seed=None, options=None):
        super().reset(seed=seed)
        if static_plan is not None:
            self.static_plan = static_plan
        else:
            self.static_plan = self.static_optimizer()
        self.simulator.reset(self.static_plan)
        self.last_time = 0.0
        return self._get_obs(), {}

    def step(self, action: int):
        action_penalty = 0.0
        if action != -1:
            if action in self.simulator.pending_cluster_ids:
                self.simulator.inject_cluster(action)
            else:
                action_penalty = -1.0

        # 시뮬레이션 시간을 1초 진행시키고 그동안 발생한 이벤트들을 받아옴
        processed_events, terminated = self.simulator.move_one_second()

        # 보상 계산: 시간 경과에 대한 기본 페널티 (-1.0) + 이벤트 보상/패널티
        reward = -1.0

        for event in processed_events:
            if event["type"] == "BYPASSED":
                reward -= 50.0
            elif event["type"] == "COMPLETED_CLUSTER":
                reward += 100.0

        reward += action_penalty    # 액션 패널티는 발생할 일이 거의 없어야 함

        if terminated:
            final_stats = self.simulator._get_state_snapshot()
            makespan = final_stats['time']
            reward += 10000 / makespan if makespan > 0 else 0
            busy_times = final_stats['station_total_busy_time']
            if np.sum(busy_times) > 0:
                cv = np.std(busy_times) / np.mean(busy_times)
                reward -= 500 * cv

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.renderer:
            state_snapshot = self.simulator._get_state_snapshot()
            self.renderer.render(state_snapshot)

    def close(self):
        if self.renderer:
            self.renderer.close()