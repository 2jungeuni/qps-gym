import time
import copy
import math
import json
import copy
import random
import pygame
import numpy as np
import pandas as pd
from collections import Counter

import argparse

from config import qps_params as cfg
from env import qps_env

from tqdm import tqdm


def generate_mock_logistics_output(fixed_sku_to_station = None,
                                   fixed_invoice_to_sku = None,
                                   fixed_num_stations = None,
                                   fixed_zone_capacity = None):
    """테스트용 정적 할당 계획을 생성"""
    # print("Generating mock logistics data...")
    rng = np.random.default_rng()

    # --- num stations 결정 ---
    if fixed_num_stations is not None:
        num_stations = fixed_num_stations
    else:
        num_stations = rng.integers(10, 15)

    # --- 나머지 변수 설정 ---
    num_skus = rng.integers(50, 70)     # TODO: 원하는 문제 스케일에 맞추어 조정
    num_clusters = rng.integers(20, 30) # TODO: 원하는 문제 스케일에 맞추어 조정

    # 어떤 스테이션에 어떤 SKU
    if not fixed_sku_to_station:
        sku_to_station_map = rng.choice(num_stations, size=num_skus)
    else:
        sku_to_station_map = fixed_sku_to_station.copy()

    sku_loads = 3                               # TODO: 하나의 SKU를 처리하는데 걸리는 시간
    sku_total_quantities = rng.integers(5, 16, size=num_skus)
    total_item_pool = [sku_id for sku_id, total_qty in enumerate(sku_total_quantities) for _ in range(total_qty)]
    rng.shuffle(total_item_pool)

    # 어떤 택배상자에 어떤 SKU
    invoices = [[] for _ in range(num_clusters)]
    if not fixed_invoice_to_sku:
        for sku_item in total_item_pool:
            invoices[rng.integers(num_clusters)].append(sku_item)
    else:
        invoices = fixed_invoice_to_sku.copy()
        num_clusters = len(invoices)

    records = [{
        "cluster_id": cid,
        "sku_id": sku_id,
        "station_id": sku_to_station_map[sku_id],
        "load": sku_loads
    } for cid, sku_list in enumerate(invoices) for sku_id in sku_list]
    assignment_df = pd.DataFrame(records)

    station_stats_df = pd.DataFrame({
        "station_id": range(num_stations),
        "zone_cd": [f"{i + 1:02d}" for i in range(num_stations)]
    })

    # --- Capcity 정보 반영 로직 수정 ---
    if fixed_zone_capacity:
        # 전달받은 용량(dict)을 Series로 변환하여 사용
        capacity_series = pd.Series(fixed_zone_capacity, name="capacity")
        station_stats_df = station_stats_df.merge(capacity_series,
                                                  left_on="station_id",
                                                  right_index=True,
                                                  how="left").fillna(0)
    else:
        # 기존 방식: 할당된 SKU 개수를 용량으로 간주
        station_capacity = pd.Series(sku_to_station_map).value_counts().sort_index().rename("capacity")
        station_stats_df = station_stats_df.merge(station_capacity,
                                                  left_on="station_id",
                                                  right_index=True,
                                                  how="left").fillna(0)

    total_loads = assignment_df.groupby('station_id')['load'].sum().rename('total_load')
    station_stats_df = station_stats_df.merge(total_loads, on='station_id', how='left').fillna(0)

    return {
        "num_stations": num_stations,
        "num_clusters": num_clusters,
        "assignment_df": assignment_df,
        "station_stats_df": station_stats_df
    }

def run_no_gui(data, invoice_list=None, sku_zone_matching=None):
    num_skus = data["num_sku"]
    num_zones = data["num_zone"]
    num_invoices = len(data["invoices"])
    skus = range(num_skus)
    zones = range(num_zones)
    invoices = range(num_invoices)
    skus_per_invoices = data["invoices"]

    # 시뮬레이션 환경
    env = qps_env.LogisticsQPSEnv(static_optimizer=generate_mock_logistics_output,
                                  sim_params=cfg.sim_params,
                                  render_mode="human" if cfg.USE_GUI else None)

    # 데이터를 통한 Station 용량 설정
    total_skus = [sku_id for sku_list in skus_per_invoices if sku_list for sku_id in sku_list]
    zone_cap = {zone: math.ceil(len(total_skus) / num_zones) for zone in range(num_zones)}

    if not sku_zone_matching:
        sku_zone_matching = {}
        # SKU-Zone할당: 각 작업을 가능한 Zone에 무작위로 할당
        tmp_zone_cap = zone_cap.copy()
        for sku_id in skus:
            possible_zones = [zone for zone in zones if tmp_zone_cap[zone] > 0]
            if not possible_zones:
                possible_zones = list(zones)
            rnd_zone = random.choice(possible_zones)
            sku_zone_matching[sku_id] = rnd_zone
            if tmp_zone_cap.get(rnd_zone, 0) > 0:
                tmp_zone_cap[rnd_zone] -= 1

    if invoice_list:
        invoice_list = invoice_list
    else:
        invoice_list = list(invoices)
        invoice_list = random.sample(invoice_list, k=len(invoice_list))

    static_plan = generate_mock_logistics_output(
        fixed_sku_to_station=sku_zone_matching,
        fixed_invoice_to_sku=skus_per_invoices,
        fixed_num_stations=num_zones,
        fixed_zone_capacity=zone_cap
    )

    _, __ = env.reset(static_plan=static_plan)
    terminated = False

    invoice_idx = 0
    clock = pygame.time.Clock()

    # 시뮬레이션이 종료될 때가지 루프를 실행
    while not terminated:
        # 창이 없더라도 Pygame이 OS와 통신
        pygame.event.pump()

        # 기본 행동은 -1 (대기)로 설정
        action = -1
        # 아직 처리할 송장 순서가 남아있다면, 다음 송장을 action으로 저장
        if invoice_idx < len(invoice_list):
            action = invoice_list[invoice_idx]
            invoice_idx += 1    # 다음 송장을 가리키도록 인덱스를 증가시킴

        # 환경의 시간을 한 스텝 진행시킴
        obs, reward, terminated, truncated, info = env.step(action)

        # ⚠️ CPU 과부하를 막고 타이밍 문제를 완화하기 위해 아주 짧은 지연을 줌
        # GUI 모드의 FPS 제한과 유사한 효과를 냄
        clock.tick(1000000)

    return env.simulator._get_state_snapshot()['time']

def run_gui(data, invoice_list=None, sku_zone_matching=None):
    num_skus = data["num_sku"]
    num_zones = data["num_zone"]
    num_invoices = len(data["invoices"])
    skus = range(num_skus)
    zones = range(num_zones)
    invoices = range(num_invoices)
    skus_per_invoices = data["invoices"]

    # 시뮬레이션 환경
    env = qps_env.LogisticsQPSEnv(static_optimizer=generate_mock_logistics_output,
                                  sim_params=cfg.sim_params,
                                  render_mode="human" if cfg.USE_GUI else None)

    # 데이터를 통한 Station 용량 설정
    total_skus = [sku_id for sku_list in skus_per_invoices if sku_list for sku_id in sku_list]
    zone_cap = {zone: math.ceil(len(total_skus) / num_zones) for zone in range(num_zones)}

    if not sku_zone_matching:
        sku_zone_matching = {}
        # SKU-Zone할당: 각 작업을 가능한 Zone에 무작위로 할당
        tmp_zone_cap = zone_cap.copy()
        for sku_id in skus:
            possible_zones = [zone for zone in zones if tmp_zone_cap[zone] > 0]
            if not possible_zones:
                possible_zones = list(zones)
            rnd_zone = random.choice(possible_zones)
            sku_zone_matching[sku_id] = rnd_zone
            if tmp_zone_cap.get(rnd_zone, 0) > 0:
                tmp_zone_cap[rnd_zone] -= 1

    if invoice_list:
        invoice_list = invoice_list
    else:
        invoice_list = list(invoices)
        invoice_list = random.sample(invoice_list, k=len(invoice_list))

    static_plan = generate_mock_logistics_output(
        fixed_sku_to_station=sku_zone_matching,
        fixed_invoice_to_sku=skus_per_invoices,
        fixed_num_stations=num_zones,
        fixed_zone_capacity=zone_cap
    )

    obs, info = env.reset(static_plan=static_plan)
    terminated = False
    running = True

    while running and not terminated:
        env.render()
        action_request = env.renderer.handle_events()

        if action_request == "QUIT":
            running = False
            continue

        # action 변수는 pause 상태가 아닐 때만 갱신되도록 루프 안으로 이동
        action = -1

        if env.renderer.is_paused:
            if action_request == "RESET":
                obs, info = env.reset()
            elif action_request == "STEP_FORWARD":
                if not env.simulator.is_done():
                    # Step 모드: 유효한 행동을 하나 선택하여 env.step() 호출
                    pending_mask = obs['pending_mask']
                    valid_actions = np.where(pending_mask == 1)[0]
                    if len(valid_actions) > 0:
                        action = np.random.choice(valid_actions)
                    obs, reward, terminated, truncated, info = env.step(action)
        else:
            # Play 모드: 유효한 행동을 하나 선택
            pending_mask = obs['pending_mask']
            valid_actions = np.where(pending_mask == 1)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)

    if running:
        print("Simulated finished. You can inspect the final state or press Reset.")
        return  env.simulator._get_state_snapshot()['time']

def optimize_sku_zone(invoices):
    # 1. 모든 SKU 등장 횟수 계산
    all_skus = [sku for inv in invoices for sku in inv]
    sku_counts = Counter(all_skus)

    # 2. zone 개수 설정 (기존 매칭 구조 참고)
    num_zones = 10  # 0~9
    zone_workload = [0 for _ in range(num_zones)]
    sku_zone_matching = {}

    # 3. SKU를 등장 빈도 순서로 정렬 (많이 등장하는 SKU를 먼저 할당)
    sorted_skus = sorted(sku_counts.items(), key=lambda x: -x[1])

    # 4. 최소 workload를 가진 zone에 순차적으로 배정 (greedy)
    for sku, count in sorted_skus:
        min_zone = np.argmin(zone_workload)
        sku_zone_matching[sku] = min_zone
        zone_workload[min_zone] += count

    return sku_zone_matching, zone_workload

def get_metrics(invoices, invoice_list, sku_zone_matching):
    total_stops_per_invoice = 0.0
    total_visit_to_sku_ratio = 0.0
    station_workload = [0.0 for zone in set(sku_zone_matching.values())]
    for inv in invoice_list:
        stops = []
        for sku in invoices[inv]:
            stops.append(sku_zone_matching[sku])
        total_stops_per_invoice += len(set(stops))
        total_visit_to_sku_ratio += len(set(stops)) / len(invoices[inv])
    avg_stops_per_invoice = total_stops_per_invoice / len(invoices)
    avg_visit_to_sku_ratio = total_visit_to_sku_ratio / len(invoices)

    for (sku, zone) in sku_zone_matching.items():
        for skus in invoices:
            for a_sku in skus:
                if a_sku == sku:
                    station_workload[zone] += 1

    print("Avg stops per invoice: ", avg_stops_per_invoice)
    print("Avg visit per invoice: ", avg_visit_to_sku_ratio)

    station_workload = np.array(station_workload, dtype=float)
    mean = np.mean(station_workload)
    std = np.std(station_workload, ddof=1)
    cv = (std / mean) * 100 if mean != 0 else np.nan

    print("Avg station workload: ", mean)
    print("STD station workload: ", std)
    print("CV station workload: ", cv)
    print("Station workload: ", station_workload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--data_set",
        type=str,
        required=True,
        help="Name of the dataset (e.g., A1, B1, etc.)"
    )
    parser.add_argument(
        "--gui",
        type=bool,
        required=True,
        help="Option to enable or disable GUI mode"
    )
    args = parser.parse_args()

    data_file_path = args.data_set
    with open(data_file_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    if args.gui == True:
        cfg.USE_GUI = True
    else:
        cfg.USE_GUI = False

    if not cfg.USE_GUI:
        makespan = run_no_gui(d)
    else:
        makespan = run_gui(d)

    print("Makespan: ", makespan)