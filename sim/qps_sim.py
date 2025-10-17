import numpy as np
import heapq

from collections import deque, defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple

from numpy.ma.core import nonzero


class LogisticsQPSSimulator:
    """
    [순차 처리 모델] 물류 QPS 시스템 핵심 동적 로직을 담당하는 시뮬레이터
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.static_plan = None

        # 택배상자 정보
        self.cluster_routes = {}
        self.cluster_workloads = {}
        self.cluster_route_progress = {}
        self.cluster_injection_time = {}

        # 전체 시스템 정보
        self.completed_cluster_info = []
        self.skus_by_cluster_station = {}
        self.remaining_skus_by_station = {}
        self.last_injected_cluster_id = None
        self.bypassed_stations_per_cluster = defaultdict(set)
        self.in_transit_clusters = []

    def reset(self, static_plan: Dict[str, Any]):
        """
        정적 할당 계획을 기반으로 시뮬레이션 상태를 초기화
        """
        self.static_plan = static_plan
        self.num_stations = static_plan['num_stations']
        self.num_clusters = static_plan['num_clusters']

        self.time = 0.0
        self.event_heap = []
        self.event_counter = 0

        # 스테이션 상태
        self.station_status = [{
            "status": "idle",
            "cluster_id": -1,
            "finish_time": -1
        } for _ in range(self.num_stations)]
        self.station_total_busy_time = np.zeros(self.num_stations)
        self.queues = [deque() for _ in range(self.num_stations)]

        # 택배 상자 상태
        self.pending_cluster_ids = set(range(self.num_clusters))
        self.completed_cluster_ids = set()
        self.cluster_routes = {}
        self.cluster_workloads = {}
        self.cluster_route_progress = {}
        self.cluster_injection_time = {}
        self.completed_cluster_info = []
        self.last_injected_cluster_id = None
        self.bypassed_stations_per_cluster.clear()
        self.in_transit_clusters = []   # 이동 중인 택배 상자들
        self.remaining_skus_by_station = defaultdict(Counter)

        self.skus_by_cluster_station = static_plan['assignment_df'].groupby(['cluster_id', 'station_id'])[
            'sku_id'].apply(list).to_dict()

        for (cluster_id, station_id), sku_list in self.skus_by_cluster_station.items():
            self.remaining_skus_by_station[station_id].update(sku_list)

        assignment_df = self.static_plan['assignment_df']
        for cid in range(self.num_clusters):
            cluster_tasks = assignment_df[assignment_df['cluster_id'] == cid]
            station_loads = cluster_tasks.groupby('station_id')['load'].sum().to_dict()
            self.cluster_workloads[cid] = station_loads
            route = sorted(station_loads.keys())
            self.cluster_routes[cid] = route

        self.stats = {'bypasses': 0}
        return self._get_state_snapshot()

    def _get_state_snapshot(self):
        """
        렌더링 및 강화학습 상태 구성을 위한 현재 시스템 상태 스냅샷 반환
        """
        # 스테이션 로드: 현재 처리 중인 작업의 남은 시간 + 대기열에 쌓여있는 모든 작업들의 처리 시간 총합
        station_loads = np.array(
            [max(0, s['finish_time'] - self.time) + sum(workload for _, workload in self.queues[i]) for i, s in
             enumerate(self.station_status)]
        )

        # 상태정보로 쓰고 싶은 것에 #! 표시
        return {
            "time": self.time,  # !
            "station_status": self.station_status,  # !
            "station_loads": station_loads,  # !
            "station_total_busy_time": self.station_total_busy_time,
            "queues": [list(q) for q in self.queues],  # !
            "pending_cluster_ids": self.pending_cluster_ids,  # !
            "completed_cluster_ids": self.completed_cluster_ids,
            # ! pending_cluster_ids와 completed_cluster_ids 둘 다 처리하기보다 전체 처리해야할 택배 상자들 중 처리한 택배상자들 비율
            "stats": self.stats,
            "static_plan": self.static_plan,
            "cluster_injection_times": self.cluster_injection_time,
            "completed_cluster_info": self.completed_cluster_info,
            "cluster_routes": self.cluster_routes,
            "cluster_route_progress": self.cluster_route_progress,
            "remaining_skus_by_station": self.remaining_skus_by_station,    #!
            "last_injected_cluster_id": self.last_injected_cluster_id,
            "bypassed_info": self.bypassed_stations_per_cluster,    #!
            "in_transit_clusters": self.in_transit_clusters # 이동 중인 택배 상자들
        }

    def inject_cluster(self, cluster_id: int):
        """
        선택된 주문 클러스터를 시스템 (셩로의 첫 번째 스테이션)에 투입
        """
        if cluster_id not in self.pending_cluster_ids:
            return

        self.last_injected_cluster_id = cluster_id              # 마지막으로 들어간 택배 상자 변경
        self.pending_cluster_ids.remove(cluster_id)             # 기다리고 있는 택배 상자에서 제거
        self.cluster_injection_time[cluster_id] = self.time     # 마지막 투입 시점 변경

        route = self.cluster_routes.get(cluster_id)             # 선택한 택배 상자의 경로 조회
        if not route:                                           # 만약 더 이상 경로가 없다면
            self.completed_cluster_ids.add(cluster_id)          # 처리 완료된 택배 상자에 추가
            return

        self.cluster_route_progress[cluster_id] = 0
        self._schedule_next_station(cluster_id, source_station_id=None)

    #! 매우 중요
    def _schedule_next_station(self, cluster_id: int, source_station_id: Optional[int]) -> bool:
        """
        클러스터가 다음에 방문할 스테이션을 결정하고 이벤트를 생성 (무한 루프 방지 로직 추가)
        """
        route = self.cluster_routes.get(cluster_id, [])
        visit_round = self.cluster_route_progress.get(cluster_id, 0)
        next_station_to_visit = None

        # 1순위: 원래 경로를 먼저 모두 순회
        if visit_round < len(route):
            next_station_to_visit = route[visit_round]
            self.cluster_route_progress[cluster_id] += 1
        # 2순위: 원래 경로 순회 후, Bypass된 스테이션 방문
        else:
            bypassed_set = self.bypassed_stations_per_cluster.get(cluster_id, set())

            # --- [핵심 수정] 무한 루프 방지 로직 ---
            # 목적지 후보에서 현재 출발지를 제외합니다.
            potential_destinations = bypassed_set.copy()
            if source_station_id is not None and source_station_id in potential_destinations:
                potential_destinations.remove(source_station_id)

            if potential_destinations:
                next_station_to_visit = min(potential_destinations)
            # --- 수정 끝 ---

        # 방문할 스테이션이 더 이상 없으면 클러스터 완료 처리
        if next_station_to_visit is None:
            completion_time = self.time
            injection_time = self.cluster_injection_time.get(cluster_id, self.time)
            had_bypasses = cluster_id in self.bypassed_stations_per_cluster

            if cluster_id not in self.completed_cluster_ids:
                self.completed_cluster_ids.add(cluster_id)
                self.completed_cluster_info.append({
                    'cluster_id': cluster_id, 'completion_time': completion_time,
                    'injection_time': injection_time, 'had_bypasses': had_bypasses
                })

            if cluster_id in self.cluster_route_progress: del self.cluster_route_progress[cluster_id]
            if cluster_id in self.cluster_injection_time: del self.cluster_injection_time[cluster_id]
            if cluster_id in self.bypassed_stations_per_cluster: del self.bypassed_stations_per_cluster[cluster_id]

            return True

        # 이동 거리에 비례한 이동 시간 계산
        workload = self.cluster_workloads[cluster_id][next_station_to_visit]
        travel_duration = 0
        if source_station_id is not None:
            if next_station_to_visit < source_station_id:
                distance = (self.num_stations - source_station_id) + next_station_to_visit
                travel_duration = distance * self.params['travel_time']
            else:
                distance = next_station_to_visit - source_station_id
                travel_duration = distance * self.params['travel_time']
        else:
            travel_duration = self.params['travel_time']

        arrival_time = self.time + travel_duration

        # 도착 이벤트 생성
        event = (arrival_time, self.event_counter, 'SKU_GROUP_ARRIVAL',
                 {'cluster_id': cluster_id, 'station_id': next_station_to_visit, 'workload': workload})
        self.event_counter += 1
        heapq.heappush(self.event_heap, event)

        # 애니메이션 정보 추가 (Renderer가 있다면 사용됨)
        if source_station_id is not None:
            self.in_transit_clusters.append({
                'cluster_id': cluster_id, 'source_id': source_station_id, 'dest_id': next_station_to_visit,
                'start_time': self.time, 'arrival_time': arrival_time
            })
        return False

    def is_done(self) -> bool:
        """
        모든 클러스터가 완료되었는지 확인
        """
        return len(self.completed_cluster_ids) == self.num_clusters

    def move_one_second(self):
        """
        시뮬레이션 시간을 1초로 진행하고, 그 사이에 발생하는 모든 이벤트를 처리
        """
        target_time = self.time + 1.0   # 목표 시간
        processed_events = []

        # 현재 시간과 목표 시간 (현재 시간 + 1초) 사이에 예약된 모든 이벤트를 순차적으로 처리
        while self.event_heap and self.event_heap[0][0] <= target_time:
            event_time, _, event_type, data = heapq.heappop(self.event_heap)

            # 시뮬레이션 시간을 실제 이벤트 발생 시간으로 업데이트
            self.time = event_time

            # 이벤트 처리 로직
            if event_type == 'SKU_GROUP_ARRIVAL':
                station_id = data['station_id']
                cluster_id = data['cluster_id']
                is_bypassed_station = station_id in self.bypassed_stations_per_cluster.get(cluster_id, set())

                # Bypass 로직
                if len(self.queues[station_id]) < self.params['queue_limit']:
                    self.queues[station_id].append((cluster_id, data['workload']))
                    processed_events.append({'type': 'ARRIVED', **data})
                    if is_bypassed_station:
                        self.bypassed_stations_per_cluster[cluster_id].remove(station_id)
                else:
                    self.stats['bypasses'] += 1
                    self.bypassed_stations_per_cluster[cluster_id].add(station_id)
                    processed_events.append({'type': 'BYPASSED', **data})
                    self._schedule_next_station(cluster_id, source_station_id=station_id)

                self._try_start_task(station_id)

            # 어떤 스테이션에서 작업이 완료되었을 때
            elif event_type == 'TASK_COMPLETE':
                # 어떤 스테이션에서 어떤 택배 상자의 작업이 끝났는지 확인
                station_id = data['station_id']
                cluster_id = data['cluster_id']

                # 해당 작업으로 처리된 SKU 목록을 가져와, 스테이션의 '처리해야 할 SKU 재고'에서 삭제
                skus_processed = self.skus_by_cluster_station.get((cluster_id, station_id), [])
                self.remaining_skus_by_station[station_id].subtract(skus_processed)
                self.station_total_busy_time[station_id] += (self.time - data['start_time'])    # 이 작업을 처리하는데 걸린 시간
                self.station_status[station_id] = {'status': 'idle',
                                                   'cluster_id': -1,
                                                   'finish_time': -1}                           # 스테이션 상태를 변경 -> 새로운 작업 할당 가능
                is_cluster_finished = self._schedule_next_station(cluster_id, source_station_id=station_id) # 택배 상자의 다음 여정 계획
                processed_events.append({'type': 'COMPLETED', **data})
                if is_cluster_finished:
                    processed_events.append({'type': 'COMPLETED_CLUSTER',
                                             'cluster_id': cluster_id})
                self._try_start_task(station_id)

        # 중간 이벤트를 모두 처리한 후, 최종 시간을 목표 시간으로 설정
        self.time = target_time

        return processed_events, self.is_done()

    def _try_start_task(self, station_id: int):
        """
        유휴 상태인 스테이션이 대기열에서 작업을 시작하도록 시도
        """
        if self.station_status[station_id]['status'] == 'idle' and self.queues[station_id]:
            cluster_id, workload = self.queues[station_id].popleft()

            start_time = self.time
            finish_time = self.time + workload
            self.station_status[station_id] = {'status': 'busy',
                                               'cluster_id': cluster_id,
                                               'finish_time': finish_time}
            event = (finish_time,
                     self.event_counter,
                     'TASK_COMPLETE',
                     {'cluster_id': cluster_id,
                      'station_id': station_id,
                      'start_time': start_time})
            self.event_counter += 1
            heapq.heappush(self.event_heap, event)