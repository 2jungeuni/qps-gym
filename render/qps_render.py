import pygame
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Tuple

from config.qps_params import *


class LogisticsQPSRenderer:
    """
    Pygame을 사용하여 물류 시뮬레이션 상태를 GUI로 렌더링
    """
    def __init__(self, params):
        pygame.init()
        # 화면
        pygame.display.set_caption("Logistic QPS Simulator (RL Environment)")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # 폰트
        self.font_title = pygame.font.Font("./CJONLYONENEWtitleBold.ttf", 18)
        self.font = pygame.font.Font("./CJONLYONENEWtitleBold.ttf", 14)
        self.font_small = pygame.font.Font("./CJONLYONENEWtitleBold.ttf", 12)
        self.font_icon = pygame.font.Font("./CJONLYONENEWtitleBold.ttf", 18)

        self.params = params

        # 현재 상태 기록
        self.selected_station_id = None
        self.station_rects = []
        self.skus_by_station = {}
        self.selected_cluster_id = None
        self.skus_by_cluster = {}
        self.clickable_cluster_rects = []
        self.is_paused = True

        # 패널과 버튼 사이즈
        self.panel_width = 260
        self.panel_start_x = SCREEN_WIDTH - self.panel_width - 10
        button_y_pos = SCREEN_HEIGHT - 50
        self.play_button_rect = pygame.Rect(10, button_y_pos - 50, 115, 40)
        self.pause_button_rect = pygame.Rect(130, button_y_pos - 50, 115, 40)
        self.reset_button_rect = pygame.Rect(10, button_y_pos, 115, 40)
        self.step_forward_button_rect = pygame.Rect(130, button_y_pos, 120, 40)

        # 패널과 버튼 색
        self.color_button = (70, 70, 90)
        self.color_button_hover = (100, 100, 120)
        self.color_button_inactive = (40, 40, 50)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"

            # [핵심 수정] 마우스 클릭 이벤트가 발생했을 때만 아래 로직을 실행합니다.
            if event.type == pygame.MOUSEBUTTONDOWN:
                # 1. Play/Pause 버튼 클릭 확인
                if self.play_button_rect.collidepoint(event.pos):
                    self.is_paused = False
                    continue
                elif self.pause_button_rect.collidepoint(event.pos):
                    self.is_paused = True
                    continue

                # 2. (일시정지 상태에서) Reset/Step 버튼 클릭 확인
                if self.is_paused:
                    if self.reset_button_rect.collidepoint(event.pos):
                        return "RESET"
                    elif self.step_forward_button_rect.collidepoint(event.pos):
                        return "STEP_FORWARD"

                # --- [수정] 아래의 클릭 확인 로직들을 MOUSEBUTTONDOWN 블록 안으로 이동 ---
                clicked_on_ui = False

                # 3. 클러스터(Invoice) 상자 클릭 확인
                for rect, cluster_id in self.clickable_cluster_rects:
                    if rect.collidepoint(event.pos):
                        self.selected_cluster_id = cluster_id if self.selected_cluster_id != cluster_id else None
                        self.selected_station_id = None
                        clicked_on_ui = True
                        break
                if clicked_on_ui: continue

                # 4. 스테이션 클릭 확인
                for i, rect in enumerate(self.station_rects):
                    if rect.collidepoint(event.pos):
                        self.selected_station_id = i if self.selected_station_id != i else None
                        self.selected_cluster_id = None
                        clicked_on_ui = True
                        break
                if clicked_on_ui: continue

                # 5. UI 요소가 아닌 배경 클릭 시 선택 해제
                self.selected_station_id = None
                self.selected_cluster_id = None

        return None

    def render(self, state: Dict[str, Any]):
        # 최초 1회 데이터 준비
        # self.skus_by_station은 최초에 비어있음
        # 시뮬레이션 진행되는 동안 변하지 않는 정보들(어떤 스테이션에 어떤 SKU가 할당되었는가)을 매 프레임마다 계산하는 것은 비효율적
        # 최초 렌더링 시에만 필요한 데이터를 미리 계산하여 변수에 저장해두고, 다음부터는 이 저장된 값을 재사용
        if not self.skus_by_station and state['static_plan']:
            assignment_df = state['static_plan']['assignment_df']
            self.skus_by_station = assignment_df.groupby('station_id')['sku_id'].apply(list).to_dict()
            self.skus_by_cluster = assignment_df.groupby('cluster_id')['sku_id'].apply(list).to_dict()
            self.skus_by_cluster_station = assignment_df.groupby(['cluster_id', 'station_id'])['sku_id'].apply(list).to_dict()

        self.clickable_cluster_rects = []
        self.screen.fill(COLOR_BACKGROUND)

        self._draw_stations_and_queues(state)
        self._draw_info_panel(state)
        self._draw_buttons()
        self._draw_completed_clusters(state)

        pygame.display.flip()
        self.clock.tick(self.params.get("render_fps", 30))

    # --- 그리기 함수들 ---
    def _draw_stations_and_queues(self, state: Dict[str, Any]):
        num_stations = state['static_plan']['num_stations']
        station_capacities = state['static_plan']['station_stats_df']['capacity'].values
        max_load = state['static_plan']['station_stats_df']['total_load'].max() or 1

        self.station_rects = []
        base_height, height_per_capacity = 40, 10
        max_cap = max(station_capacities) if len(station_capacities) > 0 else 0
        for i in range(num_stations):
            capacity = station_capacities[i]
            rect = pygame.Rect(280 + i * 100,
                               base_height + ((max_cap - capacity) * height_per_capacity),
                               90,
                               base_height + (capacity * height_per_capacity))
            self.station_rects.append(rect)

        for i, rect in enumerate(self.station_rects):
            status = state['station_status'][i]
            load_ratio = state['station_loads'][i] / max_load

            if status['status'] == 'idle':
                color = COLOR_STATION_IDLE
            elif load_ratio < 0.5:
                color = COLOR_STATION_LOW
            elif load_ratio < 0.9:
                color = COLOR_STATION_MED
            else:
                color = COLOR_STATION_HIGH
            pygame.draw.rect(self.screen, color, rect, border_radius=8)

            if self.selected_station_id == i:
                pygame.draw.rect(self.screen, (255, 255, 0), rect, 3, border_radius=8)

            zone_cd = state['static_plan']['station_stats_df']['zone_cd'][i]
            station_text = self.font.render(f"Zone {zone_cd}", True, COLOR_TEXT_DARK)
            self.screen.blit(station_text, station_text.get_rect(centerx=rect.centerx, y=rect.y + 10))

            cap_text = self.font_small.render(f"Cap: {station_capacities[i]}", True, COLOR_TEXT_DARK)
            self.screen.blit(cap_text, cap_text.get_rect(centerx=rect.centerx, y=rect.y + 35))

            if status['status'] == 'busy':
                cluster_text = self.font_small.render(f"INV:{status['cluster_id']}", True, COLOR_TEXT_LIGHT)
                pygame.draw.rect(self.screen, (60, 60, 60), (rect.x, rect.bottom - 18, rect.width, 18),
                                 border_bottom_left_radius=8, border_bottom_right_radius=8)
                self.screen.blit(cluster_text, (rect.x + 5, rect.bottom - 17))

            queue = state['queues'][i]
            q_box_height, q_box_spacing = 50, 55
            for j, (cluster_id, workload) in enumerate(queue):
                q_rect = pygame.Rect(rect.x + 5, rect.bottom + 15 + j * q_box_spacing, 80, q_box_height)
                self.clickable_cluster_rects.append((q_rect, cluster_id))
                box_color = COLOR_QUEUE_BOX
                if cluster_id in state['bypassed_info'] and state['bypassed_info'][cluster_id]:
                    box_color = (150, 80, 150)
                if self.selected_cluster_id == cluster_id:
                    box_color = (220, 170, 0)
                pygame.draw.rect(self.screen, box_color, q_rect, border_radius=5)

                injection_time = state['cluster_injection_times'].get(cluster_id, state['time'])
                elapsed_time = state['time'] - injection_time
                q_text1 = self.font_small.render(f"INV: {cluster_id}", True, COLOR_TEXT_LIGHT)
                q_text2 = self.font_small.render(f"L: {workload:.0f}", True, COLOR_TEXT_LIGHT)
                q_text3 = self.font_small.render(f"T: {elapsed_time:.0f}s", True, (255, 255, 100))
                self.screen.blit(q_text1, (q_rect.x + 5, q_rect.y + 3))
                self.screen.blit(q_text2, (q_rect.x + 5, q_rect.y + 18))
                self.screen.blit(q_text3, (q_rect.x + 5, q_rect.y + 33))

            limit_y = rect.bottom + 15 + self.params['queue_limit'] * q_box_spacing
            pygame.draw.line(self.screen, COLOR_RED_LINE, (rect.left, limit_y), (rect.right, limit_y), 2)

    def _draw_single_panel(self,
                           icon: str,
                           title: str,
                           lines: List[Tuple[str, str]],
                           pos: Tuple[int, int],
                           width: int) -> pygame.Rect:
        """
        그려진 패널의 Rect를 반환
        """
        # --- 폰트 준비 ---
        title_font = self.font_title
        body_font = self.font
        padding = 15

        # --- 패널 높이 동적 계산 ---
        line_height_title = title_font.get_height()
        line_height_body = body_font.get_height() + 5
        panel_height = (padding * 2) + line_height_title + 10 + (len(lines) * line_height_body)
        panel_rect = pygame.Rect(pos[0], pos[1], width, panel_height)

        # --- 테두리/그림자 그리기 ---
        shadow_rect = panel_rect.copy()
        shadow_rect.move_ip(3, 3)
        pygame.draw.rect(self.screen, (20, 20, 25), shadow_rect, border_radius=10)

        # --- 기본 패널 그리기 ---
        pygame.draw.rect(self.screen, COLOR_INFO_PANEL, panel_rect, border_radius=10)

        # --- 제목 그리기 ---
        # icon_surf = self.font.render(icon, True, COLOR_TEXT_LIGHT)
        # icon_pos = (panel_rect.x + padding, panel_rect.y + padding)
        # self.screen.blit(icon_surf, icon_pos)

        title_surf = title_font.render(title, True, COLOR_TEXT_LIGHT)
        # title_pos = (icon_pos[0] + icon_surf.get_width() + 10, panel_rect.y + padding)
        title_pos = (panel_rect.x + padding + 5, panel_rect.y + padding)
        self.screen.blit(title_surf, title_pos)

        # --- 구분선 그리기 ---
        line_y = title_pos[1] + line_height_title + 5
        pygame.draw.line(self.screen, (80, 80, 90),
                         (panel_rect.left + padding, line_y),
                         (panel_rect.right - padding, line_y), 1)

        # --- 내용 그리기 ---
        current_y = line_y + 10
        max_label_width = 0
        if lines:
            max_label_width = max(body_font.size(line[0])[0] for line in lines) if any(lines) else 0

        for i, (label, value) in enumerate(lines):
            label_surf = body_font.render(label, True, (180, 180, 190))
            label_pos = (panel_rect.x + padding, current_y + i * line_height_body)
            self.screen.blit(label_surf, label_pos)

            value_surf = body_font.render(value, True, (255, 255, 255))
            value_pos = (panel_rect.x + padding + max_label_width + 10, current_y + i * line_height_body)
            self.screen.blit(value_surf, value_pos)

        return panel_rect

    def _draw_summary_panels(self, state: Dict[str, Any]) -> int:
        """전체 시뮬레이션 요약 정보가 담긴 여러 패널을 그리기"""
        panel_width = self.panel_width
        current_y = 10
        panel_gap = 10

        # --- 데이터 준비 ---
        time_lines = [
            ("Time:", f"{state['time']:.2f} s"),
            ("Makespan:", f"{state['time']:.2f} s")
        ]

        last_action_id = state.get("last_injected_cluster_id")
        action_lines = [
            ("Injected INV:", f"{last_action_id}")
        ] if last_action_id is not None else [("Last Action:", "(None yet)")]

        cluster_lines = [
            ("Pending:", f"{len(state['pending_cluster_ids'])}"),
            ("Completed:", f"{len(state['completed_cluster_ids'])} / {state['static_plan']['num_clusters']}")
        ]

        lb_lines = []
        busy_times = state['station_total_busy_time']
        if np.sum(busy_times) > 0:
            mean_busy, std_busy = np.mean(busy_times), np.std(busy_times)
            cv_busy = std_busy / mean_busy if mean_busy > 0 else 0
            lb_lines = [
                ("Mean:", f"{mean_busy:.2f}"),
                ("Std Dev:", f"{std_busy:.2f}"),
                ("CV:", f"{cv_busy:.3f}")
            ]
        else:
            lb_lines = [("Status:", "(Not started)")]

        # --- [수정] 패널 그리기 호출 시 아이콘 추가 ---
        rect1 = self._draw_single_panel("•", "Simulation Time", time_lines, (self.panel_start_x, current_y),
                                        panel_width)
        current_y += rect1.height + panel_gap

        rect_action = self._draw_single_panel("•", "Last Action", action_lines, (self.panel_start_x, current_y),
                                              panel_width)
        current_y += rect_action.height + panel_gap

        rect2 = self._draw_single_panel("•", "Invoice Status", cluster_lines, (self.panel_start_x, current_y),
                                        panel_width)
        current_y += rect2.height + panel_gap

        rect4 = self._draw_single_panel("•", "Load Balancing", lb_lines, (self.panel_start_x, current_y), panel_width)
        current_y += rect4.height + panel_gap
        # --- 수정 끝 ---

        return current_y

    def _draw_cluster_detail_panel(self, state: Dict[str, Any], cluster_id: int) -> int:
        """선택된 클러스터의 상세 정보를 더 예쁘고 유용하게 그리기"""
        # --- 기본 설정 ---
        panel_x, panel_y = self.panel_start_x, 10
        panel_width = self.panel_width
        padding = 15

        # --- 데이터 준비 ---
        route = state['cluster_routes'].get(cluster_id, [])
        injection_time = state['cluster_injection_times'].get(cluster_id)

        # 1. 클러스터의 현재 상태와 위치 파악
        status_text = "Unknown"
        location_text = "-"
        is_completed = any(c['cluster_id'] == cluster_id for c in state['completed_cluster_info'])

        if cluster_id in state['pending_cluster_ids']:
            status_text = "Pending"
            location_text = "Waiting for Injection"
        elif is_completed:
            status_text = "Completed"
            completion_info = next(c for c in state['completed_cluster_info'] if c['cluster_id'] == cluster_id)
            lead_time = completion_info['completion_time'] - completion_info['injection_time']
            location_text = f"Finished in {lead_time:.1f}s"
        else:  # In Progress
            # 작업 중인지 확인
            for i, station_stat in enumerate(state['station_status']):
                if station_stat['cluster_id'] == cluster_id:
                    status_text = "Processing"
                    location_text = f"at Station {i + 1}"
                    break
            # 대기열에 있는지 확인
            if status_text == "Unknown":
                for i, queue in enumerate(state['queues']):
                    if any(c_id == cluster_id for c_id, _ in queue):
                        status_text = "In Queue"
                        location_text = f"at Station {i + 1}"
                        break
            # 이동 중인지 확인
            if status_text == "Unknown":
                for transit_info in state['in_transit_clusters']:
                    if transit_info['cluster_id'] == cluster_id:
                        status_text = "In Transit"
                        location_text = f"to Station {transit_info['dest_id'] + 1}"
                        break

        # 경과 시간 계산
        elapsed_time_str = "-"
        if injection_time is not None and not is_completed:
            elapsed_time = state['time'] - injection_time
            elapsed_time_str = f"{elapsed_time:.1f} s"

        # 2. 요약 정보 텍스트 준비
        summary_lines = [
            ("Status:", status_text),
            ("Location:", location_text),
            ("Elapsed Time:", elapsed_time_str)
        ]

        # --- 패널 크기 계산 (내용이 많아져 동적으로 계산) ---
        # 실제 그리기보다 크기 계산을 먼저 수행
        title_height = self.font_title.get_height()
        line_height = self.font.get_height() + 5
        small_line_height = self.font_small.get_height()

        # 예상 높이 계산: 기본 여백 + 제목 + 구분선 + 요약 정보 + 구분선 + 경로 제목 + 경로 시각화 + 구분선 + 작업 목록...
        content_height = title_height + 10  # 제목 + 구분선
        content_height += len(summary_lines) * line_height + 10  # 요약 정보 + 구분선
        content_height += line_height + 40 + 10  # 경로 제목 + 경로 시각화 영역 + 구분선
        content_height += line_height  # 작업 목록 제목
        for station_id in route:  # 작업 목록
            skus_at_station = self.skus_by_cluster_station.get((cluster_id, station_id), [])
            content_height += line_height + (len(skus_at_station) * small_line_height)

        panel_height = min(padding * 2 + content_height, SCREEN_HEIGHT - 20)
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)

        # --- 패널 그리기 시작 ---
        # 그림자 및 배경
        # shadow_rect = panel_rect.copy();
        # shadow_rect.move_ip(3, 3)
        # pygame.draw.rect(self.screen, (20, 20, 25), shadow_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_INFO_PANEL, panel_rect, border_radius=10)

        current_y = panel_y + padding

        # 아이콘 + 제목
        # icon_surf = self.font_title.render("🧾", True, COLOR_TEXT_LIGHT)
        # self.screen.blit(icon_surf, (panel_x + padding, current_y))
        title_surf = self.font_title.render(f"Details for Invoice {cluster_id}", True, COLOR_TEXT_LIGHT)
        self.screen.blit(title_surf, (panel_x + padding + 5, current_y))
        current_y += title_height + 5

        # 구분선
        pygame.draw.line(self.screen, (80, 80, 90), (panel_rect.left + padding, current_y),
                         (panel_rect.right - padding, current_y), 1)
        current_y += 10

        # [개선 1] 핵심 정보 요약 그리기
        max_label_width = max(self.font.size(label)[0] for label, _ in summary_lines)
        for label, value in summary_lines:
            label_surf = self.font.render(label, True, (180, 180, 190))
            self.screen.blit(label_surf, (panel_x + padding, current_y))
            value_color = (255, 255, 100) if status_text in ["In Queue", "In Transit", "Processing"] else (255, 255,
                                                                                                           255)
            value_surf = self.font.render(value, True, value_color)
            self.screen.blit(value_surf, (panel_x + padding + max_label_width + 10, current_y))
            current_y += line_height

        # 구분선
        pygame.draw.line(self.screen, (80, 80, 90), (panel_rect.left + padding, current_y),
                         (panel_rect.right - padding, current_y), 1)
        current_y += 10

        # [개선 2] 경로 진행률 시각화
        route_title_surf = self.font.render("Route Progress:", True, COLOR_TEXT_LIGHT)
        self.screen.blit(route_title_surf, (panel_x + padding, current_y))
        current_y += line_height

        if route:
            progress = state['cluster_route_progress'].get(cluster_id, 0)

            node_radius = 8
            node_y = current_y + 15
            start_x = panel_x + padding + node_radius
            end_x = panel_rect.right - padding - node_radius

            # 경로 선 그리기
            pygame.draw.line(self.screen, (80, 80, 90), (start_x, node_y), (end_x, node_y), 2)

            for i, station_id in enumerate(route):
                node_x = start_x + (end_x - start_x) * (i / (len(route) - 1)) if len(route) > 1 else start_x

                # 상태에 따라 노드 색 결정
                if i < progress and not (station_id in state['bypassed_info'].get(cluster_id, set())):
                    node_color = COLOR_STATION_LOW  # 완료
                elif location_text.endswith(str(station_id + 1)):
                    node_color = COLOR_STATION_MED  # 현재 위치
                else:
                    node_color = (90, 90, 110)  # 예정

                pygame.draw.circle(self.screen, node_color, (int(node_x), node_y), node_radius)
                num_surf = self.font_small.render(str(station_id + 1), True, COLOR_TEXT_DARK)
                self.screen.blit(num_surf, num_surf.get_rect(center=(int(node_x), node_y)))

        current_y += 40  # 시각화 영역 높이만큼 y좌표 이동

        # 구분선
        pygame.draw.line(self.screen, (80, 80, 90), (panel_rect.left + padding, current_y),
                         (panel_rect.right - padding, current_y), 1)
        current_y += 10

        # 작업 목록 (기존과 유사)
        task_title_surf = self.font.render("Tasks by Station:", True, COLOR_TEXT_LIGHT)
        self.screen.blit(task_title_surf, (panel_x + padding, current_y))
        current_y += line_height

        for station_id in route:
            if current_y > panel_rect.bottom - padding - 20: break  # 공간 없으면 그만 그리기

            station_header_surf = self.font.render(f"→ Station {station_id + 1}:", True, COLOR_TEXT_LIGHT)
            self.screen.blit(station_header_surf, (panel_x + padding, current_y))
            current_y += line_height

            skus_at_station = self.skus_by_cluster_station.get((cluster_id, station_id), [])
            for sku in skus_at_station:
                if current_y > panel_rect.bottom - padding - 15: break
                sku_line = f"  - SKU: {sku}"
                sku_surf = self.font_small.render(sku_line, True, (200, 200, 200))
                self.screen.blit(sku_surf, (panel_x + padding + 10, current_y))
                current_y += small_line_height

        return panel_rect.bottom + 10

    def _draw_completed_clusters(self, state: Dict[str, Any]):
        """완료된 클러스터들을 더 예쁘고 정보 중심으로 그리기"""
        completed_clusters = state['completed_cluster_info']
        if not completed_clusters: return

        # --- [개선 3] 완료 시간 순으로 정렬 ---
        # 타임라인처럼 보이도록 완료된 순서대로 정렬합니다.
        completed_clusters.sort(key=lambda c: c['completion_time'])

        # --- [개선 2] 리드타임에 따른 색상 계산을 위한 준비 ---
        # 현재까지 완료된 모든 클러스터의 평균 리드타임을 계산합니다.
        lead_times = [c['completion_time'] - c['injection_time'] for c in completed_clusters]
        avg_lead_time = np.mean(lead_times) if lead_times else 0

        # --- 제목 및 레이아웃 설정 ---
        title_text = self.font.render("Completed Invoices (Sorted by Completion Time)", True, COLOR_TEXT_LIGHT)
        start_x = 280
        start_y = SCREEN_HEIGHT - 120  # 공간을 조금 더 확보
        self.screen.blit(title_text, (start_x, start_y - 25))
        pygame.draw.line(self.screen, COLOR_TEXT_LIGHT, (start_x, start_y - 5), (self.panel_start_x - 20, start_y - 5),
                         1)

        box_width, box_height = 80, 50  # 상자 크기 조정
        gap = 8
        boxes_per_row = (self.panel_start_x - start_x - 20) // (box_width + gap)
        if boxes_per_row == 0: boxes_per_row = 1

        for i, cluster_info in enumerate(completed_clusters):
            row = i // boxes_per_row
            col = i % boxes_per_row
            box_x = start_x + col * (box_width + gap)
            box_y = start_y + row * (box_height + gap)
            rect = pygame.Rect(box_x, box_y, box_width, box_height)

            # 화면에 너무 많이 그려지는 것을 방지
            if rect.top > SCREEN_HEIGHT: continue

            cluster_id = cluster_info['cluster_id']
            self.clickable_cluster_rects.append((rect, cluster_id))

            lead_time = cluster_info['completion_time'] - cluster_info['injection_time']

            # --- [개선 1 & 2] 그림자 및 리드타임 기반 색상 적용 ---
            # 그림자 그리기
            # shadow_rect = rect.copy();
            # shadow_rect.move_ip(3, 3)
            # pygame.draw.rect(self.screen, (20, 20, 25), shadow_rect, border_radius=8)

            # 리드타임에 따라 기본 색상 결정
            if avg_lead_time > 0:
                if lead_time < avg_lead_time * 0.8:  # 평균보다 20% 이상 빠르면
                    box_color = (60, 150, 90)  # Greenish
                elif lead_time > avg_lead_time * 1.2:  # 평균보다 20% 이상 느리면
                    box_color = (160, 80, 80)  # Reddish
                else:
                    box_color = COLOR_COMPLETED_BOX  # Normal
            else:
                box_color = COLOR_COMPLETED_BOX

            # 선택 시 노란색으로 덮어쓰기
            if self.selected_cluster_id == cluster_id:
                box_color = (220, 170, 0)

            pygame.draw.rect(self.screen, box_color, rect, border_radius=8)

            # 카드 헤더 그리기
            header_height = 20
            header_rect = pygame.Rect(rect.left, rect.top, rect.width, header_height)
            pygame.draw.rect(self.screen, (0, 0, 0, 50), header_rect, border_top_left_radius=8,
                             border_top_right_radius=8)

            # 텍스트 그리기 (위치 조정)
            text1 = self.font_small.render(f"INV: {cluster_info['cluster_id']}", True, COLOR_TEXT_LIGHT)
            self.screen.blit(text1, (rect.x + 6, rect.y + 4))

            text2 = self.font.render(f"{lead_time:.1f}s", True, COLOR_TEXT_LIGHT)
            text2_rect = text2.get_rect(centerx=rect.centerx, y=rect.y + 26)
            self.screen.blit(text2, text2_rect)

            # Bypass 아이콘
            if cluster_info.get('had_bypasses', False):
                icon_surf = self.font.render("!", True, COLOR_RED_LINE)
                icon_rect = icon_surf.get_rect(centery=header_rect.centery, right=header_rect.right - 8)
                self.screen.blit(icon_surf, icon_rect)

    def _draw_station_detail_panel(self, state: Dict[str, Any], station_id: int) -> int:
        """선택된 스테이션의 상세 정보를 더 예쁘고 유용하게 그리기"""
        # --- 1. 데이터 준비 ---
        stats_df = state['static_plan']['station_stats_df']
        station_info = stats_df.loc[stats_df['station_id'] == station_id].iloc[0]

        # 실시간 데이터
        status_info = state['station_status'][station_id]
        queue_len = len(state['queues'][station_id])
        queue_limit = self.params['queue_limit']
        current_load = state['station_loads'][station_id]

        # 재고 데이터
        remaining_sku_counts = state['remaining_skus_by_station'].get(station_id, Counter())
        initial_sku_counts = Counter(self.skus_by_station.get(station_id, []))
        sorted_initial_skus = sorted(initial_sku_counts.items())

        # --- 2. 표시할 텍스트 라인 구성 ---
        # [개선 2] 실시간 상태 정보
        status_text = status_info['status'].capitalize()
        status_color = (100, 200, 100) if status_text == 'Idle' else (220, 180, 80)
        status_lines = [
            ("Status:", (status_text, status_color)),
            ("Queue:", (f"{queue_len} / {queue_limit}", COLOR_TEXT_LIGHT)),
            ("Load:", (f"{current_load:.0f}", COLOR_TEXT_LIGHT))
        ]
        # 초기 할당 정보
        stat_lines = [
            ("Assigned SKUs:", f"{station_info['capacity']}"),
            ("Total Initial Load:", f"{station_info['total_load']:.0f}")
        ]

        # --- 3. 패널 크기 및 기본 구조 그리기 ---
        panel_x, panel_y = self.panel_start_x, 10
        panel_width, padding = self.panel_width, 15

        # 동적 높이 계산
        title_height = self.font_title.get_height()
        line_height = self.font.get_height() + 5
        small_line_height = self.font_small.get_height() + 8  # 프로그레스 바 높이 고려

        content_height = title_height + 10  # 제목
        content_height += line_height + 10  # 실시간 정보 제목
        content_height += len(status_lines) * line_height
        content_height += line_height + 10  # 초기 정보 제목
        content_height += len(stat_lines) * line_height
        content_height += line_height + 10  # SKU 목록 제목
        content_height += len(sorted_initial_skus) * small_line_height

        panel_height = min(padding * 2 + content_height, SCREEN_HEIGHT - 20)
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)

        # 그림자 및 배경
        # shadow_rect = panel_rect.copy();
        # shadow_rect.move_ip(3, 3)
        # pygame.draw.rect(self.screen, (20, 20, 25), shadow_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_INFO_PANEL, panel_rect, border_radius=10)

        # --- 4. 내용 그리기 ---
        current_y = panel_y + padding

        # 아이콘 + 제목
        # icon_surf = self.font_title.render("⚙️", True, COLOR_TEXT_LIGHT)
        # self.screen.blit(icon_surf, (panel_x + padding, current_y))
        title_surf = self.font_title.render(f"Zone {station_info['zone_cd']} Details", True, COLOR_TEXT_LIGHT)
        self.screen.blit(title_surf, (panel_x + padding + 5, current_y))
        current_y += title_height + 5

        # 구분선
        pygame.draw.line(self.screen, (80, 80, 90), (panel_rect.left + padding, current_y),
                         (panel_rect.right - padding, current_y), 1)
        current_y += 10

        # 실시간 상태 정보 그리기
        sub_title_surf = self.font.render("Current Status:", True, COLOR_TEXT_LIGHT)
        self.screen.blit(sub_title_surf, (panel_x + padding, current_y))
        current_y += line_height

        max_label_width = max(self.font.size(label)[0] for label, _ in status_lines)
        for label, (value, color) in status_lines:
            label_surf = self.font.render(label, True, (180, 180, 190))
            self.screen.blit(label_surf, (panel_x + padding, current_y))
            value_surf = self.font.render(value, True, color)
            self.screen.blit(value_surf, (panel_x + padding + max_label_width + 10, current_y))
            current_y += line_height

        # 초기 할당 정보 그리기 (생략 가능, 필요시 주석 해제)

        # [개선 3] SKU 진행률 시각화
        current_y += 10
        pygame.draw.line(self.screen, (80, 80, 90), (panel_rect.left + padding, current_y),
                         (panel_rect.right - padding, current_y), 1)
        current_y += 10

        sku_title_surf = self.font.render("SKU Progress:", True, COLOR_TEXT_LIGHT)
        self.screen.blit(sku_title_surf, (panel_x + padding, current_y))
        current_y += line_height

        for sku, initial_count in sorted_initial_skus:
            if current_y > panel_rect.bottom - padding - 15: break  # 공간 없으면 중단

            remaining_count = remaining_sku_counts.get(sku, 0)

            # SKU 텍스트
            sku_text = f"{sku}: {remaining_count}/{initial_count}"
            text_color = (150, 150, 150) if remaining_count <= 0 else COLOR_TEXT_LIGHT
            text_surf = self.font_small.render(sku_text, True, text_color)
            self.screen.blit(text_surf, (panel_x + padding, current_y))

            # 프로그레스 바
            progress = (initial_count - remaining_count) / initial_count if initial_count > 0 else 0
            bar_width = panel_width - (padding * 2)
            bar_x = panel_x + padding
            bar_y = current_y + self.font_small.get_height() - 2

            # 바 배경
            pygame.draw.rect(self.screen, (70, 70, 80), (bar_x, bar_y, bar_width, 4), border_radius=2)
            # 바 내용
            if progress > 0:
                pygame.draw.rect(self.screen, COLOR_STATION_LOW, (bar_x, bar_y, bar_width * progress, 4),
                                 border_radius=2)

            current_y += small_line_height

        return panel_rect.bottom + 10

    def _draw_single_button(self, rect: pygame.Rect, text: str, is_active: bool):
        """그림자가 포함된 버튼 하나를 그리는 헬퍼 함수"""
        mouse_pos = pygame.mouse.get_pos()
        is_hover = rect.collidepoint(mouse_pos)

        # [개선 2] 그림자 그리기
        # shadow_rect = rect.copy()
        # shadow_rect.move_ip(2, 2)
        # pygame.draw.rect(self.screen, (20, 20, 25), shadow_rect, border_radius=8)

        # 버튼 색상 결정
        if not is_active:
            btn_color = self.color_button_inactive
            text_color = (100, 100, 100)  # [개선 3] 비활성 시 텍스트 색
        elif is_hover:
            btn_color = self.color_button_hover
            text_color = COLOR_TEXT_LIGHT
        else:
            btn_color = self.color_button
            text_color = COLOR_TEXT_LIGHT

        # 버튼 배경 그리기
        pygame.draw.rect(self.screen, btn_color, rect, border_radius=8)

        # 버튼 텍스트 그리기
        text_surf = self.font_title.render(text, True, text_color)
        self.screen.blit(text_surf, text_surf.get_rect(center=rect.center))

    def _draw_buttons(self):
        """UI 컨트롤 버튼과 PAUSED 화면 그리기"""
        is_paused = self.is_paused

        # [개선 1] 버튼들을 담을 컨트롤 패널 배경 그리기
        # control_panel_rect = pygame.Rect(5, SCREEN_HEIGHT - 100, 255, 95)
        # pygame.draw.rect(self.screen, (0, 0, 0, 100), control_panel_rect, border_radius=12)

        # 새로 만든 헬퍼 함수를 사용하여 버튼들을 간결하게 그리기
        self._draw_single_button(self.play_button_rect, "▶ Play", True)
        self._draw_single_button(self.pause_button_rect, "❚❚ Pause", True)
        self._draw_single_button(self.reset_button_rect, "<< Reset", is_paused)
        self._draw_single_button(self.step_forward_button_rect, "> Step", is_paused)

        # [개선 4] PAUSED 화면 개선
        if self.is_paused:
            # 반투명 오버레이
            pause_overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            pause_overlay.fill((0, 0, 0, 120))
            self.screen.blit(pause_overlay, (0, 0))

            # PAUSED 텍스트와 그 배경
            paused_text_surf = self.font_title.render("PAUSED", True, COLOR_RED_LINE)
            text_rect = paused_text_surf.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))

            # 텍스트 배경 그리기
            bg_rect = text_rect.inflate(40, 20)  # 텍스트보다 가로 40, 세로 20 더 크게
            pygame.draw.rect(self.screen, (20, 20, 25, 230), bg_rect, border_radius=10)

            self.screen.blit(paused_text_surf, text_rect)

    def _draw_info_panel(self, state: Dict[str, Any]) -> int:
        if self.selected_station_id is not None:
            return self._draw_station_detail_panel(state, self.selected_station_id)
        elif self.selected_cluster_id is not None:
            return self._draw_cluster_detail_panel(state, self.selected_cluster_id)
        else:
            self._draw_summary_panels(state)
            return SCREEN_HEIGHT

    # -------------------
    def close(self):
        pygame.quit()