# --- 문제 스케일 조정 ---
MIN_N_STATIONS = 5
MAX_N_STATIONS = 5
MIN_N_SKUS = 50
MAX_N_SKUS = 100
MIN_N_CLUSTERS = 25
MAX_N_CLUSTERS = 35
MIN_QTY_SKUS = 2
MAX_QTY_SKUS = 5

# --- 시뮬레이션 및 렌더링을 위한 상수 정의 ---
# 색상
COLOR_BACKGROUND = (30, 30, 40)
COLOR_STATION_IDLE = (100, 140, 100)
COLOR_STATION_LOW = (100, 180, 100)     # 초록
COLOR_STATION_MED = (220, 180, 80)      # 노랑
COLOR_STATION_HIGH = (220, 100, 80)     # 빨강
COLOR_QUEUE_BOX = (80, 80, 150)
COLOR_TEXT_LIGHT = (240, 240, 240)
COLOR_TEXT_DARK = (20, 20, 20)
COLOR_RED_LINE = (255, 80, 80)
COLOR_INFO_PANEL = (50, 50, 60)
COLOR_COMPLETED_BOX = (120, 120, 140)

# 화면 크기
SCREEN_WIDTH = 2000
SCREEN_HEIGHT = 1000

# 시뮬레이션 파라미터 설정
sim_params = {'queue_limit': 5,
              'travel_time': 3.0,
              'render_fps': 10}

SKU_WORKLOAD = 3    # SKU 처리 시간 (상수)
N_IMPROVEMENT_STEPS = 1

# GUI를 켜고 싶으면 True, 끄고 텍스트 모드로만 실행하려면 False로 변경
# USE_GUI는 시뮬레이션마다 GUI를 띄우면 학습이 매우 느려지므로 False로 설정해야함
USE_GUI = True

AFFINITY_REWARD_WEIGHT = 0.3
DISCOUNT_FACTOR = 0.99
