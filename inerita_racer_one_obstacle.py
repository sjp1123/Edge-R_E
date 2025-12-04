import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from stable_baselines3 import PPO
import os

# ==========================================
# 1. 환경 정의 (Physics & Game Logic)
# ==========================================
class InertiaRacerEnv(gym.Env):
    """
    관성 주행 환경
    - 목표: A를 줍고, 바로 B로 향해야 함 (멈추지 않고 지나치기)
    - 상태: 내 속도, A와의 거리, B와의 거리
    - 행동: X, Y축 가속도 조절
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(InertiaRacerEnv, self).__init__()
        
        # --- 설정 상수 ---
        self.SCREEN_SIZE = 800
        self.AGENT_RADIUS = 10
        self.TARGET_RADIUS = 15
        self.MAX_SPEED = 150.0
        self.ACCEL_POWER = 1.0
        self.FRICTION = 0.92  # 공기 저항 (1.0이면 저항 없음)
        self.obstacle_radius = 30.0

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # --- Action Space: [가속도X, 가속도Y] (-1.0 ~ 1.0) ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # --- Observation Space: 정규화된 관측값 (크기 6) ---
        # [내Vx, 내Vy, A_RelX, A_RelY, B_RelX, B_RelY]
        # 값의 범위는 대략 -inf ~ inf 지만 학습을 위해 정규화된 값을 주는 게 좋음
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. 에이전트 초기화 (화면 중앙)
        self.pos = np.array([self.SCREEN_SIZE / 2, self.SCREEN_SIZE / 2], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)

        # 2. 목표물 초기화 (A와 B 생성)
        self.target_A = self._spawn_target()
        self.target_B = self._spawn_target()
        self.prev_dist_to_A = np.linalg.norm(self.pos - self.target_A)

        while True:
            self.obstacle_pos = np.random.uniform(100, self.SCREEN_SIZE-100, size=2).astype(np.float32)
            if np.linalg.norm(self.obstacle_pos - self.pos) > 100:
                break

        self.score = 0
        self.steps = 0
        self.max_steps = 1000 # 한 에피소드 최대 길이

        return self._get_obs(), {}

    def _spawn_target(self):
        # 화면 가장자리를 제외한 랜덤 위치 반환
        padding = 50
        return np.random.uniform(padding, self.SCREEN_SIZE - padding, size=2).astype(np.float32)

    def _get_obs(self):
        # 신경망에 넣어줄 데이터 (상대 좌표로 변환하여 학습 효율 증대)
        # 모든 값을 대략 -1 ~ 1 사이로 스케일링
        scale = self.SCREEN_SIZE
        vec_to_obs = self.obstacle_pos - self.pos
        dist_to_obs = np.linalg.norm(vec_to_obs) + 1e-5
        closing_speed = np.dot(self.vel, vec_to_obs / dist_to_obs) / self.MAX_SPEED
        obs = np.array([
            self.vel[0] / self.MAX_SPEED,     # 속도 X
            self.vel[1] / self.MAX_SPEED,     # 속도 Y
            (self.target_A[0] - self.pos[0]) / scale, # A와의 거리 X
            (self.target_A[1] - self.pos[1]) / scale, # A와의 거리 Y
            (self.target_B[0] - self.pos[0]) / scale, # B와의 거리 X
            (self.target_B[1] - self.pos[1]) / scale,  # B와의 거리 Y
            (self.obstacle_pos[0] - self.pos[0]) / scale, #장애물과의 거리 X
            (self.obstacle_pos[1] - self.pos[1]) / scale, #장애물과의 거리 Y
            closing_speed,
        ], dtype=np.float32)
        return obs
   
    def step(self, action):
        self.steps += 1
        
        # 1. 물리 엔진 (동일)
        accel = np.array(action, dtype=np.float32) * self.ACCEL_POWER
        self.vel += accel
        
        speed = np.linalg.norm(self.vel)
        if speed > self.MAX_SPEED:
            self.vel = (self.vel / speed) * self.MAX_SPEED
            
        self.vel *= self.FRICTION
        self.pos += self.vel

        # -----------------------------------------------------
        # 2. 보상 계산
        # -----------------------------------------------------
        reward = 0.0
        terminated = False
        
        dist_to_A = np.linalg.norm(self.pos - self.target_A)
        dist_to_obs = np.linalg.norm(self.pos - self.obstacle_pos)
        
        # [기존 유지] 거리 쉐이핑 (움직임 유도)
        reward += (self.prev_dist_to_A - dist_to_A) * 1.0 
        self.prev_dist_to_A = dist_to_A
        
        # -----------------------------------------------------
        # [신규 처방] 게으름 방지 페널티 (Anti-Freezing)
        # -----------------------------------------------------
        # 속도가 2.0 미만이면 매 프레임 -0.5점 (멈추면 죽는 것만큼 괴롭게 만듦)
        if speed < 2.0:
            reward -= 1.0
        
        # 기본 시간 페널티
        reward -= 0.01

        # -----------------------------------------------------
        # 3. 미래 예측 경고 (유지하되 살짝 수정)
        # -----------------------------------------------------
        future_pos = self.pos + self.vel * 15.0
        dist_future_to_obs = np.linalg.norm(future_pos - self.obstacle_pos)
        
        # 만약 "이 속도대로면 들이받는다"면 미리 경고(감점)
        if dist_future_to_obs < (self.AGENT_RADIUS + self.obstacle_radius):
            # 속도가 빠를수록 더 큰 공포를 느낌
            reward -= 2.0 * (speed / self.MAX_SPEED)

        # -----------------------------------------------------
        # 4. 충돌 처리 (즉사)
        # -----------------------------------------------------
        if dist_to_obs < (self.AGENT_RADIUS + self.obstacle_radius):
            reward -= 500.0  # 절대 금기
            terminated = True
            return self._get_obs(), reward, terminated, False, {}

        # 벽 충돌
        hit_wall = False
        if self.pos[0] < 0: self.pos[0] = 0; self.vel[0] *= -0.5; hit_wall = True
        if self.pos[0] > self.SCREEN_SIZE: self.pos[0] = self.SCREEN_SIZE; self.vel[0] *= -0.5; hit_wall = True
        if self.pos[1] < 0: self.pos[1] = 0; self.vel[1] *= -0.5; hit_wall = True
        if self.pos[1] > self.SCREEN_SIZE: self.pos[1] = self.SCREEN_SIZE; self.vel[1] *= -0.5; hit_wall = True
        
        if hit_wall:
            reward -= 5.0 # 벽에 붙어서 멈추는 꼼수 방지 (벽도 아프게)

        # -----------------------------------------------------
        # 5. 목표 획득
        # -----------------------------------------------------
        if dist_to_A < (self.AGENT_RADIUS + self.TARGET_RADIUS):
            reward += 25.0 # 점수 획득
            self.score += 1
            
            # 목표 교체
            self.target_A = self.target_B
            self.target_B = self._spawn_target()
            self.prev_dist_to_A = np.linalg.norm(self.pos - self.target_A)

        truncated = self.steps >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
                pygame.display.set_caption("Inertia Racer RL Visualization")
                self.clock = pygame.time.Clock()

            self.screen.fill((30, 30, 30)) # 배경

            pygame.draw.circle(self.screen, (100, 100, 100), self.obstacle_pos.astype(int), int(self.obstacle_radius))

            # --- 그리기 도우미 함수 ---
            def draw_vec(start, vec, color, scale=10):
                end = start + vec * scale
                pygame.draw.line(self.screen, color, start.astype(int), end.astype(int), 3)
            # ------------------------

            # 1. 다음 목표물 B
            pygame.draw.circle(self.screen, (100, 100, 255), self.target_B.astype(int), self.TARGET_RADIUS, 2)
            pygame.draw.line(self.screen, (100, 100, 255), self.target_A.astype(int), self.target_B.astype(int), 1)

            # 2. 현재 목표물 A
            pygame.draw.circle(self.screen, (255, 50, 50), self.target_A.astype(int), self.TARGET_RADIUS)

            # 3. 에이전트
            pygame.draw.circle(self.screen, (255, 255, 255), self.pos.astype(int), self.AGENT_RADIUS)

            # 4. 벡터 시각화
            draw_vec(self.pos, self.vel, (0, 255, 0), scale=10) # 속도 (초록)
            
            # --- UI 정보 표시 ---
            font = pygame.font.SysFont("Arial", 20)
            
            # (1) 점수 표시
            score_surf = font.render(f"Score: {self.score}", True, (255, 255, 255))
            self.screen.blit(score_surf, (10, 10))
            
            # (2) [추가됨] 현재 속력 표시
            # np.linalg.norm(self.vel)은 벡터의 길이(속력)를 구합니다.
            current_speed = np.linalg.norm(self.vel)
            
            # 속도가 빠르면(10 이상) 노란색, 아니면 하늘색으로 표시
            color = (255, 255, 0) if current_speed > 10.0 else (0, 255, 255)
            
            speed_surf = font.render(f"Speed: {current_speed:.2f}", True, color)
            self.screen.blit(speed_surf, (10, 35)) # 점수 바로 아래(y=35)에 배치
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()

# ==========================================
# 2. 메인 실행 코드 (학습 및 테스트)
# ==========================================
if __name__ == "__main__":
    MODEL_PATH = "model_prod_minv3_30.zip"
    
    # 환경 생성
    env = InertiaRacerEnv(render_mode="human") # 학습 시엔 "rgb_array" 권장하나, 보는 맛을 위해 human

    # --- A. 학습 (모델이 없으면 학습 시작) ---
    if not os.path.exists(MODEL_PATH):
        print(">>> 새로운 모델 학습을 시작합니다 (약 50,000 steps)...")
        print(">>> 학습 중에는 화면이 뜨지 않거나 검게 보일 수 있습니다.")
        
        # 시각화 없이 빠르게 학습하기 위해 더미 환경 생성
        train_env = InertiaRacerEnv(render_mode=None)
        
        # PPO 모델 생성
        model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=0.0003)
        
        # 학습 실행 (약 3~5분 소요, steps를 늘리면 더 똑똑해짐)
        model.learn(total_timesteps=400000)
        
        # 저장
        model.save(MODEL_PATH)
        print(">>> 학습 완료 및 저장됨!")
        train_env.close()

    # --- B. 테스트 및 시각화 ---
    print(">>> 학습된 모델을 불러와 시각화합니다.")
    
    # 모델 로드
    model = PPO.load(MODEL_PATH)
    
    # 테스트 루프
    obs, _ = env.reset()
    running = True
    while running:
        # AI가 행동 결정
        action, _ = model.predict(obs, deterministic=True)
        
        # 환경에 적용
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # 화면 그리기
        env.render()
        
        # Pygame 종료 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
