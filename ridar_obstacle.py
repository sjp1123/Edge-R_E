import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import math
from datetime import datetime

class InertiaRacerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(InertiaRacerEnv, self).__init__()
        
        # --- 설정 ---
        self.SCREEN_SIZE = 800
        self.AGENT_RADIUS = 10
        self.TARGET_RADIUS = 15
        self.OBSTACLE_RADIUS = 30 
        
        self.MAX_SPEED = 150.0
        self.ACCEL_POWER = 2.0  # [고정] 가속력 제한 유지
        self.FRICTION = 0.92
        
        self.NUM_OBSTACLES = 1
        
        # LIDAR
        self.RAY_NUM = 15 
        self.RAY_ANGLES = np.linspace(-90, 90, self.RAY_NUM) 
        self.BASE_RAY_LENGTH = 200.0

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        obs_size = 5 + self.RAY_NUM
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # 상태 변수
        self.active_waypoint = None
        self.last_accel = np.array([0.0, 0.0])
        self.prev_action = np.array([0.0, 0.0])
        self.avoidance_timer = 0 
        self.last_blocking_obs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.array([self.SCREEN_SIZE/2, self.SCREEN_SIZE/2], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.target = self._spawn_entity(padding=50)
        
        self.obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            self.obstacles.append(self._spawn_entity(padding=50, check_overlap=True))
        
        self.score = 0
        self.steps = 0
        self.max_steps = 1000
        self.prev_action = np.array([0.0, 0.0])
        self.avoidance_timer = 0
        self.last_blocking_obs = None
        return self._get_obs(), {}

    def _spawn_entity(self, padding=50, check_overlap=False):
        while True:
            pos = np.random.uniform(padding, self.SCREEN_SIZE - padding, size=2).astype(np.float32)
            if not check_overlap: return pos
            
            safe = True
            if np.linalg.norm(pos - self.pos) < 150: safe = False
            if np.linalg.norm(pos - self.target) < 100: safe = False
            for obs in self.obstacles:
                if np.linalg.norm(pos - obs) < 80: safe = False
            if safe: return pos

    def _get_rays(self):
        speed = np.linalg.norm(self.vel)
        heading = math.atan2(self.vel[1], self.vel[0]) if speed > 1.0 else 0.0
        
        current_ray_length = self.BASE_RAY_LENGTH + (speed * 6.0)

        ray_readings = []
        self.current_rays_rendering = [] 

        for angle_deg in self.RAY_ANGLES:
            angle_rad = heading + math.radians(angle_deg)
            ray_dir = np.array([math.cos(angle_rad), math.sin(angle_rad)])
            
            min_dist = current_ray_length
            
            for obs in self.obstacles:
                to_obs = obs - self.pos
                proj = np.dot(to_obs, ray_dir)
                if proj > 0:
                    perp_dist = np.linalg.norm(to_obs - (ray_dir * proj))
                    if perp_dist < self.OBSTACLE_RADIUS:
                        hit_dist = proj - math.sqrt(self.OBSTACLE_RADIUS**2 - perp_dist**2)
                        if 0 < hit_dist < min_dist:
                            min_dist = hit_dist
            
            reading = min_dist / current_ray_length
            ray_readings.append(reading)
            self.current_rays_rendering.append((ray_dir, min_dist, reading))
            
        return np.array(ray_readings, dtype=np.float32)

    def _get_blocking_obstacle(self):
        to_target = self.target - self.pos
        dist_target = np.linalg.norm(to_target)
        if dist_target == 0: return None
        
        dir_target = to_target / dist_target
        closest_blocking_obs = None
        min_dist_to_obs = float('inf')

        for obs in self.obstacles:
            to_obs = obs - self.pos
            proj = np.dot(to_obs, dir_target)
            
            if 0 < proj < dist_target:
                perp_dist = np.linalg.norm(to_obs - (dir_target * proj))
                safety_margin = self.OBSTACLE_RADIUS + self.AGENT_RADIUS + 40.0
                
                if perp_dist < safety_margin:
                    dist_to_obs = np.linalg.norm(to_obs)
                    if dist_to_obs < min_dist_to_obs:
                        min_dist_to_obs = dist_to_obs
                        closest_blocking_obs = obs
        return closest_blocking_obs

    def _get_detour_point(self, blocking_obs):
        # [해결책 1] 빙빙 돌기(Orbiting) 방지 로직
        to_obs = blocking_obs - self.pos
        
        # 1. 우회 반경을 크게 잡음
        detour_offset = (self.OBSTACLE_RADIUS + self.AGENT_RADIUS) * 4.0
        
        perp_vec = np.array([-to_obs[1], to_obs[0]])
        perp_vec = (perp_vec / (np.linalg.norm(perp_vec) + 1e-6)) * detour_offset
        
        # 2. [핵심] 장애물 옆(perp) + 장애물 뒤(target 방향)으로 좌표를 밈
        # 이렇게 하면 웨이포인트가 장애물보다 '앞서' 나가게 되어 에이전트를 끌어당김
        to_real_target = self.target - self.pos
        target_dir = to_real_target / (np.linalg.norm(to_real_target) + 1e-6)
        
        # "전진형 바이어스" 추가 (장애물 뒤쪽으로 60픽셀 더 미룸)
        forward_bias = target_dir * 60.0 
        
        waypoint1 = blocking_obs + perp_vec + forward_bias
        waypoint2 = blocking_obs - perp_vec + forward_bias
        
        ref_vec = self.vel if np.linalg.norm(self.vel) > 1.0 else (self.target - self.pos)
        
        if np.dot(ref_vec, waypoint1 - self.pos) > np.dot(ref_vec, waypoint2 - self.pos):
            return waypoint1
        else:
            return waypoint2

    def _get_obs(self, sensor_data=None):
        if sensor_data is None: sensor_data = self._get_rays()
        
        blocking_obs = self._get_blocking_obstacle()
        
        # 판단 고정 (Hysteresis)
        if blocking_obs is not None:
            self.last_blocking_obs = blocking_obs
            self.avoidance_timer = 5
        
        if self.avoidance_timer > 0 and self.last_blocking_obs is not None:
            self.active_waypoint = self._get_detour_point(self.last_blocking_obs)
            self.avoidance_timer -= 1
        else:
            self.active_waypoint = self.target
            self.last_blocking_obs = None

        to_objective = self.active_waypoint - self.pos
        dist_objective = np.linalg.norm(to_objective)
        scale = self.SCREEN_SIZE
        
        obs = np.concatenate([
            self.vel / self.MAX_SPEED,
            to_objective / scale,
            [dist_objective / scale],
            sensor_data
        ])
        return obs.astype(np.float32)

    def step(self, action):
        self.steps += 1
        
        accel = np.array(action) * self.ACCEL_POWER
        self.last_accel = accel
        
        self.vel += accel
        speed = np.linalg.norm(self.vel)
        if speed > self.MAX_SPEED:
            self.vel = (self.vel / speed) * self.MAX_SPEED
        self.vel *= self.FRICTION
        self.pos += self.vel
        
        reward = -0.05 
        terminated = False
        
        # --- [해결책 2] 벽 반발장 (Wall Repulsion Field) ---
        # 벽에 닿지 않았더라도 근처에 가면 페널티를 주어 중앙으로 밀어냄
        WALL_MARGIN = 60.0 # 벽 근처 60픽셀부터 위험지역
        wall_penalty = 0.0
        
        # X축 벽 감지
        if self.pos[0] < WALL_MARGIN:
            wall_penalty += (WALL_MARGIN - self.pos[0]) / WALL_MARGIN
        elif self.pos[0] > self.SCREEN_SIZE - WALL_MARGIN:
            wall_penalty += (self.pos[0] - (self.SCREEN_SIZE - WALL_MARGIN)) / WALL_MARGIN
            
        # Y축 벽 감지
        if self.pos[1] < WALL_MARGIN:
            wall_penalty += (WALL_MARGIN - self.pos[1]) / WALL_MARGIN
        elif self.pos[1] > self.SCREEN_SIZE - WALL_MARGIN:
            wall_penalty += (self.pos[1] - (self.SCREEN_SIZE - WALL_MARGIN)) / WALL_MARGIN
            
        if wall_penalty > 0:
            reward -= wall_penalty * 0.5 # 벽 근처에 있으면 지속적인 고통
            
        # 실제 벽 충돌 처리
        if self.pos[0] < 0 or self.pos[0] > self.SCREEN_SIZE or self.pos[1] < 0 or self.pos[1] > self.SCREEN_SIZE:
            self.pos = np.clip(self.pos, 0, self.SCREEN_SIZE)
            self.vel *= 0.5
            reward -= 5.0 # 충돌 시 추가 감점

        # TTC (속도 제어 유도)
        sensor_data = self._get_rays()
        min_dist_normalized = np.min(sensor_data)
        
        speed_ratio = speed / self.MAX_SPEED
        urgency = (speed_ratio ** 2) / (min_dist_normalized + 0.05)
        
        TTC_THRESHOLD = 2.5
        if urgency > TTC_THRESHOLD:
            reward -= speed_ratio * 1.5 
            reward -= (urgency - TTC_THRESHOLD) * 0.5

        # 장애물 충돌
        for obs in self.obstacles:
            dist = np.linalg.norm(self.pos - obs)
            if dist < (self.AGENT_RADIUS + self.OBSTACLE_RADIUS):
                terminated = True
                reward -= 30.0
                return self._get_obs(sensor_data), reward, terminated, False, {}

        # 방향 보상
        to_objective = self.active_waypoint - self.pos
        dist_objective = np.linalg.norm(to_objective)
        
        if speed > 5.0:
            cosine = np.dot(self.vel, to_objective) / (speed * dist_objective + 1e-8)
            reward += cosine * 0.1 
            
            if cosine > 0.8 and urgency < TTC_THRESHOLD:
                reward += speed_ratio * 0.1

        # 목표 달성
        dist_target = np.linalg.norm(self.target - self.pos)
        if dist_target < (self.AGENT_RADIUS + self.TARGET_RADIUS):
            reward += 30.0 
            self.score += 1
            self.target = self._spawn_entity(padding=50, check_overlap=True)
        
        # 행동 일관성
        action_diff = np.linalg.norm(np.array(action) - self.prev_action)
        reward -= action_diff * 0.1 
        self.prev_action = np.array(action)

        truncated = self.steps >= self.max_steps
        return self._get_obs(sensor_data), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
                self.clock = pygame.time.Clock()
                self.font = pygame.font.SysFont("Consolas", 18)
            
            self.screen.fill((20, 20, 20))
            
            # 벽 반발장 시각화 (디버깅용 - 옅은 붉은색 테두리)
            pygame.draw.rect(self.screen, (50, 0, 0), (0, 0, 60, self.SCREEN_SIZE))
            pygame.draw.rect(self.screen, (50, 0, 0), (self.SCREEN_SIZE-60, 0, 60, self.SCREEN_SIZE))
            pygame.draw.rect(self.screen, (50, 0, 0), (0, 0, self.SCREEN_SIZE, 60))
            pygame.draw.rect(self.screen, (50, 0, 0), (0, self.SCREEN_SIZE-60, self.SCREEN_SIZE, 60))
            
            if hasattr(self, 'current_rays_rendering'):
                for ray_dir, dist, reading in self.current_rays_rendering:
                    c_val = int(255 * reading)
                    c_val = max(0, min(255, c_val))
                    start = self.pos.astype(int)
                    end = (self.pos + ray_dir * dist).astype(int)
                    pygame.draw.line(self.screen, (255-c_val, c_val, 0), start, end, 1)

            for obs in self.obstacles:
                pygame.draw.circle(self.screen, (100, 100, 100), obs.astype(int), self.OBSTACLE_RADIUS)
                
            pygame.draw.circle(self.screen, (255, 50, 50), self.target.astype(int), self.TARGET_RADIUS)
            pygame.draw.circle(self.screen, (255, 255, 255), self.pos.astype(int), self.AGENT_RADIUS)
            
            if self.active_waypoint is not None and not np.array_equal(self.active_waypoint, self.target):
                 # 가상 목표 (전진형 웨이포인트) - 파란색으로 표시
                 pygame.draw.circle(self.screen, (0, 100, 255), self.active_waypoint.astype(int), 5)
                 pygame.draw.line(self.screen, (0, 100, 255), self.pos.astype(int), self.active_waypoint.astype(int), 1)

            speed = np.linalg.norm(self.vel)
            info_texts = [
                f"Score: {self.score}",
                f"Speed: {speed:.1f}",
                f"Accel: ({self.last_accel[0]:.1f}, {self.last_accel[1]:.1f})"
            ]
            
            start_y = 50 
            for i, text in enumerate(info_texts):
                ts = self.font.render(text, True, (255, 255, 255))
                self.screen.blit(ts, (10, start_y + i * 20))
            
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        if self.screen is not None: pygame.quit()

if __name__ == "__main__":
    current_time = datetime.now().strftime("%H%M%S")
    MODEL_PATH = f"exp/smart_path_last_hope_{current_time}"
    
    env = InertiaRacerEnv()
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
    print(">>> Training (Fixing Orbiting & Wall Hugging)...")
    
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        learning_rate=0.0003, 
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    )
    
    model.learn(total_timesteps=300000)
    model.save(MODEL_PATH)
    
    print(">>> Testing...")
    test_env = InertiaRacerEnv(render_mode="human")
    test_vec_env = DummyVecEnv([lambda: test_env])
    test_vec_env = VecFrameStack(test_vec_env, n_stack=4)
    
    model = PPO.load(MODEL_PATH)
    obs = test_vec_env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = test_vec_env.step(action)
        test_env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: exit()