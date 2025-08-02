import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import math 

class FarmAIEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None):
        super(FarmAIEnv, self).__init__()
        
        # State dimensions
        self.soil_moisture = 0.5
        self.soil_nutrients = 0.5
        self.crop_health = 0.5
        self.water_availability = 0.7
        self.pest_level = 0.1
        self.weather = 0  # 0: normal, 1: drought, 2: heavy rain
        self.day = 0
        self.last_action = None  # Track last action taken
        
        # Action space: [irrigate, fertilize, pesticide, plant, harvest, nothing]
        self.action_space = spaces.Discrete(6)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]), 
            high=np.array([1, 1, 1, 1, 1, 2, 365]),
            dtype=np.float32
        )
        
        # Rendering setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.crop_sprites = []
        self.load_sprites()
        self.font = None

    def load_sprites(self):
        self.crop_sprites = [
            pygame.Surface((20, 20)),
            pygame.Surface((30, 30)),
            pygame.Surface((40, 40))
        ]
        colors = [(100, 200, 100), (50, 180, 50), (30, 150, 30)]
        for i, sprite in enumerate(self.crop_sprites):
            sprite.fill(colors[i])
    
    def _get_obs(self):
        return np.array([
            self.soil_moisture,
            self.soil_nutrients,
            self.crop_health,
            self.water_availability,
            self.pest_level,
            self.weather,
            self.day
        ], dtype=np.float32)
    
    def _get_info(self):
        return {
            "yield_potential": self.crop_health * self.soil_nutrients,
            "sustainability": self.water_availability - (self.pest_level * 0.5),
            "last_action": self.last_action
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset state
        self.soil_moisture = random.uniform(0.3, 0.5)
        self.soil_nutrients = random.uniform(0.3, 0.5)
        self.crop_health = 0.0
        self.water_availability = random.uniform(0.4, 0.6)
        self.pest_level = random.uniform(0.2, 0.4)
        self.weather = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]
        self.day = 0
        self.last_action = None
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.last_action = action  # Track the action before processing
        reward = 0
        
        # Action effects
        if action == 0:  # Irrigate
            self.soil_moisture = min(1.0, self.soil_moisture + 0.3)
            self.water_availability = max(0.0, self.water_availability - 0.25)
            reward -= 0.1
        elif action == 1:  # Fertilize
            self.soil_nutrients = min(1.0, self.soil_nutrients + 0.4)
            reward -= 0.2
        elif action == 2:  # Pesticide
            self.pest_level = max(0.0, self.pest_level - 0.4)
            reward -= 0.15
        elif action == 3:  # Plant
            if self.crop_health < 0.1:
                self.crop_health = 0.5
                reward += 1
        elif action == 4:  # Harvest
            if self.crop_health > 0.8:
                reward += 10 * self.crop_health
                self.crop_health = 0.0
            else:
                reward -= 2
        
        # Natural processes
        self.day += 1
        self.crop_health += 0.05 * self.soil_moisture * self.soil_nutrients
        self.crop_health -= 0.03 * self.pest_level
        
        # Weather effects
        if self.weather == 1:  # Drought
            self.soil_moisture = max(0.0, self.soil_moisture - 0.2)
            if random.random() < 0.3:
                self.crop_health *= 0.9
        elif self.weather == 2:  # Heavy rain
            self.soil_moisture = min(1.0, self.soil_moisture + 0.3)
            self.pest_level = min(1.0, self.pest_level + 0.15)
            if random.random() < 0.2:
                self.crop_health *= 0.85
        
        # Random events
        if random.random() < 0.15:
            self.pest_level = min(1.0, self.pest_level + 0.25)
        if random.random() < 0.1:
            self.weather = random.choices([0, 1, 2], weights=[0.5, 0.4, 0.1])[0]
        
        # Reward components
        reward += self.crop_health * 0.2
        reward -= self.pest_level * 0.3
        reward -= (1 - self.water_availability) * 0.2
        reward += min(self.soil_nutrients, 0.7) * 0.1
        
        terminated = self.day >= 365
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode is None:
            return
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)
        
        self.screen.fill((135, 206, 235))  # Sky blue
        
        # Farm plot
        pygame.draw.rect(self.screen, (139, 69, 19), (200, 200, 400, 300))
        
        # Crops
        if self.crop_health > 0.1:
            crop_size = min(2, int(self.crop_health * 3))
            for i in range(5):
                for j in range(5):
                    self.screen.blit(self.crop_sprites[crop_size], (250 + i*70, 250 + j*50))
        
        # Status bars
        self.draw_bar(50, 50, self.soil_moisture, (0, 0, 255), "Moisture")
        self.draw_bar(50, 100, self.soil_nutrients, (0, 255, 0), "Nutrients")
        self.draw_bar(50, 150, self.crop_health, (50, 150, 50), "Crops")
        self.draw_bar(50, 200, self.water_availability, (0, 100, 255), "Water")
        self.draw_bar(50, 250, self.pest_level, (255, 0, 0), "Pests")
        
        # Info text
        weather_text = ["Normal", "Drought", "Heavy Rain"][self.weather]
        text = self.font.render(f"Weather: {weather_text} | Day: {self.day}", True, (0, 0, 0))
        self.screen.blit(text, (50, 300))
        
        action_names = ["Irrigate", "Fertilize", "Pesticide", "Plant", "Harvest", "Nothing"]
        action_text = self.font.render(
            f"Last Action: {action_names[self.last_action] if self.last_action is not None else 'None'}", 
            True, (0, 0, 0))
        self.screen.blit(action_text, (50, 350))
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))
    
    def draw_bar(self, x, y, value, color, label):
        pygame.draw.rect(self.screen, (200, 200, 200), (x, y, 200, 20))
        pygame.draw.rect(self.screen, color, (x, y, int(200 * value), 20))
        pygame.draw.rect(self.screen, (0, 0, 0), (x, y, 200, 20), 2)
        text = self.font.render(f"{label}: {value:.2f}", True, (0, 0, 0))
        self.screen.blit(text, (x + 210, y))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None