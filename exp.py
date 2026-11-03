"""
역진자(InvertedPendulum-v5) 강화학습 입문 과제

설치:  pip install -r requirements.txt
실행:  python exp.py     # compute_reward()로 학습·평가 → models/ppo_<날짜_시간>.zip 저장
       python watch.py <모델 이름>   # 저장된 모델을 화면으로 확인
"""
import os
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

ENV_ID = "InvertedPendulum-v5"


# ---------------------------------------------------------------------------
# 1. 보상 함수 설계
#    obs = [카트 위치 x, 막대 각도 θ, 카트 속도 ẋ, 막대 각속도 θ̇]
# ---------------------------------------------------------------------------
def compute_reward(obs, action, terminated):
    # 가중치
    w_alive  = 0.0
    w_angle  = 0.0
    w_pos    = 0.0
    w_action = 0.0

    x, theta, _, _ = obs
    a = float(action[0])

    # 보상 항목
    r_alive = w_alive * float(not terminated)   # 막대가 서 있으면 +1

    # 감점 항목
    p_angle  = w_angle  * theta**2              # 막대가 기울수록
    p_pos    = w_pos    * x**2                  # 카트가 중앙에서 멀수록
    p_action = w_action * a**2                  # 힘을 세게 쓸수록

    reward  = r_alive
    penalty = p_angle + p_pos + p_action

    terms = {"r_alive": r_alive, "p_angle": p_angle,
             "p_pos": p_pos, "p_action": p_action}
    return reward - penalty, terms


class ShapedReward(gym.Wrapper):
    """환경 보상을 compute_reward() 결과로 바꿔 주는 래퍼"""

    def step(self, action):
        obs, reward_raw, terminated, truncated, info = self.env.step(action)
        reward, terms = compute_reward(obs, action, terminated)
        info["reward_raw"] = reward_raw       # 원래 환경 보상
        info["reward_terms"] = terms          # 항목별 값
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# 2. 환경 세팅
# ---------------------------------------------------------------------------
def make_env(shaped: bool, render: bool = False):
    noise = 0.01            # 초기 상태 무작위성 (고정)

    env = gym.make(ENV_ID, reset_noise_scale=noise,
                   render_mode="human" if render else None)
    return ShapedReward(env) if shaped else env


# ---------------------------------------------------------------------------
# 3. 학습 / 평가
# ---------------------------------------------------------------------------
def train():
    steps  = 100_000        # 총 학습 스텝
    n_envs = 8              # 병렬 환경 수
    save_dir   = "models"
    model_name = f"ppo_{datetime.now():%Y%m%d_%H%M%S}"
    save_path  = os.path.join(save_dir, model_name + ".zip")

    venv = make_vec_env(lambda: make_env(shaped=True), n_envs=n_envs)
    model = PPO("MlpPolicy", venv, device="cpu", verbose=0)
    model.learn(total_timesteps=steps)

    os.makedirs(save_dir, exist_ok=True)
    model.save(save_path)   # watch.py 에서 불러와 확인
    print(f"모델 저장: {save_path}")
    print(f"시각화:    python watch.py {model_name}")
    return model


def evaluate(model):
    episodes = 20           # 평가 에피소드 수

    env = make_env(shaped=False)    # 평가는 항상 원래 보상(+1/step)으로
    mean, std = evaluate_policy(model, env, n_eval_episodes=episodes)
    return mean, std        # = 막대를 세운 평균 스텝 수, 최대 1000


if __name__ == "__main__":
    model = train()
    mean, std = evaluate(model)
    print(f"평균 지속 스텝: {mean:7.1f} ± {std:5.1f}  (최대 1000)")
