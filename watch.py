"""
학습된 역진자 모델을 불러와서 화면으로 확인 (먼저 python exp.py 로 학습)

실행:  python watch.py <모델 이름>
  예)  python watch.py ppo_20261103_142530
"""
import os
import sys
from stable_baselines3 import PPO
from exp import make_env


def watch(model_name: str):
    model_dir = "models"    # exp.py 가 모델을 저장하는 폴더
    steps     = 2000        # 화면에 보여줄 스텝 수 (에피소드 최대 1000스텝)

    model = PPO.load(os.path.join(model_dir, model_name), device="cpu")
    env = make_env(shaped=False, render=True)
    obs, _ = env.reset()
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("사용법: python watch.py <모델 이름>")
    watch(sys.argv[1])
