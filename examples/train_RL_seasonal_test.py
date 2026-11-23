"""덴버 계절 모델 평가. 저장소 루트에서 python -m examples.train_RL_seasonal_test 실행."""

from contextlib import ExitStack
from pathlib import Path
import random

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import check_for_correct_spaces

from boptestGymEnv import BoptestGymEnv, NormalizedActionWrapper, NormalizedObservationWrapper
from examples.test_and_plot import test_agent_air


# ===== 실행 설정: 터미널 하나에서 실험 하나와 계절 하나를 평가 =====
TEST_NO = 7                       # 5, 6, 7, 8
SEASON = "win"                    # "all", "sum", "win"
SEEDS = [123456, 199580, 173827, 101291, 362507]
CHECKPOINTS = {                   # 단위: 학습 스텝. 선택한 계절의 목록만 실행
    "all": [100000, 150000, 200000, 250000, 300000],
    "sum": [70000, 100000, 150000, 200000, 250000, 300000],
    "win": [80000, 100000, 150000, 200000, 250000, 300000],
}
BOPTEST_URL = "http://127.0.0.1"
AGENTS_DIR = Path(__file__).resolve().parent / "agents2"
# ===== 실행 설정 끝 =====

DAY = 24 * 3600

# Keys are the experiment numbers in the training directory names.
COST_WEIGHTS = {
    5: {"all": 29, "sum": 13, "win": 76},
    6: {"all": 16, "sum": 16, "win": 16},
    7: {"all": 8, "sum": 8, "win": 8},
    8: {"all": 26, "sum": 26, "win": 26},
}
# Start seven days before each Denver reference day, as in train_RL_test.py.
TEST_START_DAYS = {
    "peak_heat": 341 - 7,
    "typical_heat": 322 - 7,
    "peak_cool": 244 - 7,
    "typical_cool": 188 - 7,
    "mix": 318 - 7,
}
OBSERVATIONS = {
    "time": (0, 604800),
    "year_sin": (-1., 1.),
    "year_cos": (-1., 1.),
    "zon_reaTRooAir_y": (280., 310.),
    "TDryBul": (250., 315.),
    "HDirNor": (0., 1000.),
    "HGloHor": (0., 1200.),
    "PriceElectricPowerDynamic": (-0.4, 0.4),
    "LowerSetp[1]": (280., 310.),
    "UpperSetp[1]": (280., 310.),
}


class DenverEvaluationEnv(BoptestGymEnv):
    def __init__(self, cost_weight, **kwargs):
        self.cost_weight = cost_weight
        super().__init__(**kwargs)

    def get_reward(self):
        kpis = self.get_kpis()
        objective = self.cost_weight * kpis["cost_tot"] + kpis["tdis_tot"]
        reward = self.objective_integrand - objective
        self.objective_integrand = objective
        return reward


def find_run_dir(agents_dir, experiment, season, seed):
    weight = COST_WEIGHTS[experiment][season]
    # Both "denvor" and "denver" occur in the training archive.
    candidates = [
        agents_dir / f"SAC_D_3e05_air_{city}_{season}_34_{weight}-1_{seed}_300k_{experiment}_{suffix}"
        for city in ("denvor", "denver")
        for suffix in ("log", "logdir")
    ]
    matches = [path for path in candidates if path.is_dir()]
    if not matches:
        raise FileNotFoundError(
            f"No run under {agents_dir} for experiment={experiment}, "
            f"season={season}, seed={seed}, weight={weight}:1"
        )
    if len(matches) > 1:
        raise ValueError(f"Multiple matching run directories: {matches}")
    return matches[0]


def find_checkpoint(run_dir, steps):
    path = run_dir / f"model_{steps}.zip"
    if path.is_file():
        return path
    # All supported runs have a 300k training budget.
    final_model = run_dir / "last_model.zip"
    if steps == 300000 and final_model.is_file():
        return final_model
    raise FileNotFoundError(f"Missing checkpoint: {path}")


def evaluate(run_dir, steps, cost_weight, url):
    checkpoint = find_checkpoint(run_dir, steps)
    print(f"Loading {checkpoint}", flush=True)
    model = SAC.load(str(checkpoint))
    # Use the step count in result paths even when loading last_model.zip.
    model_name = f"model_{steps}"
    monitor_dir = run_dir / "evaluation" / model_name

    with ExitStack() as cleanup:
        base_env = DenverEvaluationEnv(
            cost_weight=cost_weight,
            url=url,
            testcase="bestest_air",
            actions=["fcu_oveFan_u", "fcu_oveTSup_u"],
            observations=OBSERVATIONS,
            predictive_period=DAY,
            regressive_period=6 * 3600,
            scenario={"electricity_price": "dynamic"},
            random_start_time=False,
            start_time=TEST_START_DAYS["peak_heat"] * DAY,
            max_episode_length=14 * DAY,
            warmup_period=7 * DAY,
            step_period=900,
            render_episodes=False,
            log_dir=str(run_dir),
        )
        cleanup.callback(base_env.stop)
        env = NormalizedActionWrapper(NormalizedObservationWrapper(base_env))
        check_for_correct_spaces(env, model.observation_space, model.action_space)
        monitor_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(monitor_dir / "test.monitor.csv"))
        cleanup.callback(env.close)

        for period, start_day in TEST_START_DAYS.items():
            _, _, rewards, kpis = test_agent_air(
                env, model,
                start_time=start_day * DAY,
                episode_length=14 * DAY,
                warmup_period=7 * DAY,
                log_dir=str(run_dir),
                model_name=model_name,
                save_to_file=True,
                save_step_log=True,
                plot=False,
            )
            print(f"{period} (day {start_day})  reward={sum(rewards):.3f}  KPIs={kpis}")


def main():
    if TEST_NO not in COST_WEIGHTS or SEASON not in COST_WEIGHTS[TEST_NO]:
        raise ValueError("TEST_NO는 5~8, SEASON은 all, sum, win 중 하나로 설정하세요.")

    checkpoints = CHECKPOINTS[SEASON]
    if not SEEDS or not checkpoints:
        raise ValueError("SEEDS와 선택한 계절의 CHECKPOINTS에는 값을 하나 이상 넣으세요.")

    cost_weight = COST_WEIGHTS[TEST_NO][SEASON]
    print(
        f"test{TEST_NO} | {SEASON} | seeds={SEEDS} | checkpoints={checkpoints} | "
        f"BOPTEST={BOPTEST_URL}",
        flush=True,
    )
    for seed in SEEDS:
        run_dir = find_run_dir(AGENTS_DIR, TEST_NO, SEASON, seed)
        for steps in checkpoints:
            random.seed(seed)
            print(f"[test{TEST_NO} {SEASON}] seed={seed}  steps={steps}", flush=True)
            evaluate(run_dir, steps, cost_weight, BOPTEST_URL)


if __name__ == "__main__":
    main()
