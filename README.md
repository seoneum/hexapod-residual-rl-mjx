# Hexapod Residual RL with MuJoCo MJX

6족 로봇의 기구·전자 하드웨어 자산부터 MuJoCo/MJX 보행 제어와 강화학습까지 한곳에서 관리하는 통합 저장소다.

현재 강화학습의 기준 경로는 **고전 Tripod 보행 제어기 + Cartesian residual RL**이다. 정책이 보행 전체를 새로 만들지 않고, 고전 제어기가 생성한 안전한 nominal motion을 제한된 범위에서 보정한다.

> Current source of truth: [`docs/RESIDUAL_RL.md`](docs/RESIDUAL_RL.md)
>
> Current contract: `classical_wbc_cartesian_body6d_residual_v1` 24-D action + `gt_attitude_collision_contact6_coarse9_touchdown6_v3` 113-D observation

## 목차

- [프로젝트가 해결하려는 문제](#프로젝트가-해결하려는-문제)
- [현재 상태와 범위](#현재-상태와-범위)
- [시스템 구조](#시스템-구조)
- [Action과 Observation 계약](#action과-observation-계약)
- [저장소 구조](#저장소-구조)
- [설치](#설치)
- [5분 검증](#5분-검증)
- [학습 워크플로](#학습-워크플로)
- [Terrain curriculum](#terrain-curriculum)
- [실험 산출물과 W&B](#실험-산출물과-wb)
- [체크포인트 호환성](#체크포인트-호환성)
- [레거시 자료의 위치와 의미](#레거시-자료의-위치와-의미)
- [문제 해결](#문제-해결)
- [검증 체크리스트](#검증-체크리스트)
- [한계와 안전 주의사항](#한계와-안전-주의사항)
- [라이선스와 재사용](#라이선스와-재사용)

## 프로젝트가 해결하려는 문제

End-to-end locomotion policy는 강력하지만, 실패 원인을 분리하기 어렵고 실제 로봇으로 옮길 때 안전 경계를 설명하기 힘들다. 이 프로젝트는 책임을 다음처럼 나눈다.

- 고전 제어기: command filtering, position/heading PI, gait phase, swing/stance trajectory, contact adaptation, body attitude PI, workspace gate, IK, actuator safety
- 학습 정책: swing 발의 XYZ, stance 발의 Z, 몸체 6-DOF를 제한된 범위에서 보정
- 시뮬레이터: dynamics, collision contact, terrain, root ground-truth attitude 제공
- 학습기: Brax PPO를 이용한 GPU 병렬 학습, 평가, checkpoint 및 영상 기록

이 구조의 핵심은 **zero residual 상태에서도 nominal controller가 완전한 보행을 수행한다**는 점이다. 학습은 rough terrain과 계단에서 부족한 부분만 보완한다.

## 현재 상태와 범위

| 영역 | 상태 | 기준 구현 |
|---|---|---|
| 로봇 모델 | 사용 가능 | Xacro/URDF → MJCF 변환, primitive collision RL scene |
| Nominal gait | 사용 가능 | Tripod PULL + quintic-scaled cubic Bezier swing |
| Residual action | 현재 기준 | Cartesian foot 18-D + body pose 6-D |
| Observation | 현재 기준 | command, body/joint/foot state, collision contact, terrain, previous action |
| Flat command 학습 | 사용 가능 | forward/lateral/yaw Stage 0→2 curriculum |
| Mixed terrain 학습 | 사용 가능 | flat, curb, ramp, blocks, rough, stairs lane |
| Competence curriculum | 사용 가능 | 평가 성공률 기반 승급·강등 또는 sequential 진행 |
| 실험 추적 | 사용 가능 | local monitor/checkpoint/GIF + 선택적 W&B |
| 실제 하드웨어 배포 | 미검증 | 시뮬레이션 결과를 바로 서보에 적용하면 안 됨 |

이 저장소는 연구·개발 단계다. 실제 로봇의 폐루프 실험, 센서 지연, 통신 손실, 전류·온도 제한까지 검증된 production controller를 제공하지 않는다.

## 시스템 구조

```text
forward / lateral / yaw-rate command
  └─ dead zone + slew limit
      └─ world XY Position PI + Heading Hold PI
          └─ final body twist
              └─ Tripod phase manager
                  ├─ stance: -v - ω×p PULL
                  └─ swing: quintic-time-scaled cubic Bezier
                      └─ early/late collision-contact adaptation
                          └─ bounded RL residual
                              ├─ foot: swing XYZ / stance Z-only
                              └─ body: translation XYZ + roll/pitch/yaw
                                  └─ six-leg workspace accept/hold
                                      └─ numerical workspace projection
                                          └─ analytical IK
                                              └─ joint jump/rate/torque safety
```

제어 우선순위는 아래와 같다.

```text
safety > contact adaptation > bounded RL residual > nominal gait
```

정책이 소유하지 않는 항목:

- gait phase와 tripod 교대 시점
- stride/frequency와 nominal swing height
- nominal radial offset
- contact state와 early/late landing 처리
- workspace 승인, IK, joint rate와 torque clamp

Phase 0.5초가 끝나도 swing 3발의 착지가 모두 확인되지 않으면 다음 tripod를 들지 않는다. 현재 phase를 유지하면서 late-landing 탐색을 계속한다.

## Action과 Observation 계약

### Action: 24-D

| Slice | 차원 | 의미 | 권한 경계 |
|---|---:|---|---|
| `0:18` | 18 | `RF, RM, RB, LF, LM, LB` 발의 `[Δx, Δy, Δz]` | swing XYZ, stance Z-only |
| `18:21` | 3 | body forward/lateral/height | 최대 `±0.05/±0.05/±0.10 m` |
| `21:24` | 3 | body roll/pitch/yaw | 최대 `±45/±45/±25 deg` |

- stance X/Y residual은 코드에서 정확히 0으로 강제된다.
- body request는 기본 0.15초 low-pass filter를 통과한다.
- body pose는 6개 다리 모두 workspace와 `±135 deg` joint limit를 만족할 때만 함께 승인된다.
- 승인 불가능한 pose는 억지로 적용하지 않고 직전 승인 상태를 유지한다.

### Observation: 113-D

| 구성 | 차원 |
|---|---:|
| forward/lateral/yaw command | 3 |
| body local velocity, angular velocity, gravity | 9 |
| joint position, scaled velocity | 36 |
| body-frame foot position | 18 |
| MuJoCo foot–world collision contact | 6 |
| heading-aligned 3×3 terrain grid | 9 |
| six nominal-touchdown heights | 6 |
| gait phase sin/cos | 2 |
| previous applied action | 24 |
| 합계 | **113** |

접촉은 발 높이 threshold나 가상 압력센서를 사용하지 않는다. MuJoCo의 `foot_collision`과 floor/ramp/block/rough/stair geom 사이 실제 collision pair만 사용한다. Roll/pitch/yaw는 현재 MuJoCo root quaternion ground truth다.

세부 수식, terrain observation, reward, termination 기준은 [`docs/RESIDUAL_RL.md`](docs/RESIDUAL_RL.md)를 따른다.

## 저장소 구조

```text
.
├── HW/                              # CAD, STL/3MF, PCB, 배선, URDF 자산
├── SW/
│   ├── mjx/                         # 현재 MuJoCo/MJX 제어·학습 코드
│   │   ├── hexapod_mjx/             # 모델, 제어기, environment, CEM/PPO 유틸
│   │   ├── tests/                   # residual/terrain contract 테스트
│   │   ├── train_command_curriculum.py
│   │   ├── train_rough_terrain.py
│   │   └── train_competence_curriculum.py
│   ├── Jetson/                      # Jetson 관련 자료
│   ├── MATLAB/                      # 기존 Simulink 모델과 파라미터
│   ├── STM32/                       # STM32 설정 기록
│   └── Sensor_test/                 # 센서·PCB 실험 코드
├── docs/
│   ├── RESIDUAL_RL.md               # 현재 residual RL 명세의 source of truth
│   ├── Hexapod_MJX_Obsidian_Study_Vault.md
│   └── SPIDER_MUJOCO_STUDY_GUIDE.md # 과거 standalone SB3 변환 과정 해설
├── reference/spider_rl/             # mujoco_tuto에서 보존한 비교·학습용 snapshot
├── Hexapod-MJX-가이드/              # 실행 wrapper와 한국어 운영 가이드
├── 00_mjx_minimal.py                # 최소 MJX 예제
└── mjx_tutorial.ipynb               # 학습용 notebook
```

코드를 수정할 때의 기준:

| 목적 | 먼저 볼 파일 |
|---|---|
| nominal gait와 safety | `SW/mjx/tripod_core.py` |
| residual action 적용 | `SW/mjx/rough_terrain_env.py` |
| 모델·scene 생성 | `SW/mjx/hexapod_mjx/model.py`, `SW/mjx/prepare_*.py` |
| flat command 학습 | `SW/mjx/train_command_curriculum.py` |
| terrain 학습 | `SW/mjx/train_rough_terrain.py` |
| 전체 curriculum | `SW/mjx/train_competence_curriculum.py` |
| 평가·영상 | `SW/mjx/best_policy_video.py`, `SW/mjx/visualize_residual_policy.py` |
| 계약과 설계 의도 | `docs/RESIDUAL_RL.md`, `SW/mjx/RL_DESIGN.md` |

## 설치

### 1. Clone과 가상환경

```bash
git clone https://github.com/seoneum/hexapod-residual-rl-mjx.git
cd hexapod-residual-rl-mjx

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r SW/mjx/requirements-train.txt
```

시각화·nominal controller만 확인할 경우에는 더 작은 의존성을 사용할 수 있다.

```bash
python -m pip install -r SW/mjx/requirements.txt
```

현재 학습 조합은 다음 핵심 버전을 고정한다.

```text
jax==0.6.2
mujoco-mjx==3.12.0
playground==0.1.0
brax==0.14.1
```

### 2. NVIDIA CUDA 12 학습 환경

`requirements-train.txt` 설치 후 CPU JAX를 CUDA wheel로 교체한다.

```bash
python -m pip install --upgrade "jax[cuda12]==0.6.2"
unset LD_LIBRARY_PATH
python -c "import jax; print(jax.devices())"
```

본 학습 전 출력에 `GpuDevice`가 있는지 확인한다. `--smoke`는 CPU에서도 가능하지만 기본 2048-env PPO 학습은 GPU를 전제로 한다.

### 3. 선택 사항: W&B

```bash
python -m pip install wandb
wandb login
```

W&B를 사용하지 않으면 학습 명령에서 `--wandb`만 제거하면 된다. checkpoint, monitor JSON, GIF는 로컬에 그대로 남는다.

## 5분 검증

먼저 Python syntax와 핵심 계약 테스트를 확인한다.

```bash
python -m compileall -q SW/mjx
python -m unittest discover -s SW/mjx/tests -v
```

로봇 모델과 nominal gait를 확인한다.

```bash
# GUI
python SW/mjx/view_robot.py
python SW/mjx/view_rl_scene.py

# Headless/EGL
MUJOCO_GL=egl python SW/mjx/view_robot.py --headless
MUJOCO_GL=egl python SW/mjx/view_rl_scene.py --headless
```

학습 없이 environment reset/step/JIT 계약만 검사한다.

```bash
python SW/mjx/train_command_curriculum.py \
  --smoke --smoke-steps 100 --run-name command-smoke

python SW/mjx/train_rough_terrain.py \
  --smoke --smoke-steps 100 \
  --terrain-layout mixed --terrain-level 4 --terrain-randomize \
  --run-name terrain-smoke
```

Smoke가 통과해야 긴 학습을 시작한다.

## 학습 워크플로

### 1. Flat command baseline

forward-only → limited yaw → full forward/lateral/yaw의 3단계 command curriculum을 학습한다.

```bash
python SW/mjx/train_command_curriculum.py \
  --run-name flat-transfer-source \
  --timesteps 50000000 \
  --num-envs 2048 \
  --num-evals 100 \
  --wandb \
  --wandb-project hexapod-command-curriculum
```

기본 Stage 속도 상한은 `0.10/0.18/0.27 m/s`, yaw 상한은 `0.00/0.15/0.35 rad/s`다. 가장 빠른 다리의 horizontal stroke는 140 mm로 제한되며, 고속에서는 기구학 범위에 맞게 yaw command가 자동 축소된다.

### 2. Flat → mixed terrain transfer

flat checkpoint의 policy와 normalizer를 terrain 학습의 초기값으로 사용한다. Terrain reward가 달라 critic은 기본적으로 초기화하지 않는다.

```bash
python SW/mjx/train_rough_terrain.py \
  --run-name terrain-transfer-level0 \
  --terrain-layout mixed \
  --terrain-level 0 \
  --init-checkpoint SW/mjx/runs/command/<flat-run>/checkpoints \
  --timesteps 50000000 \
  --num-envs 2048 \
  --num-evals 100 \
  --terrain-randomize \
  --wandb \
  --wandb-project hexapod-rough-terrain
```

### 3. 전체 competence curriculum

호환 checkpoint가 없으면 짧은 flat baseline을 먼저 만들고, 이후 rough/mixed terrain에서 최대 총 20 cm 계단까지 진행한다.

```bash
python SW/mjx/train_competence_curriculum.py \
  --run-name mixed-body6d \
  --flat-baseline-timesteps 1000000 \
  --stages 8 \
  --stage-timesteps 5000000 \
  --level-progression sequential \
  --wandb \
  -- --num-envs 2048 --num-evals 20 --terrain-randomize --best-video
```

이미 호환되는 24-D/113-D checkpoint가 있으면 다음처럼 시작한다.

```bash
python SW/mjx/train_competence_curriculum.py \
  --run-name mixed-body6d-resume \
  --init-checkpoint SW/mjx/runs/command/<run>/checkpoints \
  --stages 8 \
  --stage-timesteps 5000000 \
  --wandb \
  -- --num-envs 2048 --num-evals 20 --terrain-randomize
```

`--level-progression competence`는 evaluation success가 0.80을 넘을 때 승급하고 0.50보다 낮으면 강등한다. `sequential`은 stage마다 level을 순서대로 올린 뒤 level 4를 유지한다.

### 4. 학습 전 terrain preview

```bash
python SW/mjx/preview_terrain_curriculum.py \
  --checkpoint SW/mjx/runs/terrain/<terrain-run>/checkpoints \
  --wandb
```

`--checkpoint`를 생략하면 zero-residual nominal controller로 모든 level을 확인한다. 학습 결과는 이 baseline과 비교해야 한다.

## Terrain curriculum

| Level | 계단 전체 상승 | Ramp 상승 | yaw 제한 | Stairs lane 확률 |
|---:|---:|---:|---:|---:|
| 0 | 2–4 cm | 4–8 cm | 0.00 rad/s | 0% |
| 1 | 4–8 cm | 8–12 cm | 0.05 rad/s | 5% |
| 2 | 8–12 cm | 12–16 cm | 0.10 rad/s | 15% |
| 3 | 12–16 cm | 16–20 cm | 0.20 rad/s | 25% |
| 4 | 16–20 cm | 20–24 cm | 0.35 rad/s | 30% |

Level 4의 6단 계단은 최대 `0.20 / 6 = 0.0333 m` 단차, 총 상승 0.20 m다. `--terrain-total-rise`는 총 높이, `--terrain-step-height`는 한 단 높이 override이며 동시에 사용할 수 없다. 총 상승이 0.20 m를 넘으면 학습 시작 전에 거부한다.

`--terrain-randomize`는 level 범위 안에서 재현 가능한 terrain을 만들고 reset마다 lane을 확률적으로 선택한다. Level 4에서는 friction, mass, servo, damping randomization도 적용한다.

## 실험 산출물과 W&B

각 실행은 서로 덮어쓰지 않는 독립 디렉터리를 만든다.

```text
SW/mjx/runs/<task>/<name>_<timestamp>_seed<seed>/
├── checkpoints/                    # Brax checkpoint
├── monitor/
│   ├── latest_metrics.json
│   ├── best_score.json
│   ├── stage_metrics_latest.json
│   └── stage_metrics_history.jsonl
├── videos/
│   ├── progress/                   # 기본 0/25/50/75/100% snapshot
│   └── best_*.gif                  # NEW_BEST deterministic rollout
├── config.json
└── run_metadata.json
```

같은 `--run-name`을 다시 사용해도 timestamp와 seed suffix가 붙어 기존 결과와 섞이지 않는다. Trainer는 기존 checkpoint/monitor/video가 있는 경로를 덮어쓰지 않는다.

먼저 볼 metric:

| Metric | 의미 |
|---|---|
| `eval/episode_reward` | 종합 평가 reward |
| `eval/episode_terrain_success` | 비종료 + 속도 오차 + 진행 거리 조건을 만족한 비율 |
| `velocity_error_mps`, `yaw_error_rps` | command tracking 오차 |
| `position_error_m`, `heading_error_rad` | controller tracking 오차 |
| `projection_cost` | workspace projection 개입량 |
| `torque_rms_nm`, `torque_saturation` | actuator 부담과 포화 |
| `self_collision` | self-collision 발생량 |
| `contact_early_landing`, `contact_lost` | contact adaptation 상태 |
| `posture_command_accepted` | body 6-DOF workspace 승인 여부 |

W&B에서는 x축을 `train/global_step`으로 고정해야 eval 주기가 다른 run을 올바르게 비교할 수 있다.

네트워크가 없는 서버에서는 다음처럼 기록 후 동기화한다.

```bash
python SW/mjx/train_rough_terrain.py ... --wandb --wandb-mode offline
wandb sync wandb/offline-run-*
```

## 체크포인트 호환성

체크포인트는 action/observation의 차원뿐 아니라 **의미와 순서까지 정확히 같아야** 재사용할 수 있다.

| 계열 | Action | Observation | 현재 환경 재사용 |
|---|---:|---:|---|
| Current body6d v1/v3 | 24 | 113 | 가능 |
| 이전 Cartesian residual | 22 | 110 | 불가 |
| 이전 collision/contact 113-D v2 | 22 또는 24 | 113 | 불가: contact 의미가 다름 |
| Legacy residual PPO | 7 | 별도 계약 | 불가 |
| `reference/spider_rl` SB3 direct env | 18 | 48 | 불가 |

호환되지 않는 checkpoint에 차원 padding, slicing, 임의 normalizer 재사용을 하지 않는다. 계약이 바뀌면 `fresh` 학습을 시작한다.

## 레거시 자료의 위치와 의미

과거 private `mujoco_tuto` 저장소의 고유 자료는 이 canonical 저장소 안으로 보존했다.

- [`docs/SPIDER_MUJOCO_STUDY_GUIDE.md`](docs/SPIDER_MUJOCO_STUDY_GUIDE.md): HEXAPEDAL URDF를 standalone MuJoCo + Gymnasium + SB3 PPO stack으로 옮긴 과정
- `reference/spider_rl/`: 당시 모델 생성기, 환경, train/eval/check script, 테스트의 slim snapshot
- `완전 튜토리얼.md`: 초기 MuJoCo/Isaac 학습 배경

`reference/spider_rl/`는 현재 구현의 dependency가 아니며 비교·학습·provenance 보존용이다. 여기의 18-D/48-D SB3 contract와 현재 24-D/113-D MJX contract를 섞지 않는다. 현재 코드를 수정할 때는 `SW/mjx/`와 `docs/RESIDUAL_RL.md`를 기준으로 한다.

## 문제 해결

### JAX가 CPU만 표시됨

```bash
python -c "import jax; print(jax.devices())"
```

- CUDA 12 wheel을 설치했는지 확인한다.
- 현재 shell의 `LD_LIBRARY_PATH`가 pip CUDA library와 충돌하면 `unset LD_LIBRARY_PATH` 후 다시 확인한다.
- CPU에서는 `--smoke` 또는 매우 작은 debug run만 수행한다. 긴 PPO 학습을 CPU로 돌리지 않는다.

### SSH/tmux에서 MuJoCo viewer 또는 GIF가 실패함

```bash
export MUJOCO_GL=egl
```

Headless NVIDIA 환경에서는 EGL을 사용한다. 렌더링 실패는 trainer의 checkpoint 저장을 중단하지 않고 `best_video_error`로 기록되지만, 영상 산출물은 별도로 확인해야 한다.

### 기존 run 디렉터리를 덮어쓸 수 없음

의도된 보호 동작이다. 새 `--run-name`을 사용하거나 자동 timestamp 디렉터리를 그대로 사용한다. 기존 결과를 삭제한 뒤 재사용하는 방식은 권장하지 않는다.

### Checkpoint load가 실패함

- checkpoint가 24-D/113-D current contract인지 확인한다.
- `run_metadata.json`의 contract version을 확인한다.
- 과거 22-D, 7-D, 18-D checkpoint는 변환하지 말고 fresh run을 시작한다.

### 결과가 움직이지만 학습 품질이 나쁨

다음 순서로 원인을 좁힌다.

1. zero-residual nominal gait 영상 확인
2. collision contact와 late landing metric 확인
3. `projection_cost`, joint rate, torque saturation 확인
4. command tracking과 posture acceptance 확인
5. 그다음 reward scale과 residual penalty 조정

Nominal controller가 실패하는 상태에서 reward tuning부터 시작하지 않는다.

## 검증 체크리스트

Pull request 또는 중요한 실험 전에 최소한 다음을 확인한다.

```bash
python -m compileall -q SW/mjx
python -m unittest discover -s SW/mjx/tests -v
python SW/mjx/train_command_curriculum.py \
  --smoke --smoke-steps 100 --run-name pr-command-smoke
python SW/mjx/train_rough_terrain.py \
  --smoke --smoke-steps 100 --terrain-layout mixed --terrain-level 4 \
  --terrain-randomize --run-name pr-terrain-smoke
```

수동 확인 항목:

- action/observation contract version이 바뀌었다면 문서와 checkpoint gate도 함께 변경했는가
- stance XY residual이 0으로 유지되는가
- 발 접촉이 collision pair에서만 계산되는가
- body pose가 six-leg workspace gate를 우회하지 않는가
- 새 실험이 별도 run directory에 기록되는가
- best/progress 영상이 의도한 terrain과 command를 보여주는가

## 한계와 안전 주의사항

- 현재 attitude observation은 실제 IMU가 아니라 MuJoCo root ground truth다.
- 접촉은 실제 압력센서가 아니라 simulator collision contact다.
- MJX policy step은 50 Hz이며 목표 실제 제어 주기는 200 Hz다. 두 주기의 차이는 실기 전 검증 대상이다.
- actuator `±8 Nm` clamp는 시뮬레이션 안전 경계이지 실제 서보·기구·전원 보호를 보장하지 않는다.
- CAD/URDF, 질량·관성, friction, actuator model과 실제 제작체 사이 오차가 존재할 수 있다.
- 학습된 policy를 실제 로봇에 올리기 전에 joint limit, 비상 정지, 전류 제한, tether, 저속 stand test, 단일 다리 검증을 별도로 수행해야 한다.

## 라이선스와 재사용

저장소 전체를 포괄하는 root license는 현재 선언되어 있지 않다. 따라서 전체 프로젝트를 임의로 재배포하거나 상업적으로 사용해도 된다고 가정하면 안 된다.

- `HW/urdf/LICENSE`와 각 source file의 SPDX/copyright header를 개별 확인한다.
- `reference/spider_rl/`은 provenance 보존용 snapshot이며 파일별 라이선스 경계를 유지해야 한다.
- 외부 사용이나 재배포가 필요하면 먼저 저장소 소유자에게 문의한다.

문서와 구현이 충돌하면 현재 계약은 [`docs/RESIDUAL_RL.md`](docs/RESIDUAL_RL.md), 실제 동작은 `SW/mjx/` 코드를 최종 기준으로 판단한다.
