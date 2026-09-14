# Closed-Loop Data Engine for Robot Skill Adaptation

조건과 궤적을 설계해 실물 시연을 만들고, 그 데이터로 학습한 VLA 정책을 검증한다. 구현·검증 플랫폼은 FAIRINO FR5이다.

**[포트폴리오 다운로드](https://github.com/hasemu1211/fr5-lerobot-connector/releases/download/portfolio-2026-09-14/FR5-Portfolio.html)** · [시스템 아키텍처](docs/architecture.md)

[![Pick & Place 시연의 고정 카메라와 손목 카메라 관측](docs/portfolio/assets/pick-place-latest-poster.jpg)](https://github.com/hasemu1211/fr5-lerobot-connector/releases/download/portfolio-2026-09-14/FR5-Portfolio.html)

Pick & Place · 21.1초

양방향 Pick & Place 40개로 SmolVLA를 학습하고, 같은 평가 관측에서 18k까지 학습률·동작 오차를 비교했다. 실물 정책 실행과 중단 원인을 기록했으며, 작업 전체 성공은 확인되지 않았다. [학습 비교와 실행 근거](docs/training-and-evaluation.md#pick--place--20260912)

## Robot Skill Adaptation

![상위 작업 목표를 실행 skill로 구체화하는 확장 맥락과, 실물 시연·정책 실행·평가·다음 수집을 잇는 데이터 엔진](docs/portfolio/skill-adaptation.drawio.svg)

그림은 폐루프의 목표 구조이다. 수집·선별·학습·오프라인 평가와 조건별 수집 추천을 구현했으며, 실물 정책 개선 효과는 후속 검증 대상이다.

| 핵심 기능 | 설계 목적 |
| --- | --- |
| 조건·궤적 설계 | [볼록 작업영역과 물체 외곽을 반영해 위치·yaw와 접근 경로를 설계한다.](docs/data-factory.md#시연-조건과-접근-궤적) |
| 동기 기록 | [영상·관절 상태·그리퍼 명령을 같은 시각의 학습 표본으로 정렬한다.](docs/dataset-quality.md#필수-자동-기준) |
| 데이터 선별 | [선택한 데이터의 원본과 학습·평가 배정을 유지해 데이터 변경의 영향을 비교한다.](docs/training-and-evaluation.md) |
| 정책 비교 | [같은 관측에서 생성한 동작을 관절·그리퍼 단위로 비교한다.](docs/training-and-evaluation.md#오프라인-평가) |
| 조건 피드백 | [수집·실행 기록을 원래 조건에 대응시켜 다음 수집 계획에 연결한다.](docs/architecture.md#task--evidence-contracts) |

<details>
<summary>직접 실행하기 · 기술 문서 · 라이선스</summary>

## 실행 준비

**직접 실행할 때:** [시작하기](docs/getting-started.md)에서 환경을 준비한다. 다음 명령은 합성 입력으로 Collection 운영 화면을 열며 로봇·카메라·데이터셋을 사용하지 않는다.

```bash
direnv exec . python3 -m tools.data_factory.operator_console --effect-scope FAKE
```

실물 장비의 준비·실행·중단은 [운영자 런북](docs/operator-runbook.md)을 따른다.

## 문서

| 알고 싶은 것 | 문서 |
| --- | --- |
| 실제 화면·관측·수치로 프로젝트 살펴보기 | [포트폴리오 열람 안내](docs/portfolio/README.md) |
| 설치와 로봇 없는 첫 실행 | [시작하기](docs/getting-started.md) |
| 입력·출력·권한·산출물 소유권 | [데이터팩토리 계약](docs/data-factory.md) |
| 장비 준비·안전·중단·복구 | [운영자 런북](docs/operator-runbook.md) |
| 전체 데이터 흐름과 모듈별 책임 | [시스템 아키텍처](docs/architecture.md) |
| 저장 구조·시간 정합·품질 판정 | [데이터셋 품질](docs/dataset-quality.md) |
| policy 학습·checkpoint·오프라인 평가 | [학습과 평가](docs/training-and-evaluation.md) |
| 설계 선택과 검증 근거 | [설계 원리](docs/engineering-story.md) |

## 라이선스

직접 작성한 코드와 문서는 [Apache License 2.0](LICENSE)을 따른다. FAIRINO 하위 모듈과 DH-Robotics CAD mesh의 권리·고지는 [Third-party notices](THIRD_PARTY_NOTICES.md)에 보존한다.

</details>
