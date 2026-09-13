## Purpose

FR5가 Collection, Curation, Training/Evaluation, Rollout, Learning Evidence, Public Documentation을 거치며 만든 주장을 기존 authority를 침범하지 않고 재현 가능한 증거와 연결한다.

## ADDED Requirements

### Requirement: Portfolio proof loop outcome
시스템은 Collection, Curation, Training/Evaluation, Rollout, Learning Evidence, Public Documentation의 결과와 소비자를 canonical output과 handoff로 연결해야 한다(SHALL). 이 연결은 모든 작업이 같은 순서를 통과하는 선형 파이프라인이 아니라, 필요한 증거와 권한이 충족된 작업들이 분기·병렬 진행·재평가할 수 있는 제품의 동작이어야 한다(SHALL). 외부에 공개할 결과는 현재 범위에 맞는 immutable evidence로 추적할 수 있어야 한다(SHALL).

#### Scenario: Evidence-backed result reaches public documentation
- **WHEN** 한 결과가 그 주장에 필요한 선행 증거를 갖추어 공개 후보가 된다
- **THEN** 각 handoff는 canonical output과 exact evidence reference를 가리키고 Public Documentation은 증명된 범위만 설명한다

### Requirement: Evidence enables independent work and bounded feedback
새로운 canonical evidence는 관련된 복수 소비자가 독립적으로 사용할 수 있어야 하며(SHALL), 평가 완료를 모든 수집의 선행 조건으로 강제해서는 안 된다(MUST NOT). 데이터 부족, 선별 결과, 학습·평가 결과와 실물 관찰은 각각 필요한 수집·선별·평가 작업을 열 수 있어야 한다(SHALL). 같은 입력과 같은 작업의 재전달은 중복 실행을 만들지 않아야 하며(SHALL), 변경된 입력은 영향을 받는 결과를 식별하고 이전 결과를 해당 입력에 대한 현재 증거로 오인하지 않게 해야 한다(SHALL). 반복 작업에는 유한한 자원·시도 범위와 종료 사유가 있어야 한다(SHALL).

#### Scenario: New data has several consumers
- **WHEN** 새 episode가 canonical technical admission을 완료한다
- **THEN** 품질 분석과 검토 준비는 각자의 입력 조건에 따라 진행할 수 있고, semantic 또는 training 승인을 기다리는 작업이 독립적인 safe 작업을 막지 않는다

#### Scenario: Evidence changes a previous conclusion
- **WHEN** 새 데이터 또는 평가 결과가 이전 판단의 입력을 바꾼다
- **THEN** 시스템은 영향받는 작업과 입력 계보를 구분해 필요한 작업을 다시 준비하며, 변경되지 않은 입력의 중복 작업이나 무한 재시도를 만들지 않는다

### Requirement: Product owns collection and evaluation execution
장기 완료 결과는 recommendation 생성이나 에이전트의 수동 연결에 그쳐서는 안 된다(MUST NOT). 제품은 적격 범위 안에서 필요한 수집을 선택하고 기존 execution owner를 통해 실행하며, 저장된 결과를 선별·승인된 학습 및 평가로 전달하고 후속 행동을 결정할 수 있어야 한다(SHALL). Recommendation의 advisory authority는 유지하며, 실제 효과와 실패 복구는 해당 제품 owner가 책임져야 한다(SHALL). Orca의 개발 Task 그래프는 이 제품 동작의 구현 증거를 대신하지 않는다(MUST NOT).

#### Scenario: A selected collection action becomes real evidence
- **WHEN** 수집 필요성이 선택되고 해당 실행의 입력·권한·자원 조건이 모두 충족된다
- **THEN** 제품은 기존 single motion owner로 유한한 수집을 실행하고 canonical commit 또는 failure evidence를 소비자에게 전달하며, 추천 파일만 만든 상태를 수집 완료로 표시하지 않는다

#### Scenario: Collected experience reaches evaluation
- **WHEN** 데이터 선별, training authorization 및 실행 자원이 해당 학습·평가에 충족된다
- **THEN** 제품은 실제 학습·평가 결과를 정확한 데이터·split·checkpoint 계보로 연결하고, offline loss와 physical effectiveness를 구분해 다음 작업에 사용한다

### Requirement: Curator owns reusable acquisition recommendations
Curation은 품질·분포·선별 근거를 다음 수집 조건으로 바꾸는 recommendation의 제품 소프트웨어 책임을 소유해야 한다(SHALL). 지원하는 입력과 정책 범위의 반복 추천은 canonical evidence, 현재 task·조건과 유한한 취득 예산을 받아 재현 가능한 함수 및 기존 제품 진입점으로 산출되어야 하며(SHALL), 매회 AI 조사·수동 좌표 조립·파일 전달을 필수로 요구해서는 안 된다(MUST NOT). 기존 DQA·Rollout/evaluation 결과와 sampler를 재사용하고, 추천의 근거·입력 계보·지원 범위·제한을 다음 소비자에게 전달해야 한다(SHALL). 이는 별도 서비스나 범용 전략 계층을 요구하지 않는다.

#### Scenario: Supported evidence produces the next collection proposal
- **WHEN** 지원하는 task의 canonical 수집·분석 근거와 현재 조건·예산으로 다음 수집안을 요청한다
- **THEN** 제품은 같은 입력에서 재현 가능한 수집 조건과 선택 이유를 산출하며, 호환되는 여러 campaign의 근거를 소비할 수 있다
- **AND** pickup 근거를 pick-place 성공 근거로 바꾸거나 고정 예제 수·seed·좌표를 모든 입력의 정책으로 사용하지 않으며, 지원하지 않는 입력과 실제 데이터 부족을 구분한다

#### Scenario: A recommendation is consumed rather than manually reconstructed
- **WHEN** 추천을 다음 수집에 적용한다
- **THEN** Collection은 그 결과를 기존 authoring·계획·실행 owner에서 직접 소비하며, 현재 scene과 실행 조건은 해당 owner가 확인한다
- **AND** 추천 출력만으로 연결 완료를 선언하지 않고 실제 소비 및 재전달·입력 변경·실패 경로를 검증한다
- **AND** 연구는 정책을 선택·개선하거나 새 불확실성을 해결할 때 수행하며, 지원 범위의 정상 반복 호출을 대신하지 않는다

### Requirement: Collection cohesion and reusable execution remain distinct
Collection 전용 계획 작성·campaign 운영 기능은 책임을 식별할 수 있는 `collection/` 패키지 안에 응집되어야 한다(SHALL). 실제 VLA 동작 실행과 rollout 녹화에도 사용하는 실행·기록 기능은 Collection 밖의 독립된 공용 모듈에 있어야 하며(SHALL), 다른 소비자가 Collection UI나 campaign 상태를 구성해야만 재사용할 수 있어서는 안 된다(MUST NOT). 공용화는 기존 motion·recorder·scene·cell·admission owner를 유지하고, 작업별 시연 생성과 학습 정책 실행의 입력·종료·기록 의미를 구분해야 한다(SHALL). Curator의 추천 정책과 여러 도메인을 제공하는 사용자 인터페이스를 Collection 전용 책임으로 흡수해서는 안 된다(MUST NOT).

#### Scenario: Collection-specific behavior changes
- **WHEN** 수집 계획 작성이나 campaign 운영 기능을 변경한다
- **THEN** Collection 전용 구현과 외부 계약은 해당 패키지에서 찾고 검증할 수 있으며, 공용 실행·기록 구현을 복제하거나 rollout 전용 정책을 함께 수정하지 않는다

#### Scenario: Learned rollout reuses physical execution and recording
- **WHEN** 학습 정책 실행 또는 rollout 녹화가 공용 기능을 소비한다
- **THEN** Collection 전용 UI·draft·campaign 없이 해당 계약을 직접 사용하고, 기존 single motion owner와 기록 lifecycle·실행 권한·원본 provenance를 보존한다
- **AND** 실제 소비자 연결과 실패·중단·재시작 경로를 검증하며, 폴더 이동이나 사용되지 않는 facade만으로 재사용 완료를 주장하지 않는다

#### Scenario: Migration proceeds without interrupting acquisition
- **WHEN** 현재 구조를 단계적으로 이전한다
- **THEN** 활성 수집 프로세스와 원본 데이터를 건드리지 않는 변경·통합 경계를 사용하고, 기존 실행 진입점·저장 경로·schema·digest 의미의 호환성을 검증한다
- **AND** 작은 계약 추출은 중간 결과일 수 있으나 Collection 응집과 공용 실행·기록 분리의 전체 완료를 대신하지 않는다

### Requirement: Native owners enforce safety without duplicate operator gates
실행 안전과 데이터·승인의 유효성 검사는 해당 기존 제품 owner가 자신의 소비 경계에서 책임져야 한다(SHALL). Coordinator와 다른 lane은 적용 대상·입력·버전·유효 범위가 일치하는 canonical 검사 결과를 재사용하고, 동일한 사실을 사람이나 AI가 다시 확인하는 절차·확인 문구·별도 safety owner를 추가해서는 안 된다(MUST NOT). 재검증은 변경된 입력·실행 조건, 기존 계약의 freshness 요구 또는 구체적인 실패 근거에 필요한 범위로 제한해야 한다(SHALL). 변경된 코드의 필요한 회귀와 실제 실행 시 필요한 fresh 검사를 생략하거나 기존 gate를 우회한다는 의미가 아니다.

#### Scenario: Existing native admission already covers a repeated operation
- **WHEN** 기존 허용 범위의 반복 작업을 제품이 처리하고 필요한 native admission이 충족된다
- **THEN** 별도의 coordinator 재승인·수동 evidence 대조 없이 기존 execution owner가 진행한다
- **AND** 권한·scene·cell·exact plan·single motion owner·semantic·physical binding·training approval의 의미와 적용 범위는 유지한다

#### Scenario: Only one consumer loses valid evidence
- **WHEN** 특정 입력 변경이나 실패로 한 소비자의 검증 결과가 더 이상 유효하지 않다
- **THEN** 해당 결과를 소유한 시스템이 필요한 검사·복구와 중단 사유를 처리하고, 조사 가능한 오류를 사람 확인으로 대체하지 않는다
- **AND** 영향을 받지 않는 근거와 완료된 물리 효과는 보존하며 독립적인 적격 작업은 계속한다

### Requirement: Automation takes over qualified responsibilities rather than bypassing gates
반복적인 사람 입력을 줄이는 전환은 그 입력이 담당하던 관찰·판정·권한 범위·실패 대응을 명시하고 검증된 시스템 책임으로 인수해야 한다(SHALL). 관측 정확도, 잘못된 승인과 중단, 복구 가능성 및 사람 개입 빈도를 적용 범위 안에서 평가해야 한다(SHALL). 기존 gate를 바꾸는 개별 전환은 해당 authority의 승인된 계약과 회귀·실물 evidence를 갖추어야 하며(SHALL), 장기 자동화 intent 자체를 현재 gate 충족이나 승인으로 해석해서는 안 된다(MUST NOT).

#### Scenario: Repeated confirmation is replaced within a qualified scope
- **WHEN** 한 확인 책임을 시스템이 인수할 근거와 해당 authority의 변경 계약이 충족된다
- **THEN** 적격 범위의 반복 입력을 줄일 수 있지만 범위 밖·오래된 관측·불확실한 판정은 자동 승인하지 않고 안전한 중단 또는 사람 판단으로 전달한다

### Requirement: Decisions are product-consumable beyond the interaction surface
사람 또는 적격 시스템이 내린 판단은 정확한 대상·입력 근거·판정 주체와 적용 범위를 기존 결정 owner의 결과로 결속하여 다음 제품 기능이 소비할 수 있어야 한다(SHALL). 화면 표시, 터미널 입력 또는 에이전트의 수동 전달만으로 handoff가 완료됐다고 간주해서는 안 된다(MUST NOT). 소비자는 판단의 현재 적용 가능성과 자신의 실행 권한을 검증해야 하며(SHALL), 하나의 판단을 semantic·physical·training 등 다른 권한으로 확대해서는 안 된다(MUST NOT). 이 요구는 별도 범용 decision framework나 중복 ledger를 요구하지 않는다.

#### Scenario: A reviewed candidate reaches the next product function
- **WHEN** 정확한 candidate와 review 근거에 대한 유효한 판단이 기존 owner에 기록된다
- **THEN** 다음 기능은 경로나 digest를 사람이 옮겨 적지 않아도 기존 결과를 소비하고 입력 계보와 판단 범위를 검증한다
- **AND** 변경된 대상이나 충돌하는 재전달은 기존 판단으로 실행되지 않으며, 응답 유실 복구는 동일 효과를 다시 만들지 않는다

#### Scenario: A qualified system takes over a bounded judgment
- **WHEN** 해당 판단 owner의 승인된 계약과 실제 검증이 시스템의 관찰·판정 범위 및 실패 대응을 충족한다
- **THEN** 제품은 그 범위에서 반복적인 사람 선택 없이 시스템 판단을 소비할 수 있고, 시스템 판정을 사람 승인으로 기록하지 않는다
- **AND** 불확실하거나 범위 밖인 판정은 명시적인 미결 상태와 필요한 사람 판단으로 이어지며, 독립적으로 적격인 작업은 계속할 수 있다

### Requirement: Evidence state has one meaning
모든 material claim은 `SUPPORTED`, `PARTIAL`, `UNKNOWN` 중 하나여야 한다(SHALL). `PARTIAL`은 제한을 보존해야 하고(MUST), `UNKNOWN`은 실패·안전·승인·효과로 추정되어서는 안 된다(MUST NOT).

#### Scenario: Incomplete evidence remains bounded
- **WHEN** claim의 일부 evidence가 없거나 적용 범위를 벗어난다
- **THEN** claim은 `PARTIAL` 또는 `UNKNOWN`으로 남고 누락된 의미를 만들어 내지 않는다

### Requirement: Purpose-appropriate production presets preserve quality and lineage
운용 목적별 생산 프리셋은 구간별 동작과 촬영·기록 품질을 함께 고려해야 하며(SHALL), 실제 적용값은 각 기존 canonical owner에서 일관되게 해석되고 계획과 episode evidence에서 추적 가능해야 한다(SHALL). 적합성은 수집 시간, 안정적인 작업 성공, 사용 가능한 영상·동기화 품질의 소량 실행 evidence로 판단해야 하며(SHALL), 설정의 적격 표기만으로 최적 속도나 충분한 데이터 품질을 주장해서는 안 된다(MUST NOT). 근거 있는 기존 그리퍼 설정과 원본 데이터는 보존해야 한다(SHALL).

#### Scenario: A production preset is selected or revised
- **WHEN** 운영자가 목적에 맞는 프리셋을 선택하거나 새로운 적용값을 검토한다
- **THEN** 실제 설정과 그 근거를 재현할 수 있고, 변경된 실행은 기존 exact-plan 및 물리 gate를 충족하며, 검증 전의 용도 적합성은 UNKNOWN으로 남는다

#### Scenario: A faster or higher-fidelity preset is considered
- **WHEN** 속도 또는 촬영·기록 품질의 상향을 검토한다
- **THEN** 시간과 충분한 품질을 함께 비교하며 최대 속도·최대 화질 자체를 목표로 삼지 않고, 프리셋 선택이 TEST/GENERAL 데이터 구분이나 technical·semantic·training authority를 자동 변경하지 않는다

#### Scenario: One arm policy is selected for multiple workspaces
- **WHEN** Web 또는 CLI에서 구간별 속도 정책을 선택한다
- **THEN** 기존 명시적 phase를 사용하는 하나의 version/digest 정책이 A/B에 공통 적용되며, geometry 및 hardware/planner 최대값과 분리된다
- **AND** 기존 자격과 정확히 결속된 해석 결과만 계획·episode evidence·HOME/시작 자세 복구에 사용되고, 기존 설정을 선택한 과거 프로그램·계획·데이터의 재생은 변경되지 않는다

#### Scenario: A candidate policy has not been physically qualified
- **WHEN** 기존 자격에서 새 정책의 검증 후보를 준비한다
- **THEN** 결과는 UNQUALIFIED이며 기존 QUALIFIED 상태·qualified_at을 상속하지 않는다
- **AND** Web에서 구간별 요청값과 자격 필요 상태를 검토하고 초안에 선택할 수 있지만 필요한 endpoint 자격이 없으면 일반 수집 경로의 계획 확정·실행은 거부된다
- **AND** 기존 검증 설정으로 돌아가는 데 추가 승인·타이핑이 필요하지 않고, 객체 배치·수량·후속 편집을 보존한다

#### Scenario: A bounded trial evaluates a candidate before production qualification
- **WHEN** 이미 승인된 시험 범위에서 기존 적격 geometry와 hardware/planner 한계를 보존하는 후보 속도 정책을 평가한다
- **THEN** 기존 TEST_ONLY 실행 owner가 정확한 후보·계획·유한한 실행 범위를 결속해 시험하고, 기존 hardware·human·scene·cell·exact-plan·single-motion 조건을 그대로 소비한다
- **AND** 후보의 미검증 상태를 유지한 채 실제 실행 결과와 적용값을 남기며, 시험을 위해 QUALIFIED 표기나 과거 qualified_at을 만들어 넣지 않는다
- **AND** 영상 위치 추정이나 새 확인 문구를 추가 필수 조건으로 요구하지 않고, 실제 실행 불확실성과 무관한 UI·저장 오류를 구분한다
- **AND** 시험 성공만으로 production 자격·semantic PASS·training authority를 자동 부여하지 않는다

#### Scenario: Policy or qualification changes after selection
- **WHEN** 선택한 정책 또는 endpoint 자격의 digest·해석값이 변경되거나 서로 일치하지 않는다
- **THEN** 계획 생성과 복구는 효과 전에 거부되며, 다른 endpoint 자격이나 과거 정책의 승인으로 대체하지 않는다
- **AND** 응답 유실은 canonical 상태를 다시 읽어 복구하고 명령을 자동 재전송하지 않는다

### Requirement: Existing owners retain authority
OpenSpec은 지속 가능한 외부 행동 intent, 안정된 경계와 outcome 단위의 완료 기준 및 evidence 연결을 소유해야 한다(SHALL). Orca는 상세 실행·의존성·attempt 진척·live resource·blocker·handoff를, source와 tests는 실행 가능한 계약과 수치 truth를, MEX는 파생된 로컬 탐색 정보를, Public Documentation은 검증된 사용자 의미를 계속 소유해야 한다(SHALL).

#### Scenario: Execution state changes
- **WHEN** task의 담당자, 순서, 진행 상태 또는 blocker가 바뀐다
- **THEN** Orca가 상세 변경을 기록하며 OpenSpec의 outcome·완료 기준·evidence 연결이 달라지지 않는 한 OpenSpec을 갱신하지 않는다

#### Scenario: A small outcome is completed
- **WHEN** outcome의 완료 기준을 canonical evidence로 검증한다
- **THEN** OpenSpec tasks는 해당 결과를 완료하고 evidence owner를 참조하지만 누락된 downstream 학습·실물 효과나 승인을 완료로 간주하지 않는다

#### Scenario: A verified integration checkpoint is ready to publish
- **WHEN** 독립적으로 통합 가능한 변경을 검토하고 실제 결합된 source cutoff에 필요한 검증을 완료한다
- **THEN** coordinator는 사용자 변경과 원본을 보존하며 해당 체크포인트를 main에 commit·push하고, 실제 원격 main의 commit이 일치하는지 확인한다
- **AND** 전체 장기 Goal의 완료까지 공개를 미루지 않되 미검증 변경·비밀·무거운 데이터·run state·로컬 에이전트 도구를 함께 올리거나 원격 이력을 강제로 덮어쓰지 않는다
- **AND** push 실패나 원격 진전은 Orca에 정확한 commit·검증 범위·미반영 사유를 남겨 해결하며, 독립적인 적격 작업을 멈추지 않는다
- **AND** source 공개는 실행 중인 프로세스·드라이버의 배포, 실물 검증 또는 데이터·학습 승인을 뜻하지 않는다

### Requirement: Existing product interfaces precede custom capabilities
각 lane은 성숙한 상용·오픈소스 제품이 제공하는 범용 기능을 경쟁적으로 재구현하기보다 지원되는 인터페이스로 재사용하는 경로를 우선 검토해야 한다(SHALL). 도입·교체·확장 전에 실제 소비자가 필요한 입출력·단위와 시간 의미·lifecycle owner·오류 및 evidence 전달 계약을 기존 책임 경계에 맞춰 필요한 수준으로 먼저 정리해야 한다(SHALL). 자체 구현은 해당 인터페이스가 충족하지 못하는 FR5 고유 요구와 검증된 공백에 한정하며(SHALL), 기존 안전·승인·원본 보존 책임을 외부 제품 사용으로 생략하거나 별도 정본과 owner로 복제해서는 안 된다(MUST NOT).

#### Scenario: An existing product can satisfy a required capability
- **WHEN** 현재 제품 목표에 필요한 기능을 외부 제품의 지원 인터페이스로 소비할 수 있다
- **THEN** 실제 설치 버전·source/tests와 필요한 공식 근거를 대조해 계약의 적합성을 확인하고, 기존 owner를 보존하는 최소 연결로 소비한다
- **AND** 재사용 여부는 실제 소비 경로와 유지보수·운용 비용으로 판단하며, 연결 선언만으로 실행·검증 완료를 주장하지 않는다

#### Scenario: Reuse leaves a specific gap or delays the critical outcome
- **WHEN** 외부 인터페이스가 필요한 의미·권한·evidence를 보존하지 못하거나 교체 비용이 현재 핵심 결과에 주는 가치보다 크다
- **THEN** 근거가 있는 FR5 책임만 최소 구현하거나 기존 경로를 유지하고, 도입·보류 이유를 해당 lane의 설계 근거에 남긴다
- **AND** 이 원칙을 이유로 범용 wrapper·새 서비스·추측성 미래 인터페이스를 만들거나 가장 가까운 collect → train → rollout → diagnose → recollect의 실제 증명을 지연시키지 않는다

### Requirement: Learning evidence analysis stays separated from authority
Data Quality Analysis와 Rollout Evidence Analysis는 각자 canonical output을 가져야 한다(SHALL). Recommendation은 두 결과를 읽어 advisory synthesis만 제공해야 하며(MUST), recorder·motion·collection·promotion·training·publication authority를 가져서는 안 된다(MUST NOT).

#### Scenario: Recommendation proposes a next action
- **WHEN** 두 분석 owner의 evidence가 recommendation에 전달된다
- **THEN** recommendation은 근거와 제한을 제시하지만 어떤 runtime 또는 승인 상태도 변경하지 않는다

### Requirement: Policy artifacts and evaluation comparisons have distinct owners from execution
정책 학습·평가 owner는 승인된 데이터에서 만들어지는 checkpoint와 저장 processor의 canonical 계약을 소유해야 하며(SHALL), resume·offline evaluation·Rollout은 그 검증을 재사용해야 한다(SHALL). 같은 owner가 비교 질문·고정 cohort·예산·지표 집계를 책임지고, Rollout은 기존 single execution owner를 통해 trial별 실행·시각·중단·결과 근거를 생산해야 한다(SHALL). Offline loss, 제어 완료, semantic 성공과 데이터 유용성을 서로 대신하는 지표로 사용해서는 안 된다(MUST NOT).

#### Scenario: A policy result informs collection
- **WHEN** 정책 평가 결과가 다음 수집 조건 선택에 사용된다
- **THEN** 비교 조건과 실제 trial 근거를 추적할 수 있고 성공·실패 데이터 모두의 유용성을 검토한다
- **AND** Curation의 기존 데이터 선별, 다음 데이터 획득 전략과 실제 수집 실행은 각 결정의 owner를 유지한다

### Requirement: Evaluation and execution balance reuse with independent responsibilities
온라인·오프라인의 평가와 실행은 지원하는 정책별 checkpoint·저장 processor 검증, 결정적 관측 처리, 추론과 행동 단위·의미 해석 중 같은 의미를 가진 기능을 기존 owner에서 재사용해야 한다(SHALL). 재사용 범위는 평가 함수뿐 아니라 실제 정책 계산 경로를 포함해야 하며(SHALL), 학습 손실·open-loop 출력 비교·실물 제어처럼 다른 의미의 계산이나 lifecycle을 단일 경로로 억지로 합쳐서는 안 된다(MUST NOT). 저장 관측의 오프라인 실행, 실시간 관측의 명령 없는 실행과 허가된 실물 실행은 입력 출처·시간 의미와 효과 권한을 구분해야 하며(SHALL), 공통 코어를 사용한다는 이유로 과거 관측을 fresh로 표시하거나 로봇 명령 권한을 부여해서는 안 된다(MUST NOT). 이 요구는 새 범용 harness·서비스·모델 registry를 만들라는 뜻이 아니다.

공유와 분리는 일관성뿐 아니라 독립적인 변경·검증, 자원 사용, 실패 복구 및 전체 개발·운용 비용으로 판단해야 한다(SHALL). 서로 다른 결정·출력·lifecycle을 소유하는 기능은 그 책임을 독립적으로 개발·사용·검증할 수 있게 분리해야 하며(SHALL), 호출 모양이 비슷하다는 이유만으로 공통 계층에 결합하거나 폴더 수를 늘리는 것 자체를 개선으로 간주해서는 안 된다(MUST NOT).

#### Scenario: Stored observations exercise the same policy computation
- **WHEN** 저장 관측과 검증된 checkpoint로 오프라인 추론 또는 실행 재생을 수행한다
- **THEN** 온라인 경로와 같은 저장 전처리·추론·후처리 구현을 소비하고, 비교에 필요한 관측·checkpoint·processor·설정 및 stochastic 입력의 계보를 보존한다
- **AND** 오프라인 결과를 실제 명령·측정된 로봇 동작·작업 성공으로 표시하지 않으며, 원본 dataset과 실행·학습 승인 상태를 변경하지 않는다

#### Scenario: A policy proposal reaches authorized hardware
- **WHEN** 같은 정책 계산 결과를 실제 로봇에 적용한다
- **THEN** 기존 실행 owner가 현재 관측·시작 상태·scene·cell·exact plan 및 단일 motion 조건을 확인하며, 예측 원본·실제로 소비한 구간·전송 명령·관측된 동작과 중단 사유를 구분해 추적할 수 있다
- **AND** 모델 출력을 맞추기 위해 조용히 자르거나 반올림하거나 시간 의미를 바꾸지 않으며, 명시적 실행 변환이 있다면 비교 조건과 결과에 그 차이를 보존한다

#### Scenario: Reuse is verified through real consumers
- **WHEN** 공통 실행 코어의 연결 완료를 검증한다
- **THEN** 저장 관측 소비와 온라인 adapter가 같은 구현을 사용하는 정상 경로 및 잘못된 입력·중단 경로를 검증하고, 오프라인 재생·합성 transport·실물 실행의 증거 범위를 각각 명시한다
- **AND** 공통 인터페이스 선언이나 별도 fake 구현의 통과만으로 실제 정책·하드웨어 연결 완료를 주장하지 않는다

### Requirement: Immutable and human gates fail closed
기존 hardware, human, scene, cell, plan-digest, semantic, physical-binding, training-authorization gate는 서로 분리되어 유지되어야 한다(SHALL). 누락되거나 `PARTIAL` 또는 `UNKNOWN`인 evidence는 어떤 외부 효과도 허가해서는 안 된다(MUST NOT).

#### Scenario: Required gate evidence is missing
- **WHEN** downstream 작업에 필요한 gate evidence가 없거나 검증되지 않았다
- **THEN** 해당 외부 효과는 차단되고 다른 gate의 PASS가 이를 대신하지 않는다

### Requirement: Shared scene knowledge preserves evidence and consumer authority
Collection과 Rollout은 기존 scene owner에서 물체 상태·위치·근거·유효 범위를 공유해야 하며(SHALL), 평가와 Curation은 해당 실행에 결속된 scene snapshot을 소비할 수 있어야 한다(SHALL). 소비자는 별도 위치 정본을 만들거나 과거 snapshot을 현재 상태로 덮어써서는 안 된다(MUST NOT). 제어기 관측, 물체 상태, 충돌 환경과 작업 성공 판정은 서로 다른 책임을 유지해야 한다(SHALL).

검증된 수집 레시피의 실행 기반 놓기 계보는 기존 계약에 따라 다음 시작 위치로 재사용해야 하며(SHALL), 학습 정책의 명령 묶음 완료만으로 물체 착지·목표 도달·semantic 성공을 갱신해서는 안 된다(MUST NOT). 새로운 비전 프로세스나 사람 확인 절차를 공통 조회의 필수 조건으로 추가해서는 안 되며(MUST NOT), 필요한 측정의 도입은 실제 정보 공백과 적용 범위에 근거해야 한다(SHALL). 별도 scene-understanding 패키지 분화는 실제 소비자·변경·검증 경계와 중복 감소로 판단해야 하며(SHALL), 이 요구는 새 저장소나 서비스의 생성을 요구하지 않는다.

#### Scenario: Consumers share one supported scene snapshot
- **WHEN** Collection 또는 Rollout이 현재 scene을 사용하거나 평가·Curation이 과거 실행을 분석한다
- **THEN** 기존 owner의 revision·근거와 실행 결속을 보존하고, 현재 실행용 상태와 과거 분석용 snapshot을 구분한다
- **AND** 분석 소비가 현재 scene이나 motion·semantic·training authority를 변경하지 않는다

#### Scenario: A learned command chunk completes without placement evidence
- **WHEN** 제어기는 명령 완료를 보고하지만 적용 가능한 물체 놓기 근거가 없다
- **THEN** 완료된 로봇 동작은 기록하되 목표 물체 위치나 작업 성공을 만들어내지 않는다
- **AND** 해당 물체 상태를 필요로 하지 않는 독립적인 적격 작업은 계속할 수 있다

### Requirement: Continuation preserves completed physical effects
시스템은 검증된 실행의 위치 계보를 기존 scene owner에서 이어받아야 하며(SHALL), UI·분류·저장 실패만으로 완료된 놓기 동작을 재실행하거나 그 위치 증거를 폐기해서는 안 된다(MUST NOT). 새 동작은 현재 scene·cell·exact plan과 단일 motion owner의 조건을 계속 충족해야 한다(SHALL). 물리 상태를 알 수 없는 실패와 후처리 실패는 서로 다른 복구 대상을 가져야 한다(SHALL).

#### Scenario: Postprocessing fails after a completed placement
- **WHEN** 실행 완료와 놓기 계보는 유효하지만 독립적인 후처리가 실패한다
- **THEN** 해당 처리만 복구 대상으로 남기고 기존 위치 증거와 원본을 보존하며, 완료한 동작을 자동 재실행하지 않는다
- **AND** 다음 episode가 필요로 하는 미완료 저장·기술검사 또는 물리 gate를 우회하지 않는다

#### Scenario: The work surface is dark before new motion
- **WHEN** 이동 전 fresh 준비 구간에서 작업대 카메라가 어둡거나 필요한 밝기 증거를 확인할 수 없다
- **THEN** 새 motion을 시작하지 않고 준비 녹화만 정리하며, 이전 완료 episode의 물리 결과나 semantic·training 상태를 변경하지 않는다
- **AND** 이 시작 조건을 동작 중 소등 감지나 일반 물체 인식의 증명으로 사용하지 않는다

### Requirement: Handoff preserves claim lineage
lane handoff는 claim, evidence state, exact reference 또는 digest, 적용 범위와 known limitation, next owner를 포함해야 한다(SHALL). 각 사실은 기존 canonical owner의 output을 참조해야 하며(MUST), 별도 ledger나 복제된 truth를 만들어서는 안 된다(MUST NOT).

#### Scenario: A downstream lane consumes evidence
- **WHEN** 다음 lane이 upstream 결과를 받는다
- **THEN** 소비자는 claim에서 canonical output과 evidence, limitation, owner까지 끊김 없이 추적할 수 있다

### Requirement: Portfolio communication demonstrates job-relevant capability
Portfolio 표현 책임자는 모방학습과 데이터 엔지니어링 직무의 외부 검토자가 실제 역량을 확인할 수 있도록 연구 결과와 시스템 구현의 가치를 함께 발견·검증·표현해야 한다(SHALL). 기존 프로젝트 연구, 필요한 외부 우수 사례와 실제 source/tests/runtime을 직접 소비하며 사례·서사·매체를 자율적으로 선택하고, 산출물을 독자의 관점에서 사용·렌더링·검토해 개선해야 한다(SHALL). 아키텍처와 SSOT 등의 설계 선택은 관련성이 있을 때 실제 문제·trade-off·동작 근거로 설명하며, 기능 목록이나 좋은 원칙의 나열만으로 가치가 입증됐다고 간주해서는 안 된다(MUST NOT). 이 책임은 제품 UX나 기술 owner의 canonical 판단·출력을 대체하지 않는다(MUST NOT).

#### Scenario: The lane chooses what to communicate
- **WHEN** 표현 책임자가 다음 portfolio 결과를 선택한다
- **THEN** 직무 관련성, 실제 증거와 설명의 설득력으로 선택을 정당화하고, 미리 지정된 항목·템플릿의 소진이나 문서 수 증가를 완료 기준으로 삼지 않는다
- **AND** 전체 확장 전에 작은 실제 표현물로 사람의 취향 피드백을 받을 수 있게 하되 독립적인 기술 작업을 멈추지 않는다

### Requirement: Portfolio technical explanations grow from investigated design intent

Portfolio 표현 책임자는 핵심 기술을 새로 소개하거나 설명의 의미를 바꾸기 전에, 해결하려는 문제·설계 의도·실제 작동 원리와 기대한 이점을 조사해야 한다(SHALL). main MEX를 현재 맥락과 근거 경로의 출발점으로 사용하되 유일하거나 최종적인 진실원으로 간주해서는 안 된다(MUST NOT). 사용자 의도와 설계·변경 기록, 실제 호출 코드와 입력 계약, 관련 테스트·canonical 산출물·운용 관찰을 필요한 깊이로 대조하고, 기술적 해석에 필요한 외부 일차 자료를 활용해야 한다(SHALL). 과거 의도, 현재 구현, 관측된 효과와 표현 책임자의 추론은 구분해야 한다(SHALL).

글과 그림은 기술이 해결하려는 문제와 선택한 방법의 관계를 독자가 이해하도록 구성해야 한다(SHALL). 기능·수식·주의사항의 나열이나 오독 방지 문구만으로 설계 의도의 설명을 대신해서는 안 된다(MUST NOT). 상세 계산·조건·근거는 해당 의도를 뒷받침하도록 배치하고, 그림의 대상·좌표계·범위와 연결선이 실제 메커니즘을 표현하는지 렌더·사용으로 검토해야 한다(SHALL). SSOT·JSON 계약을 표현할 때는 원본 필드를 모두 나열하기보다, 보존되는 의미와 참조 관계·검증·실제 소비자를 추상화해 보여 주어야 한다(SHALL). 외부 시스템 연동의 확장 가치는 그 기반이 되는 현재 계약과 함께 설명하고, 아직 없는 소비자를 구현된 인터페이스로 표시해서는 안 된다(MUST NOT). 이 원칙은 공통 페이지 템플릿이나 매 편집의 승인 절차를 요구하지 않는다.

#### Scenario: An explanation is prepared from project knowledge

- **WHEN** 표현 책임자가 핵심 알고리즘이나 아키텍처를 설명할 준비를 한다
- **THEN** MEX의 요약에서 원래 의도와 실제 소비 경로를 따라가고, 입력 변화·다른 지원 조건·반례를 통해 설명하려는 일반화가 성립하는지 확인한다
- **AND** 한 예시나 테스트 PASS를 기술 전체의 성능으로 확대하지 않으며, 조사 결과를 목적이 읽히는 실제 표현물로 발전시킨다

#### Scenario: Investigation reveals a missing mechanism or contradiction

- **WHEN** MEX·설계 기록·코드·실제 산출물의 대조에서 누락, 불일치 또는 계산 반례를 발견한다
- **THEN** 표현 책임자는 자신의 설명을 바로잡고, 재현 입력과 고정 근거·영향 범위를 기존 main 또는 기술 owner에게 전달한다
- **AND** 기술 owner의 원본을 직접 바꾸거나 미검증 수정을 완료 성과로 반영하지 않으며, 관련 없는 표현 작업은 계속한다

#### Scenario: A technically correct figure obscures the purpose

- **WHEN** 실제 렌더와 독자 피드백에서 도형·용어·배치가 설계 의도보다 계산 중간값이나 다른 기능을 먼저 떠올리게 한다
- **THEN** 설명 대상과 시각적 구성을 다시 선택해 무엇을 위해 어떻게 동작하는지 드러내고, 필요한 세부는 그 설명에서 탐색할 수 있게 한다
- **AND** 기존 그림의 유지나 주의 문구 추가만을 해결로 간주하지 않는다

### Requirement: Portfolio feedback improves the reader experience while preserving core value

표현 책임자는 피드백을 개별 수정 지시로만 소진하지 않고, 독자의 이해를 방해한 원인을 찾아 다음 편집에도 적용해야 한다(SHALL). 다음 다섯 원칙으로 개선안을 판단해야 한다(SHALL).

1. **핵심 가치 유지.** 데이터의 라이프사이클을 연결하고 정책 개선을 검증하려는 폐루프 목적, 개별 기술의 설계 의도와 실제 기여를 보존한다. 최근 결과나 한 화면의 취향만으로 프로젝트 전체를 축소하지 않는다. 새로운 근거와 명시적 사용자 의도에 따라 가치의 강조점은 발전시킨다.
2. **독자가 해독하지 않아도 되는 구성.** 첫 화면과 각 구간에서 시스템의 목적·해결하는 문제·방법·기대한 이점을 빠르게 드러낸다. 전체 구조에서 개별 원리로 들어가 다시 시스템 동작과 가치로 연결한다. 중요한 아키텍처·분기·피드백·모듈 책임은 본문에 드러내며, 설명이 어렵거나 화면이 복잡하다는 이유로 핵심을 접힘 안에 숨기지 않는다. 필드·계산·원본 발췌는 핵심 설명을 뒷받침하는 위치에서 탐색하게 한다. task·skill·policy 등은 문헌의 맥락과 실제 구현을 대조해 정의하고 레포·포트폴리오에서 일관되게 사용한다. 공통 템플릿이나 고정 페이지 수를 강제하지 않는다.
3. **이해를 대신 수행하는 시각화.** 문장을 상자에 옮기거나 모듈을 화살표로 나열하는 수준에 머물지 않는다. 계층·입출력·병렬 경로·되돌아오는 피드백과 조건의 유지·변화가 그림 자체에서 읽혀야 한다. 서로 다른 관계를 같은 선이나 도형으로 혼동시키지 않는다. 화살표의 양 끝 여백·분기 접점·꺾임 공간과 라벨 간격을 실제 발표 크기에서 확인하며, 신호의 연속 변화·명령 유지·표본 선택은 서로 다른 형상으로 표현한다. 폐루프의 화살표는 반복 단위와 되돌아가는 상태를 검증한다. 같은 계획의 다음 시연 실행과 피드백을 반영한 다음 수집 계획을 구분하고, 실행 관측·평가·다음 입력 사이의 연결을 생략하지 않는다. 회전·이동·비교는 차이가 눈에 띄는 대상과 구도로 표현하며, 효과가 보이지 않는 인터랙션은 제거한다. 실제 환경의 복제보다 원리를 드러내는 설명용 형상과 강조를 선택할 수 있으나, 실제 조건·계산·구현·성과를 왜곡하지 않는다.
4. **독자의 읽기 부담을 먼저 줄이는 편집.** 영상의 명백한 동작, 그림의 흐름, 제목·범례·버튼·링크가 이미 전달하는 내용을 문장으로 반복하지 않는다. 삭제해도 목적·원리·조건·결과 해석에 필요한 의미가 남으면 그 문장을 삭제한다. 같은 역할의 그림·표·구간은 통합하거나 설명 역할을 다시 나누며, 삭제로 충분한 내용을 새 시각물로 중복 제작하지 않는다. 번역투·질문형 소제목·당연한 절차 설명·추상적인 가치 문구는 구체적인 기술 표현으로 고친다. 제목과 본문의 기준점은 외부 독자가 이해할 기능·설계·결과이다. “새 정책까지”, “최신 작업 완료”처럼 작성자의 로컬 진행을 알아야 이해되는 문구는 공개 설명으로 사용하지 않으며, 날짜·revision·실험 이력은 결과 해석에 필요한 위치로 모은다. 외부 독자가 내부 상태를 모른다고 가정하고 문제·설계 판단·검증 범위를 드러낸다. 구현 식별자와 권한·진행 관리 용어는 근거 안쪽으로 옮기고, 자기평가형 형용사와 관찰을 인과로 확대하는 표현을 피한다. 처음 쓰는 핵심 용어는 필요한 만큼만 정의하며, 이 원칙을 고정 문단 순서나 설명문 추가의 의무로 만들지 않는다. 그림이 전달하는 의미에는 설명문을 덧붙이지 않는다. 단위·비교 조건·측정 결과와 목표·구현의 구분은 보존한다.
5. **사용자가 재지적하기 전에 고치는 검토.** 여러 우수 논문·레포·포트폴리오의 내용 구성과 표현을 대조하고, 실제 렌더를 처음 보는 외부 독자 시점으로 검토한다. 작성자가 코드를 알아서 이해되는 것을 통과 기준으로 삼지 않는다. 시선의 순서·용어와 계층의 혼동·선의 실제 의미·비교의 차이·삭제 가능한 텍스트를 확인하고 정렬·줄바꿈·간격·선과 라벨 충돌까지 해결한다. 중요한 구간은 직접 열고 조작하고 근거까지 따라가며 정확성·이해 가능성·시각적 마감을 각각 검토한다. 자동 검사 PASS와 파일 생성만으로 완성을 주장하지 않는다. 피드백 요소뿐 아니라 같은 결함이 있는 다른 레포·포트폴리오 구간도 찾아 수정하며 반복 원인은 기존 원칙에 반영한다. 작은 시안으로 방향을 드러내되 매번 승인을 기다리지 않고 중요한 결함이 해결된 완성본까지 이어간다. 작은 수정량이나 당장의 비용을 완성 기준으로 삼지 않는다. 장기적인 이해·설득·유지 효과를 먼저 판단하고, 이를 높이는 데 필요한 조사·재구성·시각화·검토를 충분히 수행한다. 자원·권한 경계는 지키며 의미 없는 반복 조사·픽셀 수정은 하지 않는다.

#### Scenario: Feedback identifies a local presentation problem

- **WHEN** 독자가 중복된 자료, 과도한 글, 어색한 용어 또는 효과가 약한 그림·조작을 지적한다
- **THEN** 표현 책임자는 같은 문제가 있는 관련 구간도 살펴보고 핵심 가치와 사실을 보존하는 범위에서 수정한다
- **AND** 실제 화면에서 필요한 의미가 더 쉽게 읽히는지 확인한다. 삭제만으로 해결되면 설명이나 주의 문구를 대신 덧붙이지 않으며, 기술 원리·비교 조건·반례·구현과 목표의 구분은 보존한다

#### Scenario: An illustrative example makes a mechanism easier to understand

- **WHEN** 실제 시연의 물체나 조건으로는 기술이 의도한 차이가 잘 보이지 않는다
- **THEN** 설명용 조건으로 원리를 선명하게 보여 주고 그 조건에서도 설명과 실제 계산이 일치하는지 확인한다
- **AND** 설명용 예시를 새로운 실물 검증이나 성능 개선의 증거로 제시하지 않는다

#### Scenario: A presenter follows the main story in about twenty minutes

- **WHEN** 발표자가 포트폴리오의 기본 열람 흐름을 따라 설명한다
- **THEN** 데스크톱 기본 화면에서 세로 스크롤 없이 핵심 화면을 ‘다음’으로 넘기며 약 20분의 발표 안에서 프로젝트의 목적·핵심 설계·기술 간 관계·확인된 결과를 연결할 수 있다
- **AND** 각 화면의 비중은 독자가 프로젝트의 가치를 이해하는 데 필요한 정도로 판단하며, 기반 기술이나 운영 설계를 일괄적으로 낮은 비중에 두지 않는다
- **AND** Demonstration Design·Learning Evidence·Physical Rollout은 가치 탐색의 참고 축으로 사용한다. 이를 필수 장·고정 순서·페이지 경계로 강제하지 않으며, 외부 독자가 왜 만들었고 무엇을 확인했는지 설명할 수 있는 흐름에 맞춰 통합·분리·순서를 조정한다
- **AND** 가치의 이름에 내용을 끼워 맞추거나 같은 분량을 배정하지 않는다. 숫자의 크기·그림의 면적·반복 횟수가 실제 의미보다 강한 성과를 암시하는지 검토하며, README와 포트폴리오에서 고유 설계가 빠지거나 기반 기술이 과대 대표되는 구간을 함께 바로잡는다
- **AND** 주의를 분산시키는 계산·필드·운영 기록은 관련 화면에서 별도로 탐색하고, 결과 해석에 필수적인 비교 조건과 미검증 범위는 핵심 흐름에 남긴다
- **AND** 반복과 부연을 줄이는 작업과 함께, 깊이 묻힌 구현·실험·설계 중 프로젝트의 고유 가치를 설명하는 내용을 찾아 전면 배치를 검토한다
- **AND** 화면 수와 공통 세부 템플릿은 고정하지 않으며, 발표 시간은 실제 설명·시연·화면 전환을 포함한 흐름으로 검토한다
- **AND** 한 화면에 맞추기 위해 핵심 그림·글자를 읽기 어렵게 축소하지 않으며, 설명 역할을 나누어 전면 화면을 늘릴 수 있다. 확대·좁은 화면에서는 내용을 잘라내지 않고 탐색 가능하게 유지한다

#### Scenario: An external reader opens the published portfolio

- **WHEN** 저장소나 전달 안내를 외부 독자의 포트폴리오 진입점으로 제공한다
- **THEN** 그 매체에서 그림과 설명이 실제로 렌더되고, 독자가 작성자의 작업 폴더·계정·개발 환경 없이 열람 또는 다운로드 경로를 끝까지 사용할 수 있는지 확인한다
- **AND** 파일 존재·링크 문법·작성자 환경의 렌더 성공만으로 배포 준비를 완료하지 않으며, 시연·기술적 가치·결과를 먼저 배치하고 생성·설치·운영 상세는 필요한 독자가 탐색하도록 둔다

### Requirement: Portfolio evidence evolves without becoming a second truth
검증된 성과와 미검증 가설은 구분하되 새 증거가 나오면 같은 설명과 근거 연결을 갱신할 수 있어야 한다(SHALL). 문서·시각화·발표 자료의 수치와 주장은 기존 canonical evidence에서 추적 가능해야 하며, 독립적인 결과 장부나 중복된 수작업 정본을 만들지 않아야 한다(SHALL). 시각적 완성도는 실제 렌더 결과의 가독성·비교 가능성·정직한 범위 표현을 포함해야 하며, 장식이나 유리한 사례 선택으로 한계를 숨겨서는 안 된다(MUST NOT).

#### Scenario: New results support or refute an earlier hypothesis
- **WHEN** 새로운 canonical evidence가 기존 가설이나 공개 후보 주장을 지지·제한·반박한다
- **THEN** 해당 설명과 시각적 근거를 함께 갱신하고 과거 비교의 적용 범위를 보존하며, 같은 수치나 서사를 매체마다 별도로 재작성하도록 강제하지 않는다
- **AND** 미검증 미래 결과를 현재 성과로 선반영하지 않는다

#### Scenario: Editorial restructuring overlaps newer main documentation

- **WHEN** 오래된 worktree의 문서·포트폴리오를 재구성하는 동안 main에 기술 계약이나 소비 경로가 추가된다
- **THEN** 공통 기준점 이후 main의 변경을 영향받는 모든 문서와 OpenSpec 계약에 대조하고, 새 계약·근거·명령을 새 정보구조 안에 보존한다
- **AND** 본문의 반복 설명은 줄이되 기술적 의미를 삭제하지 않으며, 시작 시 검사·각 실행 검사·연속 감시처럼 적용 시점이 다른 보장은 실제 호출 범위로 표현한다

### Requirement: Work and intent evolve only on evidence
각 lane의 작업은 다음 lane unblock, 중요한 불확실성 또는 실패 원인 감소, 재현 가능한 evidence 강화, 외부에서 확인 가능한 portfolio proof 생성 중 적어도 하나를 충족해야 한다(SHALL). OpenSpec revision은 새 evidence가 지속 가능한 행동·authority·acceptance·handoff 의미를 흔들 때만 제안해야 한다(SHALL).

중요한 portfolio 가치 선택은 기존 프로젝트 연구, 판단 시점의 관련 최신 primary research, 실행 가능한 source/tests와 현재 PC·FR5 제약을 비교해 반증 가능한 질문과 최소 실험을 정해야 한다(SHALL). 연결·자동화·기능 개수만으로 연구에 부합하는 학습 가치를 달성했다고 간주해서는 안 된다(MUST NOT). 선택한 실험은 비교 기준, 자원 비용, 실제 관측 결과와 한계를 남겨 외부인이 결론을 검토할 수 있어야 한다(SHALL). 특정 논문의 알고리즘이나 성공 결과를 재현 전의 FR5 성능으로 전제해서는 안 된다(MUST NOT). 과거 노트의 보류 항목은 영구 금지가 아니며 현재 근거와 비용으로 다시 판단해야 한다(SHALL).

#### Scenario: A connected loop has not established learning value
- **WHEN** 수집부터 재평가까지 실행은 연결됐지만 데이터 선택의 유용성이나 중요한 실패 원인의 비교 근거가 없다
- **THEN** 연결은 enabling evidence로만 표시하고, 연구 질문을 검증할 가장 작은 실험을 다음 결과로 선택한다
- **AND** 개선이 없거나 부정적인 결과도 비용·적용 범위·불확실성과 함께 보존하며 유리한 사례만 공개하지 않는다

#### Scenario: Candidate work has no evidence value
- **WHEN** 구현·분석·문서·추상화가 네 가지 가치 조건을 하나도 충족하지 않는다
- **THEN** 작업은 보기 좋은 기능이라는 이유만으로 실행되지 않고 defer된다

#### Scenario: An existing product overlaps a proposed capability
- **WHEN** 기존 제품이 만들려는 기능 또는 이미 구현한 기능과 실질적으로 겹친다
- **THEN** 책임자는 현재 환경에서 요구 충족과 전체 비용을 근거로 재사용·최소 연결·자체 구현을 비교하고, 기존 코드를 보존하거나 새 도구를 도입하는 것 자체를 목표로 삼지 않는다
- **AND** 공개 주장은 재사용한 기능, 프로젝트가 추가한 동작과 검증한 효과를 구분하며, 기능 연결이나 기반 안정성을 학습 성능 개선의 증명으로 사용하지 않는다
- **AND** 대체 또는 연결은 원본·계보와 기존 실행 authority를 보존하며, 특정 제품 선택을 위해 프로젝트의 필수 결과를 축소하지 않는다

#### Scenario: Comparison results revise the research hypothesis
- **WHEN** 비교 실험, 교차 검토 또는 실제 환경 evidence가 선택한 가설의 전제나 예상 효과를 흔든다
- **THEN** 책임자는 단순한 기준선과 경쟁 가설을 다시 비교하고 유지·수정·폐기 근거 및 다음 선택을 바꿀 수 있는 불확실성을 명시한다
- **AND** 필요한 범위의 primary research를 다시 조사해 다음 유한한 실험을 선택하며, 한 번의 문헌 조사나 사용자 제안으로 구현 방향을 영구 고정하지 않는다
- **AND** 미검증·환경 부적합·실험으로 반증됨을 구분하고, 반복 조사 자체나 새 실험 관리 계층을 만드는 것을 성과로 간주하지 않는다

#### Scenario: Training feasibility has already been established
- **WHEN** 실제 학습·저장·재로딩·평가와 자원 적합성이 해당 입력 및 runtime에서 확인됐다
- **THEN** 후속 실험은 남은 학습·데이터·실물 성능 질문과 유한한 예산, 비교 범위, 종료 후 다음 결정을 실행 전에 정하며 동일한 feasibility 확인만을 반복하지 않는다
- **AND** 데이터 보강은 알려진 조건별 공백과 수집 가치에 따라 독립적으로 준비할 수 있고 모든 학습 설정 비교의 완료를 선행 조건으로 요구하지 않는다
- **AND** 검증 손실, 실행 가능한 checkpoint 및 시연 생성기의 성공은 학습 정책의 실물 작업 성능을 대신하지 않으며, 평가와 준비를 포함한 전체 비용으로 실험 가치를 판단한다

#### Scenario: Research findings improve subsequent acquisition and data use
- **WHEN** 수집·선별·학습이 진행되는 동안 새로운 연구 또는 실험 결과가 다음 데이터의 조건·구성·관측 표현을 바꿀 가치가 있다
- **THEN** 해당 owner는 외부에서 보고된 효과와 현재 환경에서 검증한 효과를 구분하고, 기대 효용·적용 비용·기존 데이터 재사용 가능성을 근거로 다음 수집·선별·학습 비교에 반영하거나 반영하지 않는 이유를 정한다
- **AND** 이미 승인된 진행 중 실행의 입력·계획을 소급 변경하지 않고, 변경된 조건은 다음 적절한 실행 경계에서 기존 owner와 native admission을 통해 적용하며, 관련 없는 연구나 구현 완료를 유효한 수집의 선행 조건으로 삼지 않는다
- **AND** 원본과 기존 판정·분할을 보존하고 파생본·제외·혼합 선택은 기존 request와 provenance로 추적하며, 후속 비교가 가설을 반박하면 구성과 공개 설명도 그 근거에 맞춰 수정한다

#### Scenario: New evidence challenges an intent boundary
- **WHEN** 조사 가능한 engineering unknown을 해소한 뒤에도 가치·안전·의미에 관한 선택이 남는다
- **THEN** 기존 Goal과 acceptance를 자동 변경하지 않고 bounded human decision을 요청한다

### Requirement: Lane owners evolve bounded outcomes autonomously
lane 책임자는 기존 프로젝트 연구, 중요한 판단에 필요한 최신 primary evidence, 실제 source/tests와 PC·물리 runtime 제약을 비례적으로 검토하고, 제품 가치가 높은 발전 가설을 선택해 완료 가능한 작은 결과로 구현·검증해야 한다(SHALL). 미리 정한 작은 기능에 연구 인용만 덧붙이거나 조사 보고서만으로 제품 발전을 완료해서는 안 된다(MUST NOT). 할당된 결과·자원·쓰기 경계 안의 engineering 선택에는 반복적인 사람 승인을 요구하지 않아야 한다(SHALL). 공동 계약이나 owner 간 책임 변경은 영향받는 소비자와 완료 기준을 명시해 coordinator가 통합 조율해야 하며(SHALL), lane의 Goal이나 미합의 OpenSpec 초안을 실행 권한으로 해석해서는 안 된다(MUST NOT).

#### Scenario: A lane selects its next improvement
- **WHEN** lane의 dogfooding 또는 새 근거가 발전 후보를 드러낸다
- **THEN** 책임자는 실제 다음 소비자, 반증 가능한 완료 기준, 자원 범위와 한계를 정해 할당된 경계 안에서 진행하고, 결과는 canonical evidence로 전달한다

#### Scenario: A useful idea crosses a shared boundary
- **WHEN** 선택한 개선이 다른 owner의 계약·source 또는 공유 실행 자원에 영향을 준다
- **THEN** coordinator가 충돌 범위와 소비자 검증을 조율하며, 미합의 효과만 보류하고 독립적인 안전 작업은 계속한다

### Requirement: Rollout feedback tests data needs rather than assuming them
rollout 분석은 관측·추론 지연, 제어·실행 실패와 데이터의 가시성·분포 부족을 근거에 따라 구분하고, 추가 데이터가 필요한 경우 반증 가능한 수집 가설과 평가 조건을 다음 제품 소비자에게 전달해야 한다(SHALL). 제품은 선택한 가설을 적격 수집·선별·승인된 학습·재평가로 연결해야 하며(SHALL), 상관관계나 recommendation 출력만으로 데이터 부족의 원인 또는 정책 개선을 주장해서는 안 된다(MUST NOT). 개선 주장은 비교 가능한 수집·학습 비용, 고정된 held-out 평가 조건, 입력·정책 계보와 불확실성을 포함해야 한다(SHALL).

#### Scenario: A failed rollout suggests targeted collection
- **WHEN** 실패 evidence가 특정 조건의 데이터 부족을 가설로 지지한다
- **THEN** 해당 조건과 근거가 기존 수집 owner에 소비되고, 선택된 실행과 재평가를 추적하며, 가능한 동일 비용의 비표적 수집 또는 기존 데이터 기준선과 비교하기 전에는 개선 효과를 UNKNOWN으로 유지한다

#### Scenario: Failure is attributable to runtime rather than data
- **WHEN** 유효한 관측에서 출발했지만 추론·전송 지연이나 제어 실패가 실행을 무효화한 근거가 있다
- **THEN** 해당 runtime 교정과 같은 조건의 재검증을 준비하고 추가 데이터 수집을 필수 해결책으로 단정하지 않는다

#### Scenario: Adaptive selection is compared on an untouched evaluation cohort
- **WHEN** 성공·실패 관측이나 validation 결과로 선별 또는 추가 수집을 선택한다
- **THEN** 선택에 사용한 development evidence와 최종 비교용 평가 조건을 구분하고, 최종 평가 결과를 같은 비교의 다음 선택 입력으로 재사용하지 않는다
- **AND** 기존 균형 수집 등 명시적 기준선과 비교하며 수집 수뿐 아니라 실제 수집·학습 시간, reset·중단·사람 개입을 포함한 비용을 보고한다
- **AND** 물리 phase 판정과 runtime 진단이 불확실하면 그대로 남기며, 소표본 결과나 offline loss만으로 일반화된 실물 성능 개선을 주장하지 않는다

#### Scenario: A proposed experiment is selected for discriminating value
- **WHEN** 모델 비교, 표적 수집, 카메라 비교 또는 latent 분석을 다음 실험 후보로 검토한다
- **THEN** 현재 PC·FR5의 비용과 미해결 질문을 기준으로 최소 비교를 선택하며, 제안된 실험 목록이나 특정 모델의 우승을 완료 조건으로 고정하지 않는다
- **AND** 서로 다른 모델의 실용 성능 비교와 같은 checkpoint의 solver 비교를 구분하고, 추가 최적화만의 효과와 데이터 선택 효과를 구분하는 데 필요한 기준선을 사용한다
- **AND** 카메라 입력 손상에 대한 민감도를 별도 단일 카메라 학습의 효용으로, probe의 정보 판독 가능성을 실패 원인의 증명으로, 조건별 수량을 데이터 충분성으로 해석하지 않는다
- **AND** 수집에 사용한 조건과 새 데이터의 포함 범위를 갱신하여 이미 보강한 조건을 계속 미관측 OOD로 주장하지 않으며, 관측한 차이와 인과적 해석을 구분한다

### Requirement: Evidence-leveraged collection may proceed without a renewed scheduling prompt
canonical evidence가 높은 downstream uncertainty-reduction 또는 portfolio/evidence leverage를 보이고 Orca가 기존 production system의 operational availability를 보고하면, coordinator는 추가 human scheduling 또는 availability prompt 없이 collection을 선택하고 시작할 수 있다(MAY). 이 scheduling permission은 timing만 다루며, runtime availability, individual execution, progress, UI/terminal mechanics, blocker는 Orca가 소유한다(SHALL). 이 permission은 hardware, scene, cell, plan-digest, motion lifecycle, recorder lifecycle, semantic, physical-binding, training-authorization 또는 다른 production authority를 생성·대체·충족·우회하지 않는다(MUST NOT); gate가 차단되면 dependent collection effect만 멈추고 독립적인 safe lane은 계속 eligible하다(SHALL).

#### Scenario: High-leverage evidence makes collection schedulable
- **WHEN** canonical evidence가 높은 downstream uncertainty-reduction 또는 portfolio/evidence leverage를 보이고 Orca가 기존 production system을 operationally available로 보고한다
- **THEN** coordinator는 추가 human scheduling 또는 availability prompt 없이 collection을 선택하고 시작할 수 있지만, 모든 기존 production authority와 gate는 그대로 적용되고 blocked gate의 dependent collection effect만 멈추며 독립적인 safe lane은 eligible하게 남는다
