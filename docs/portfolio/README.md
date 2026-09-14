# Closed-Loop Data Engine · 포트폴리오

**[포트폴리오 다운로드](https://github.com/hasemu1211/fr5-lerobot-connector/releases/download/portfolio-2026-09-14/FR5-Portfolio.html)**

[시스템 아키텍처](../architecture.md) · [데이터셋 품질](../dataset-quality.md) · [학습과 평가](../training-and-evaluation.md)

<details>
<summary>편집과 단일 파일 생성</summary>

## 단일 파일 만들기

저장소 루트에서 Python 3으로 실행한다. 추가 패키지는 필요하지 않다.

```sh
python3 docs/portfolio/export_single_file.py .agent-local/portfolio/FR5-Portfolio.html
```

이미지·동영상·글꼴·근거 페이지를 포함해 50MB 미만의 파일을 생성하고, 파일 크기와 SHA-256을 출력한다. 수정은 이 폴더의 원본에 반영한 뒤 같은 명령으로 다시 생성한다.

## 편집 원본

| 위치 | 용도 |
| --- | --- |
| `index.html` 및 주제별 HTML | 본문과 화면 구성 |
| `*.drawio.svg`, `assets/` | 도해·그래프·영상·이미지·글꼴 |
| `sources/` | 실제 코드·데이터·보고서의 출처와 발췌 |
| `reading.js` | 원본 구간을 연결하는 발표 순서·다음/이전·근거 복귀 |
| 그 외 `*.js`, `site.css` | 상호작용·화면 스타일 |
| `export_single_file.py` | 단일 HTML 생성 |

페이지 주소와 자산 경로는 상호작용과 내보내기에서 함께 사용한다. 파일 이동 전 소비 경로를 확인한다. 초안·스크린샷·생성본은 원본 폴더 밖에 보관한다. 외부 자산의 라이선스는 해당 자산과 함께 유지한다.

편집 판단은 [OpenSpec의 다섯 원칙](../../openspec/changes/establish-portfolio-proof-loop-intent/specs/portfolio-proof-loop/spec.md#requirement-portfolio-feedback-improves-the-reader-experience-while-preserving-core-value)을 따른다. 목적과 기술적 기여를 앞세우고 반복 설명을 줄이며, 각 주제에 적합한 매체를 선택한다.

작업명은 **Pick · Pick & Place**, 모듈명은 실제 아키텍처의 **Collection Operator · Recorder · Curator · Training Review · Policy Learning**을 일관되게 사용한다. 한국어는 목적과 원리를 설명하며, 원본 코드·데이터 식별자는 보존한다.

프로젝트의 개발 방향은 **Robot Skill Adaptation**이다. 상위 작업 목표·skill 구성과 실물 적응의 관계는 개요도로, 작업 조건과 실행 원본의 대응은 계약 도해로 드러낸다. task·skill·policy를 고정된 상하 모듈로 취급하지 않는다. README와 포트폴리오는 같은 그림을 재사용한다. 계약은 native 정의·검증·실제 소비에서 추상화하며 새 schema 정본을 만들지 않는다. 상위 시스템의 입력·피드백 소비는 확장 방향으로 구분하고, 구현된 내부 소비와 연결한다.

새 근거는 기존 `sources/` 연결에 반영한다. 설명용 예시, 실제 시연, 오프라인 정책 비교와 폐루프의 목표를 구분하고, 달성한 범위를 넘어 성능을 주장하지 않는다. 전달 전에는 단일 파일에서 바뀐 구간의 가독성·조작·근거 복귀를 직접 확인한다.

</details>
