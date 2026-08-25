# 1. Overview
- project repo for GPU accelerating programming summer camp
- 4 days long kaggle-like competition; competitors are supposed to optimize the speed of MoE inference.

# 2. Project Instructions
- 순차 코드로 구현되어 있는 딥 러닝 추론 프로그램 병렬화 최적화
- 한 개의 계산 노드를 사용 (NVIDIA GeForce TRX 3090 GPU 1장)
- LLM 추론의 Prefill에 해당하는 부분(main.cpp에서 호출되는 generate() 함수)만 최적화
    - 각 input sequence마다 1개의 다음 토큰을 생성
- Pthread, OpenMP, CUDA 사용 가능
- cuDA에 포함된 라이브러리는 사용 불가(cuBLAS, cuDNN 등)
- 이외 외부 라이브러리 사용 불가
- Target model: Phi-tiny-MoE-instruct
- 가능한 수정 예시
    - 메모리 레이아웃 변경
    - 루프 순서 변경
    - 패딩 데이터/연산 추가
    - fusion 등 이 외 연산 최적화
- 불가능한 수정 예시
    - 프로그램 로직 혹은 모델 구조 변경 금지
    - 시간 측정 부분 외에서 모델 추론 연산 수행 금지
    - warmup 등 시간 측정 외 부분에서 주요 연산 결과를 캐싱해두고 이를 시간 측정 시에 사용하는 것 금지
    - 동일한 출력을 만들어 내는 다른 모델/알고리즘 사용 금지


# 3. SAFETY GUARDRAIL
- NEVER EDIT
    - inputs.bin, answers.bin, decode_answers.bin, model.bin, main.cpp, config.h, Makefile, tensor.h, tensor.cu, layer.h, layer.cu
- can edit
    - model.h, model.cu, model_loader.h, model_loader.cu, run.sh

# 4. Rule
- Don't guess, but measure.
- Don't assume. Don't hide confusion. Surface tradeoffs
- Simplicity first
    - Minimum code that solves the problem. Nothing speculative.
    - Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.
- Touch only what you must. Clean up only your own mess.
    - Don't "improve" adjacent code, comments, or formatting.
    - Don't refactor things that aren't broken.
- Record experiments at `docs/lab-note.md`, refer to `docs/note-format.md` for template.
    - 반드시 1개의 가설씩 실험을 진행해야 한다. 동시에 여러개를 바꿔버리면 뭐가 실제 레버인지 알 수 없다.
    - 여러 가설 중 가장 파급력이 클 만한 가설을 먼저 실험한다. 먼저 했을 때 이득이 크면서도 나중에 다른 실험에 미칠 잠재적인 위험성이 적으며, 롤백할 일을 최소화할 수 있도록 다음 실험을 결정해라. 
    - speedup을 최대한 올려보겠다고 minor한 변경을 끼워넣을 필요 없이, 핵심 가설 중심으로 실험을 구성하라.
    - 기록은 최대한 간결해야 하며, bloat을 쳐내고 load-bearing한 알짜 내용만 기록
    - 실험 1개마다 반드시 commit을 수행한다. 커밋 명은 <index>-<slug>
- 실험 output binary는 `outputs/` 폴더에 저장
- 사용자가 실험 하나에 대해 **microworld**를 생성하라고 하면 `docs/microworld.md`를 참고하여 제작 후, `docs/microworld/`에 저장하라.
- 모든 CUDA API 호출에는 디버깅 용이성을 위해 에러 코드가 포함되어야 한다.

## 6. useful scripts
- `./run.sh -h`: 실행으로 실행 스크립트 옵션 확인 가능
- `./run.sh -n 1 -v`: 총 squence n은 총 1024개 지만, 빠른 테스트를 위해 n을 1개만 하는 경우, `-v`는 evaluation mode
- `python3 tools/visualize_output.py --output <output_path> --index <input index> [-d]`: output 시각화 툴