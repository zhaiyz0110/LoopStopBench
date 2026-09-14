# Third-Party Notices

This file records the principal third-party resources used to construct or
evaluate LoopStopBench. It is an attribution and provenance record, not legal
advice and not a replacement for the original license texts.

No model weights, complete upstream benchmark distributions, or arXiv source
corpus are redistributed in this repository. The frozen trajectory files may
contain task identifiers, generated responses, benchmark-derived evaluation
fields, and short source-derived text. Those portions remain subject to the
applicable upstream terms described below.

## Benchmarks and source material

### EvalPlus, HumanEval+, and MBPP+

- Project: [EvalPlus](https://github.com/evalplus/evalplus)
- Frozen test releases:
  [HumanEval+](https://github.com/evalplus/humanevalplus_release) and
  [MBPP+](https://github.com/evalplus/mbppplus_release)
- EvalPlus code and the release repositories are distributed under the
  Apache License 2.0.
- HumanEval+ is derived from OpenAI HumanEval and additionally follows the
  [HumanEval MIT license](https://github.com/openai/human-eval/blob/master/LICENSE).
- LoopStopBench does not relicense upstream prompts or tests. Use of those
  materials must comply with the relevant upstream licenses.

### HotpotQA

- Project: [HotpotQA](https://hotpotqa.github.io/)
- Repository: [hotpotqa/hotpot](https://github.com/hotpotqa/hotpot)
- The upstream repository states that the HotpotQA data is licensed under
  CC BY-SA 4.0 and its code is licensed under Apache-2.0.
- The LoopStopBench QA trajectories contain benchmark-derived question and
  evaluation information. Those portions remain subject to the HotpotQA data
  license, including attribution and share-alike obligations where applicable.

### arXiv metadata and abstracts

- API terms: [arXiv API Terms of Use](https://info.arxiv.org/help/api/tou.html)
- The writing-task collection used arXiv records as source material. The
  original task cache and source corpus are not included in this release.
- arXiv descriptive metadata is made available under CC0, but abstracts and
  e-prints retain the copyright or license selected for each submission. This
  repository does not claim ownership of source-paper text that may be
  reflected in a prompt or generated response.

## Models and services

The following models or services were used during trajectory collection or
signal generation. They are named for reproducibility; their weights are not
redistributed here.

| Resource | Governing terms |
|---|---|
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Apache-2.0 model repository license |
| [Qwen2.5-32B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ) | Apache-2.0 model repository license |
| [Qwen2.5-72B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ) | Qwen model license shown in the model repository |
| [Meta Llama 3.3 70B Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | Llama 3.3 Community License and Acceptable Use Policy |
| Anthropic API models | [Anthropic Commercial Terms](https://www.anthropic.com/legal/commercial-terms) and applicable service policies |

Generated text is not automatically covered by this repository's software
license. Users should review the applicable model license, service terms,
benchmark license, and local law before redistributing or reusing trajectories.

## Software dependencies

Python packages and container images are installed as independent dependencies
and remain under their own licenses. The authoritative dependency declarations
are `pyproject.toml` and the files under `docker/`. In particular, vLLM,
PyTorch, scikit-learn, NumPy, pandas, SciPy, PyYAML, and their transitive
dependencies are not relicensed by LoopStopBench.

All third-party names and trademarks belong to their respective owners. Their
mention does not imply endorsement.
