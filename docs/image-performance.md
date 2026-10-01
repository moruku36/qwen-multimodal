# Image generation cost/performance review (2026-10-01)

## Evidence and changes

This is a source review and CPU/mock regression verification, **not an A100 benchmark**.
No Colab units, paid API calls or GPU runs were used. No speedup or CU-saving percentage is claimed.

- The old UI hardcoded **2048 band / 50 steps**, ignoring `ImageModelConfig.default_band=1024`, its `default_steps`, and the GPU profile's 40-step A100 default. The new UI uses the same defaults as the engine. The notebook explicitly sets **1024 / 40 / one image**. This is a visible resolution/workload tradeoff, not quality-neutral acceleration. At square aspect 2048 has four times the output pixels of 1024; that is **not** a fourfold runtime prediction.
- The visible preset selector offers configured defaults, **draft 768 / 20**, and **legacy quality 2048 / 50**, within the configured/GPU resolution cap. Applying a preset resets the batch to one. You may then edit resolution/steps/batch; reapply restores the selected preset. Draft can lose detail, text accuracy, and instruction fidelity. Use legacy quality for final work where needed. Neither the model nor CFG/negative prompt is changed.
- `QMC_IMAGE_BAND`, `QMC_IMAGE_STEPS`, and `QMC_IMAGE_MAX_BAND` now reach both UI and engine; previously `default_steps` and `max_band` were unused. The band is an area-style size category, not a strict maximum side: 16:9 at 2048 is 2752×1536.
- Variants remain serial, with one/two/four selectable for generation; edits still produce one. Four variants are four complete calls, not a cheap batch.
- The normal image path no longer forces Python garbage collection, GPU synchronization, allocator cache release and IPC collection before every image. Unload/OOM cleanup remains. This preserves reusable allocations; its runtime benefit is unmeasured.
- Image prefetch now passes the same `cache_dir` as image loading. With `HF_HOME` set, the old prefetch used the Hub default `$HF_HOME/hub`, while the loader explicitly used `$HF_HOME`. That could download two copies. The change prevents future duplication; it does not delete old cache directories. [HF cache environment semantics](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)
- Progress shows elapsed time for the current image, including load/wait. Completion separates preparation (search, prompt rewrite, reference loading) from image processing. Generation history now stores **per-image**, rather than cumulative batch durations. These are application wall times, not GPU kernel timings or billing measurements.
- Cancel before image activation avoids an unnecessary model load. Cancellation during diffusion aborts before decoding a discarded image; exception cleanup restores offload hooks before reuse. Already-running model downloads/loads are not instantly interruptible.

## Remaining costs and suspected bottlenecks

The pinned pipeline already enables KV cache by default, and the manager reuses an active model. Repeated image generation does not normally redownload/rebuild weights. The full default negative prompt with CFG=2 adds a second transformer forward per diffusion step; removing it would change behavior, so it remains intact. See the [pinned pipeline source](https://github.com/huggingface/diffusers/blob/e0abab83b5df05de9e7abd788643c1a7c1e42e28/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py).

Search is on by default and prompt rewriting uses the chat model. They can add significant preparation time. For a ready-to-use prompt, users can explicitly select prompt “そのまま” and search “オフ”; defaults were not silently changed.

A100 80GB co-residency with the current Q8 chat model has not been measured. Memory pressure may trigger offload/OOM recovery and a full image retry. CPU offload moves large model components between RAM and GPU. These are plausible costs, not established explanations of this user's specific slow run. OOM policy and quantization are unchanged; a broader change needs real GPU memory measurements. Older A100/Q4 measurements do not establish Q8 performance. Lower steps mainly reduce compute, not model-weight VRAM.

Cold downloads are large (~27GB chat plus ~40GB image as currently documented). Cached downloads still need loading into RAM/VRAM, and Drive I/O can be slow. No cache migration, eager download or new provider was introduced.

## How to verify on a future already-authorized GPU session

1. Use the supported A100 runtime and record GPU type/VRAM, model versions, precision, placement and model-manager load/OOM events.
2. Compare identical prompts/seeds and reference images at configured, draft and legacy-quality settings. Record quality failures, preparation time, first (cold) image time and second (warm) image time separately. Keep CFG and prompt rewriting consistent when comparing.
3. Test text-to-image and edit independently, then cancel mid-generation and generate again. Check memory recovery and stable repeated-run memory use.
4. Compare one versus two variants; each history duration should cover its own image. Do not extrapolate a CU rate without actual Colab accounting.
5. Explicitly end the Colab session after saving. **Stop generation** and **unload model** do not end a GPU runtime or stop its allocation. The previous safe shutdown control remains unchanged.

## Verification boundary

CPU tests cover configuration/presets, cache routing, timing bookkeeping, cancellation and fake-pipeline hook cleanup, plus existing app flows. No downloaded real model or CUDA pipeline was executed. Browser visual verification was attempted against a local mock server; the cloud browser rejected localhost with `ERR_BLOCKED_BY_CLIENT`, so this is not claimed as visually verified.

Final local checks: 270 tests passed, 1 GPU integration module skipped; changed-file Ruff and diff whitespace checks passed; notebook JSON/Python cells parsed (Colab magics excluded). Full-tree Ruff still reports three baseline findings in `router.py`, `search_engine.py`, and `test_search.py`; they are outside this patch. An independent source review found no blockers. Model-load OOM recovery still retains the failed-load exception traceback through its nested retry; this separate recovery-path concern was not changed in the targeted patch.
