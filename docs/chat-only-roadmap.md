# Q8-first chat notebook roadmap

The new `Qwen-Q8-Chat-Colab.ipynb` targets an A100 80GB and uses the existing
`huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF` model file
`Huihui-Qwen3.8-27B-abliterated-UD-DW-Q8_K_L.gguf` with the compatible official
`ggml-org/Qwen3.8-27B-GGUF` projector `mmproj-Qwen3.8-27B-Q8_0.gguf`.
It starts one local llama.cpp chat/vision server. It keeps web search, chat
history and CPU speech recognition. It has no image generation UI, routes,
model registration, prefetch or diffusers dependencies.

## Stages

1. **Now:** retain Q8 and remove image generation/editing from the new notebook.
   Keep the original `Qwen-Multimodal-Colab.ipynb` as is.
2. **October 23 or later, after compute replenishment:** validate Q8 on A100 80GB
   for chat, vision, search, ASR, speed, answer quality, peak/idle VRAM, and Colab
   compute unit use. Test cold start and repeated turns. Record settings and
   measurements before making performance claims.
3. **Later:** evaluate a BF16 precision version of the same Qwen3.8-27B model
   against Q8 under matched prompts and context. BF16 is a weight precision,
   not inherently a newer or better model. Decide from measured quality,
   latency, VRAM, download size, and compute unit use. No runtime switching is
   planned.

Image generation was lazy loaded and could coexist with chat on A100; removing
it does not establish a fixed VRAM saving without measurement. The manual
Release GPU model control unloads weights but does not stop Colab charges;
the notebook's separate confirmed shutdown control ends the session.
