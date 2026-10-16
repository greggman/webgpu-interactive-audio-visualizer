# WebGPU Interactive Audio Synthesizer & Visualizer

A dependency-free interactive audio synthesizer and visualizer built with WebGPU compute shaders, Web Audio API (`AudioWorklet`), and TypeScript.

---

## Low-Latency Dataflow Architecture

Keeping an interactive instrument responsive while generating audio on the GPU requires eliminating asynchronous stalls between user input, GPU command submission, CPU readback, and the Web Audio rendering thread.

```
┌────────────────────────────────────────────────────────────────────────────┐
│ 1. Input Layer (Main Thread)                                               │
│    Pointer Events (multi-touch / drag) + Keyboard Events                   │
│    • Calls rewindLookaheadForInput() to target the next unplayed chunk     │
│      (~4.0–9.3 ms ahead of the AudioWorklet's current playhead)            │
│    • Updates NoteRuntimeState[] (isPressed, pressTime, releaseTime, freq)  │
│    • Immediately dispatches pumpAudioGeneration()                          │
└───────────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 2. Pre-Mapped Write Buffer Pool (MAP_WRITE | COPY_SRC)                     │
│    • Pop already-mapped GPUBuffer from availableBuffers (0 ms wait)        │
│    • Write 544-byte AudioUniforms into getMappedRange() & unmap()          │
│    • encoder.copyBufferToBuffer(writeStaging -> audioUniformBuffer)        │
│    • After queue.submit(): writeStaging.mapAsync(WRITE).then(returnToPool) │
└───────────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 3. Audio Synthesis Compute Shader (audio.wgsl.ts)                          │
│    • Reads audioUniformBuffer (time, BPM, beat toggle, 25 key states)      │
│    • Synthesizes 256 stereo frames (~5.33 ms @ 48 kHz) in 4 workgroups     │
│    • Writes stereo samples to audioOutputStorageBuffer (array<vec2f>)      │
└──────────────────┬────────────────────────────────────────┬────────────────┘
                   │                                        │
                   │ (GPU-to-CPU Audio Path)                │ (GPU-Only Visual Path)
                   ▼                                        ▼
┌──────────────────────────────────────────┐  ┌──────────────────────────────┐
│ 4. Pre-Mapped Read Buffer Pool           │  │ 6. Visualizer Compute Shader │
│    (MAP_READ | COPY_DST)                 │  │    (visualizer.wgsl.ts)      │
│    • Pop pre-mapped buffer & unmap()     │  │    • Reads audioUniformBuffer│
│    • copyBufferToBuffer(                 │  │      & audioOutputStorage    │
│        audioOutputStorage -> readStaging)│  │      directly on the GPU     │
│    • After queue.submit():               │  │    • Writes directly to      │
│      readStaging.mapAsync(READ).then:    │  │      Canvas Storage Texture  │
│      1. Copy Float32Array samples        │  │      (bgra8unorm / rgba8unorm│
│      2. Keep buffer mapped & push back   │  │       with STORAGE_BINDING)  │
│         to availableBuffers for next use │  └──────────────────────────────┘
│      3. Transfer ArrayBuffer + startFrame│
│         to AudioWorklet                  │
└──────────────────┬───────────────────────┘
                   │ postMessage({startFrame, samples}, [samples.buffer])
                   ▼
┌──────────────────────────────────────────┐
│ 5. Web Audio Worker (audio-worker.ts)    │
│    (AudioWorkletProcessor Thread)        │
│    • Keeps chunkQueue[0] playing cleanly │
│    • Preempts any unstarted lookahead    │
│      chunks >= incoming startFrame       │
│    • Drains 128 frames per render quantum│
│    • Reports currentPlayheadFrame back   │
│      to main thread every 2.67 ms        │
└──────────────────────────────────────────┘
```

---

## Key Mechanisms Keeping Everything Responsive

### 1. Small `256`-Frame Chunks + Lookahead Queue Preemption on Input
A naive FIFO queue has a fundamental trade-off: a larger queue (`~50 ms`) prevents audio underruns when the main thread is busy, but forces any newly pressed note to wait behind `50 ms` of already-generated audio.

We get **both** a `32 ms` underrun safety cushion in steady state **and** `~6.7 ms` synthesis latency on input by combining small timeline-indexed chunks with queue preemption:
- **Steady State (`32 ms` buffer)**:
  - Each GPU dispatch generates **`256` stereo frames** (`5.33 ms` at `48 kHz`), tagged with its exact `startFrame` on the continuous audio timeline.
  - During normal playback, `pumpAudioGeneration()` keeps `6` chunks (`1536` frames = `32 ms`) queued ahead of the `AudioWorklet`'s `currentPlayheadFrame`.
- **On Key Press / Release (`~4.0–9.3 ms` preemption)**:
  - Because `audio.wgsl.ts` is a pure deterministic function of sample time `t = startFrame / sampleRate`, any unplayed lookahead chunk can be replaced at its exact `startFrame` boundary while remaining 100% phase-continuous for the beat and already-held notes.
  - When a key is pressed or released, `rewindLookaheadForInput()` (`src/main.ts`) rewinds `nextSampleFrame` to the earliest chunk boundary at least `192` frames (`4.0 ms`) ahead of `workletPlayheadFrame` and immediately dispatches `pumpAudioGeneration()` inside the input event handler.
  - When those newly synthesized chunks arrive at `WebGpuAudioProcessor` (`src/audio-worker.ts`), the processor keeps `chunkQueue[0]` (the chunk currently playing) untouched so playback never clicks, drops all unstarted chunks in `chunkQueue[1..N]` with `startFrame >= incoming.startFrame`, and splices in the new chunks.

### 2. Pre-Mapped Staging Buffer Pools (`src/buffer-pool.ts`)
Calling `buffer.mapAsync()` on demand and waiting for its Promise before encoding GPU commands would stall the main thread and add a full CPU–GPU round-trip latency before work is even submitted. Instead, two pools of pre-mapped buffers are maintained:
- **`PreMappedWriteBufferPool` (`MAP_WRITE | COPY_SRC`)**:
  - Buffers are mapped ahead of time via `mapAsync(GPUMapMode.WRITE)` and stored in `availableBuffers`.
  - When a chunk needs to be generated, `acquireAndWrite()` immediately pops an already-mapped buffer, writes the latest keyboard and beat state into `getMappedRange()`, and calls `unmap()` synchronously so `copyBufferToBuffer` can be encoded right away.
  - Immediately after `device.queue.submit()`, `remapAfterSubmit()` calls `buffer.mapAsync(GPUMapMode.WRITE).then(...)` to remap the buffer in the background and return it to `availableBuffers`.
- **`PreMappedReadBufferPool` (`MAP_READ | COPY_DST`)**:
  - Buffers are pre-mapped via `mapAsync(GPUMapMode.READ)` and stored in `availableBuffers`.
  - When a chunk is dispatched, `acquireForCopyDst()` immediately pops a buffer from `availableBuffers` and calls `unmap()` so the GPU can copy `audioOutputStorageBuffer` into it.
  - Right after `device.queue.submit()`, `remapAndReadAfterSubmit()` calls `buffer.mapAsync(GPUMapMode.READ).then(...)`. When the Promise resolves, the audio samples are copied out of `getMappedRange()`, the buffer is left in its **mapped** state and pushed back into `availableBuffers` for the next cycle, and the sample `ArrayBuffer` is transferred zero-copy (`postMessage(..., [samples.buffer])`) to the `AudioWorkletNode`.

### 3. Zero-Readback GPU Visualization via Canvas Storage Textures
- The visualization compute shader (`src/shaders/visualizer.wgsl.ts`) reads `audioUniformBuffer` and `audioOutputStorageBuffer` **directly on the GPU** without waiting for CPU readback.
- The HTML `<canvas>` is configured with `GPUTextureUsage.STORAGE_BINDING`. On adapters supporting the `'bgra8unorm-storage'` feature where `navigator.gpu.getPreferredCanvasFormat()` is `'bgra8unorm'`, the device enables `'bgra8unorm-storage'` so the compute shader can write pixels directly via `textureStore()` into the native `'bgra8unorm'` swapchain texture without an extra copy or render pass.

### 4. Sample-Accurate Note Envelopes on the GPU
- Pointer and keyboard events (`src/keyboard.ts`) record exact audio-clock timestamps (`pressTime` and `releaseTime`) for each of the 25 piano keys.
- In `src/shaders/audio.wgsl.ts`, every thread computes its exact sample timestamp `t = params.startTime + f32(frameIndex) / params.sampleRate` and evaluates continuous attack, decay, sustain, and exponential release envelopes per sample, avoiding zipper noise or clicks across chunk boundaries.

---

## Development & Build Commands

- **Install dependencies**:
  ```bash
  npm install
  ```
- **Typecheck and bundle to `dist/`**:
  ```bash
  npm run build
  ```
- **Lint with `gts`**:
  ```bash
  npm run lint
  ```
- **Auto-format with `gts`**:
  ```bash
  npm run fix
  ```
