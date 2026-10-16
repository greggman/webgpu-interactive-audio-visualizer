import {PreMappedReadBufferPool, PreMappedWriteBufferPool} from './buffer-pool';
import {
  AUDIO_CHUNK_BYTES,
  AUDIO_CHUNK_FRAMES,
  INITIAL_PREMAPPED_BUFFERS,
  NUM_KEYS,
  TARGET_BUFFERED_FRAMES,
} from './constants';
import {PianoKeyboardController} from './keyboard';
import {createAudioShaderCode} from './shaders/audio.wgsl';
import {createVisualizerShaderCode} from './shaders/visualizer.wgsl';

const VERSION = 1;

// AudioUniforms layout in WGSL:
// 8 x 4-byte header fields (32 bytes) + 32 x KeyState (32 * 16 = 512 bytes) = 544 bytes
const UNIFORM_FLOAT_COUNT = 8 + 32 * 4;
const UNIFORM_BUFFER_BYTES =
  UNIFORM_FLOAT_COUNT * Float32Array.BYTES_PER_ELEMENT;

interface WorkletStatusMessage {
  type: 'status';
  epoch: number;
  bufferedFrames: number;
  framesReceivedInEpoch: number;
}

class WebGpuAudioVisualizerApp {
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly canvas: HTMLCanvasElement;
  private readonly canvasFormat: 'bgra8unorm' | 'rgba8unorm';

  private readonly audioUniformBuffer: GPUBuffer;
  private readonly audioOutputStorageBuffer: GPUBuffer;
  private readonly writeBufferPool: PreMappedWriteBufferPool;
  private readonly readBufferPool: PreMappedReadBufferPool;

  private readonly audioComputePipeline: GPUComputePipeline;
  private readonly audioBindGroup: GPUBindGroup;

  private readonly visualizerBindGroupLayout: GPUBindGroupLayout;
  private readonly visualizerComputePipeline: GPUComputePipeline;

  private readonly uniformScratchBuffer = new ArrayBuffer(UNIFORM_BUFFER_BYTES);
  private readonly uniformFloatView = new Float32Array(
    this.uniformScratchBuffer,
  );
  private readonly uniformUintView = new Uint32Array(this.uniformScratchBuffer);

  private readonly keyboard: PianoKeyboardController;

  private audioCtx: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private audioInitPromise: Promise<void> | null = null;

  private sampleRate = 48000;
  private nextSampleFrame = 0;
  private workletBufferedFrames = 0;
  private inFlightChunksForEpoch = 0;
  private framesPostedInEpoch = 0;
  private generationEpoch = 0;
  private pendingPreempt = false;

  private bpm = 120;
  private beatEnabled = true;
  private masterVolume = 0.85;
  private synthMode = 0;

  private uncapturedErrors: string[] = [];

  constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    canvas: HTMLCanvasElement,
    canvasFormat: 'bgra8unorm' | 'rgba8unorm',
    keyboardContainer: HTMLElement,
  ) {
    this.device = device;
    this.context = context;
    this.canvas = canvas;
    this.canvasFormat = canvasFormat;

    // Monitor uncaptured error events from WebGPU so testing sees any WebGPU errors
    this.device.addEventListener(
      'uncapturederror',
      (event: GPUUncapturedErrorEvent) => {
        const msg = event.error.message;
        this.uncapturedErrors.push(msg);
        console.error('WebGPU uncaptured error:', msg);
        this.updateErrorBanner(msg);
      },
    );

    this.device.lost
      .then((info: GPUDeviceLostInfo) => {
        console.error(`WebGPU device lost (${info.reason}): ${info.message}`);
      })
      .catch((err: unknown) => {
        console.error('Error handling device lost:', err);
      });

    // Create GPU buffers with descriptive labels
    this.audioUniformBuffer = this.device.createBuffer({
      label: 'AudioUniformsBuffer',
      size: UNIFORM_BUFFER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.audioOutputStorageBuffer = this.device.createBuffer({
      label: 'AudioOutputStorageBuffer',
      size: AUDIO_CHUNK_BYTES,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });

    // Pre-mapped staging buffer pools
    this.writeBufferPool = new PreMappedWriteBufferPool(
      this.device,
      UNIFORM_BUFFER_BYTES,
      'AudioUniformsPool',
      INITIAL_PREMAPPED_BUFFERS,
    );

    this.readBufferPool = new PreMappedReadBufferPool(
      this.device,
      AUDIO_CHUNK_BYTES,
      'AudioOutputPool',
      INITIAL_PREMAPPED_BUFFERS,
    );

    // Create Audio Synthesis Compute Pipeline
    const audioShaderModule = this.device.createShaderModule({
      label: 'AudioComputeShaderModule',
      code: createAudioShaderCode(),
    });

    const audioBindGroupLayout = this.device.createBindGroupLayout({
      label: 'AudioComputeBindGroupLayout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {type: 'uniform'},
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {type: 'storage'},
        },
      ],
    });

    const audioPipelineLayout = this.device.createPipelineLayout({
      label: 'AudioComputePipelineLayout',
      bindGroupLayouts: [audioBindGroupLayout],
    });

    this.audioComputePipeline = this.device.createComputePipeline({
      label: 'AudioComputePipeline',
      layout: audioPipelineLayout,
      compute: {
        module: audioShaderModule,
        entryPoint: 'main',
      },
    });

    this.audioBindGroup = this.device.createBindGroup({
      label: 'AudioComputeBindGroup',
      layout: audioBindGroupLayout,
      entries: [
        {binding: 0, resource: {buffer: this.audioUniformBuffer}},
        {binding: 1, resource: {buffer: this.audioOutputStorageBuffer}},
      ],
    });

    // Create Audio Visualization Compute Pipeline
    const visualizerShaderModule = this.device.createShaderModule({
      label: 'VisualizerComputeShaderModule',
      code: createVisualizerShaderCode(this.canvasFormat),
    });

    this.visualizerBindGroupLayout = this.device.createBindGroupLayout({
      label: 'VisualizerComputeBindGroupLayout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            access: 'write-only',
            format: this.canvasFormat,
            viewDimension: '2d',
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {type: 'uniform'},
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {type: 'read-only-storage'},
        },
      ],
    });

    const visualizerPipelineLayout = this.device.createPipelineLayout({
      label: 'VisualizerComputePipelineLayout',
      bindGroupLayouts: [this.visualizerBindGroupLayout],
    });

    this.visualizerComputePipeline = this.device.createComputePipeline({
      label: 'VisualizerComputePipeline',
      layout: visualizerPipelineLayout,
      compute: {
        module: visualizerShaderModule,
        entryPoint: 'main',
      },
    });

    // Initialize interactive piano keyboard with lookahead preemption on input
    this.keyboard = new PianoKeyboardController(
      keyboardContainer,
      () => this.rewindLookaheadForInput(),
      () => this.pumpAudioGeneration(),
      () => {
        void this.ensureAudioStarted();
      },
    );

    this.setupUiControls();
    this.setupResizeObserver();
  }

  async start(): Promise<void> {
    // Upload initial uniform state so the backdrop renders cleanly behind the start overlay
    const initialUniforms = this.packUniformData(0);
    const writeStaging = this.writeBufferPool.acquireAndWrite(initialUniforms);
    const encoder = this.device.createCommandEncoder({
      label: 'InitialUniformCommandEncoder',
    });
    encoder.copyBufferToBuffer(
      writeStaging,
      0,
      this.audioUniformBuffer,
      0,
      UNIFORM_BUFFER_BYTES,
    );
    this.device.queue.submit([
      encoder.finish({label: 'InitialUniformCommandBuffer'}),
    ]);
    this.writeBufferPool.remapAfterSubmit(writeStaging);

    const renderFrame = () => {
      this.onAnimationFrame();
      requestAnimationFrame(renderFrame);
    };
    requestAnimationFrame(renderFrame);
  }

  /**
   * Rewinds `nextSampleFrame` to discard unplayed lookahead chunks so newly
   * pressed/released notes are synthesized and heard within ~5-8ms.
   */
  private rewindLookaheadForInput(): number {
    if (
      this.audioCtx &&
      this.audioCtx.state === 'running' &&
      this.workletNode
    ) {
      const keepFrames = AUDIO_CHUNK_FRAMES;
      const droppableBuffered = Math.max(
        0,
        this.workletBufferedFrames - keepFrames,
      );
      const droppableChunks =
        Math.floor(droppableBuffered / AUDIO_CHUNK_FRAMES) +
        this.inFlightChunksForEpoch;

      if (droppableChunks > 0) {
        this.nextSampleFrame = Math.max(
          0,
          this.nextSampleFrame - droppableChunks * AUDIO_CHUNK_FRAMES,
        );
        this.workletBufferedFrames = Math.min(
          this.workletBufferedFrames,
          keepFrames,
        );
        this.generationEpoch++;
        this.inFlightChunksForEpoch = 0;
        this.framesPostedInEpoch = 0;
        this.pendingPreempt = true;
      }
    }
    return this.nextSampleFrame / this.sampleRate;
  }

  private async ensureAudioStarted(): Promise<void> {
    if (!this.audioInitPromise) {
      this.audioInitPromise = this.initWebAudio();
    }
    // Call resume() synchronously before awaiting audioInitPromise so the
    // browser's transient user activation is still active on the call stack.
    if (this.audioCtx && this.audioCtx.state !== 'running') {
      void this.audioCtx.resume().catch(() => {
        // Browser may require a direct user gesture before resuming
      });
    }
    await this.audioInitPromise;
    this.updateStartOverlayUi();
  }

  private async initWebAudio(): Promise<void> {
    try {
      const ctx = new AudioContext({latencyHint: 'interactive'});
      this.audioCtx = ctx;
      this.sampleRate = ctx.sampleRate;

      ctx.addEventListener('statechange', () => {
        this.updateStartOverlayUi();
        if (ctx.state === 'running') {
          this.pumpAudioGeneration();
        }
      });

      if (ctx.state !== 'running') {
        void ctx.resume().catch(() => {
          // Ignore if called outside a user gesture
        });
      }

      await ctx.audioWorklet.addModule('audio-worker.js');
      const workletNode = new AudioWorkletNode(ctx, 'webgpu-audio-processor', {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [2],
      });

      workletNode.port.onmessage = (
        event: MessageEvent<WorkletStatusMessage>,
      ) => {
        const data = event.data;
        if (!data || data.type !== 'status') {
          return;
        }
        // Ignore stale status messages from before the latest input preemption epoch
        if (data.epoch !== this.generationEpoch) {
          return;
        }
        // Account for chunks already posted in this epoch that the worklet had
        // not yet received when it sent this status message.
        const framesInTransit = Math.max(
          0,
          this.framesPostedInEpoch - data.framesReceivedInEpoch,
        );
        this.workletBufferedFrames = data.bufferedFrames + framesInTransit;
        this.pumpAudioGeneration();
      };

      workletNode.connect(ctx.destination);
      this.workletNode = workletNode;
      this.updateStartOverlayUi();
      this.pumpAudioGeneration();
    } catch (err: unknown) {
      console.error('Failed to initialize Web Audio Worklet:', err);
    }
  }

  private packUniformData(startTimeSeconds: number): ArrayBufferView {
    const f32 = this.uniformFloatView;
    const u32 = this.uniformUintView;

    f32[0] = this.sampleRate;
    f32[1] = startTimeSeconds;
    f32[2] = this.bpm;
    f32[3] = this.beatEnabled ? 1.0 : 0.0;
    f32[4] = this.masterVolume;
    u32[5] = this.synthMode;
    u32[6] = AUDIO_CHUNK_FRAMES;
    u32[7] = this.keyboard.getActiveKeyCount();

    const noteStates = this.keyboard.getNoteStates();
    for (let i = 0; i < NUM_KEYS; i++) {
      const base = 8 + i * 4;
      const note = noteStates[i];
      f32[base + 0] = note.frequency;
      f32[base + 1] = note.isPressed ? 1.0 : 0.0;
      f32[base + 2] = note.pressTime;
      f32[base + 3] = note.releaseTime;
    }

    return this.uniformFloatView;
  }

  /**
   * Generates chunks of audio in the WebGPU compute shader and sends them
   * to the Web Audio Worker whenever the buffered frame count is below target.
   */
  private pumpAudioGeneration(): void {
    if (!this.workletNode || this.audioCtx?.state !== 'running') {
      return;
    }

    while (
      this.workletBufferedFrames +
        this.inFlightChunksForEpoch * AUDIO_CHUNK_FRAMES <
        TARGET_BUFFERED_FRAMES &&
      this.inFlightChunksForEpoch < INITIAL_PREMAPPED_BUFFERS
    ) {
      const preempt = this.pendingPreempt;
      this.pendingPreempt = false;
      this.dispatchAudioComputeChunk(preempt);
    }
  }

  private dispatchAudioComputeChunk(preempt: boolean): void {
    const startTime = this.nextSampleFrame / this.sampleRate;
    this.nextSampleFrame += AUDIO_CHUNK_FRAMES;

    const uniformData = this.packUniformData(startTime);

    // 1. Immediately acquire a pre-mapped write staging buffer and copy uniform state
    const writeStagingBuffer =
      this.writeBufferPool.acquireAndWrite(uniformData);

    // 2. Immediately acquire a pre-mapped readback buffer and unmap for GPU copy destination
    const readStagingBuffer = this.readBufferPool.acquireForCopyDst();

    const encoder = this.device.createCommandEncoder({
      label: 'AudioGenerationCommandEncoder',
    });

    encoder.copyBufferToBuffer(
      writeStagingBuffer,
      0,
      this.audioUniformBuffer,
      0,
      UNIFORM_BUFFER_BYTES,
    );

    const computePass = encoder.beginComputePass({
      label: 'AudioGenerationComputePass',
    });
    computePass.setPipeline(this.audioComputePipeline);
    computePass.setBindGroup(0, this.audioBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(AUDIO_CHUNK_FRAMES / 64));
    computePass.end();

    encoder.copyBufferToBuffer(
      this.audioOutputStorageBuffer,
      0,
      readStagingBuffer,
      0,
      AUDIO_CHUNK_BYTES,
    );

    const commandBuffer = encoder.finish({
      label: 'AudioGenerationCommandBuffer',
    });
    this.device.queue.submit([commandBuffer]);

    // 3. After submit, use promises to remap both buffers and return them to the pre-mapped pools
    this.writeBufferPool.remapAfterSubmit(writeStagingBuffer);

    const dispatchEpoch = this.generationEpoch;
    this.inFlightChunksForEpoch++;
    this.readBufferPool.remapAndReadAfterSubmit(readStagingBuffer, samples => {
      if (dispatchEpoch !== this.generationEpoch) {
        return;
      }
      this.inFlightChunksForEpoch = Math.max(
        0,
        this.inFlightChunksForEpoch - 1,
      );
      if (this.workletNode) {
        this.framesPostedInEpoch += AUDIO_CHUNK_FRAMES;
        this.workletBufferedFrames += AUDIO_CHUNK_FRAMES;
        this.workletNode.port.postMessage(
          {
            type: 'audio-chunk',
            epoch: dispatchEpoch,
            samples,
            preempt,
          },
          [samples.buffer],
        );
      }
    });
  }

  private onAnimationFrame(): void {
    if (
      this.audioCtx &&
      this.audioCtx.state === 'running' &&
      this.workletNode
    ) {
      this.pumpAudioGeneration();
    }

    // Dispatch the separate visualization compute shader to write to the canvas storage texture
    const canvasTexture = this.context.getCurrentTexture();
    const canvasView = canvasTexture.createView({
      label: 'CanvasStorageTextureView',
    });

    const visualizerBindGroup = this.device.createBindGroup({
      label: 'VisualizerComputeBindGroup',
      layout: this.visualizerBindGroupLayout,
      entries: [
        {binding: 0, resource: canvasView},
        {binding: 1, resource: {buffer: this.audioUniformBuffer}},
        {binding: 2, resource: {buffer: this.audioOutputStorageBuffer}},
      ],
    });

    const encoder = this.device.createCommandEncoder({
      label: 'VisualizerCommandEncoder',
    });
    const pass = encoder.beginComputePass({
      label: 'VisualizerComputePass',
    });
    pass.setPipeline(this.visualizerComputePipeline);
    pass.setBindGroup(0, visualizerBindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(canvasTexture.width / 8),
      Math.ceil(canvasTexture.height / 8),
    );
    pass.end();

    const commandBuffer = encoder.finish({
      label: 'VisualizerCommandBuffer',
    });
    this.device.queue.submit([commandBuffer]);
  }

  private setupUiControls(): void {
    const startOverlay = document.getElementById('start-overlay');
    if (startOverlay) {
      startOverlay.addEventListener('click', () => {
        void this.ensureAudioStarted();
      });
    }

    const beatBtn = document.getElementById('beat-toggle-btn');
    if (beatBtn) {
      beatBtn.addEventListener('click', () => {
        void this.ensureAudioStarted();
        this.beatEnabled = !this.beatEnabled;
        beatBtn.textContent = this.beatEnabled ? 'Beat: ON' : 'Beat: OFF';
        beatBtn.classList.toggle('active', this.beatEnabled);
        this.rewindLookaheadForInput();
        this.pumpAudioGeneration();
      });
    }

    const bpmInput = document.getElementById(
      'bpm-slider',
    ) as HTMLInputElement | null;
    const bpmValue = document.getElementById('bpm-value');
    if (bpmInput) {
      bpmInput.addEventListener('input', () => {
        void this.ensureAudioStarted();
        const parsed = Number.parseFloat(bpmInput.value);
        if (!Number.isNaN(parsed)) {
          this.bpm = Math.min(200, Math.max(60, parsed));
          if (bpmValue) {
            bpmValue.textContent = `${Math.round(this.bpm)} BPM`;
          }
        }
      });
    }

    const synthSelect = document.getElementById(
      'synth-mode-select',
    ) as HTMLSelectElement | null;
    if (synthSelect) {
      synthSelect.addEventListener('change', () => {
        void this.ensureAudioStarted();
        const mode = Number.parseInt(synthSelect.value, 10);
        if (!Number.isNaN(mode)) {
          this.synthMode = mode;
          this.rewindLookaheadForInput();
          this.pumpAudioGeneration();
        }
      });
    }
  }

  private updateStartOverlayUi(): void {
    const startOverlay = document.getElementById('start-overlay');
    if (!startOverlay) {
      return;
    }
    const isRunning = this.audioCtx?.state === 'running';
    startOverlay.hidden = Boolean(isRunning);
  }

  private updateErrorBanner(message: string): void {
    const banner = document.getElementById('error-banner');
    if (!banner) {
      return;
    }
    banner.textContent = `WebGPU Error: ${message}`;
    banner.hidden = false;
  }

  private setupResizeObserver(): void {
    const maxDim = this.device.limits.maxTextureDimension2D;
    const observer = new ResizeObserver(entries => {
      for (const entry of entries) {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const width = Math.round(entry.contentRect.width * dpr);
        const height = Math.round(entry.contentRect.height * dpr);
        const clampedWidth = Math.max(1, Math.min(maxDim, width));
        const clampedHeight = Math.max(1, Math.min(maxDim, height));
        if (
          this.canvas.width !== clampedWidth ||
          this.canvas.height !== clampedHeight
        ) {
          this.canvas.width = clampedWidth;
          this.canvas.height = clampedHeight;
        }
      }
    });
    observer.observe(this.canvas);
  }
}

function showFatalMessage(message: string): void {
  const startOverlay = document.getElementById('start-overlay');
  if (startOverlay) {
    startOverlay.hidden = true;
  }
  const banner = document.getElementById('error-banner');
  if (banner) {
    banner.textContent = message;
    banner.hidden = false;
  }
  console.error(message);
}

async function init(): Promise<void> {
  const versionElem = document.getElementById('version');
  if (versionElem) {
    versionElem.textContent = `${VERSION}`;
  }

  if (!navigator.gpu) {
    showFatalMessage('WebGPU is not supported in this browser.');
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    showFatalMessage('Failed to request WebGPU adapter.');
    return;
  }

  const preferredFormat = navigator.gpu.getPreferredCanvasFormat();
  const canUseBgra8UnormStorage = adapter.features.has('bgra8unorm-storage');

  const device = await adapter.requestDevice({
    label: 'AudioVisualizerGPUDevice',
    requiredFeatures: canUseBgra8UnormStorage ? ['bgra8unorm-storage'] : [],
  });

  const canvas = document.getElementById(
    'visualizer-canvas',
  ) as HTMLCanvasElement | null;
  const keyboardContainer = document.getElementById('piano-keyboard');
  if (!canvas || !keyboardContainer) {
    showFatalMessage('Missing required DOM elements.');
    return;
  }

  const context = canvas.getContext('webgpu');
  if (!context) {
    showFatalMessage('Failed to obtain WebGPU canvas context.');
    return;
  }

  const canvasFormat: 'bgra8unorm' | 'rgba8unorm' =
    preferredFormat === 'bgra8unorm' && canUseBgra8UnormStorage
      ? 'bgra8unorm'
      : 'rgba8unorm';

  context.configure({
    device,
    format: canvasFormat,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    alphaMode: 'opaque',
  });

  const app = new WebGpuAudioVisualizerApp(
    device,
    context,
    canvas,
    canvasFormat,
    keyboardContainer,
  );
  await app.start();
}

void init();
