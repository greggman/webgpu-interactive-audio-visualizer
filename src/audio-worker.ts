declare abstract class AudioWorkletProcessor {
  readonly port: MessagePort;
  constructor();
  abstract process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>,
  ): boolean;
}

declare function registerProcessor(
  name: string,
  processorCtor: new () => AudioWorkletProcessor,
): void;

interface AudioChunkMessage {
  type: 'audio-chunk';
  epoch: number;
  samples: Float32Array;
  preempt?: boolean;
}

interface ResetMessage {
  type: 'reset';
}

type WorkerInboundMessage = AudioChunkMessage | ResetMessage;

interface QueuedChunk {
  frames: number;
  samples: Float32Array;
}

class WebGpuAudioProcessor extends AudioWorkletProcessor {
  private readonly chunkQueue: QueuedChunk[] = [];
  private currentFrameOffset = 0;
  private bufferedFrames = 0;
  private currentEpoch = 0;
  private framesReceivedInEpoch = 0;
  private quantaSinceStatus = 0;

  constructor() {
    super();
    this.port.onmessage = (event: MessageEvent<WorkerInboundMessage>) => {
      const data = event.data;
      if (!data) {
        return;
      }
      if (data.type === 'audio-chunk' && data.samples instanceof Float32Array) {
        // Ignore stale chunks from an older input epoch
        if (data.epoch < this.currentEpoch) {
          return;
        }

        const frames = Math.floor(data.samples.length / 2);
        if (frames <= 0) {
          return;
        }

        // If a new input epoch started (or preempt is flagged), keep only the
        // currently playing chunk (plus one more if <128 frames remain in it)
        // so playback never clicks, and drop all unplayed lookahead chunks.
        if (data.epoch > this.currentEpoch || data.preempt) {
          this.currentEpoch = data.epoch;
          this.framesReceivedInEpoch = 0;

          if (this.chunkQueue.length > 1) {
            const remainingInFirst =
              this.chunkQueue[0].frames - this.currentFrameOffset;
            const keepCount = remainingInFirst < 128 ? 2 : 1;
            this.chunkQueue.length = Math.min(
              this.chunkQueue.length,
              keepCount,
            );
          }

          // Recompute exact bufferedFrames remaining in the kept chunk(s)
          let remaining = 0;
          for (let i = 0; i < this.chunkQueue.length; i++) {
            remaining += this.chunkQueue[i].frames;
          }
          if (this.chunkQueue.length > 0) {
            remaining -= this.currentFrameOffset;
          }
          this.bufferedFrames = Math.max(0, remaining);
        }

        this.chunkQueue.push({
          frames,
          samples: data.samples,
        });
        this.bufferedFrames += frames;
        this.framesReceivedInEpoch += frames;
      } else if (data.type === 'reset') {
        this.chunkQueue.length = 0;
        this.currentFrameOffset = 0;
        this.bufferedFrames = 0;
        this.framesReceivedInEpoch = 0;
      }
    };
  }

  override process(
    _inputs: Float32Array[][],
    outputs: Float32Array[][],
  ): boolean {
    const output = outputs[0];
    if (!output || output.length === 0) {
      return true;
    }

    const left = output[0];
    const right = output.length > 1 ? output[1] : undefined;
    const frameCount = left.length;

    let outIndex = 0;
    while (outIndex < frameCount && this.chunkQueue.length > 0) {
      const currentChunk = this.chunkQueue[0];
      const availableInChunk = currentChunk.frames - this.currentFrameOffset;
      const framesToCopy = Math.min(frameCount - outIndex, availableInChunk);

      for (let i = 0; i < framesToCopy; i++) {
        const sampleBase = (this.currentFrameOffset + i) * 2;
        const l = currentChunk.samples[sampleBase];
        const r = currentChunk.samples[sampleBase + 1];
        left[outIndex + i] = l;
        if (right) {
          right[outIndex + i] = r;
        }
      }

      outIndex += framesToCopy;
      this.currentFrameOffset += framesToCopy;
      this.bufferedFrames = Math.max(0, this.bufferedFrames - framesToCopy);

      if (this.currentFrameOffset >= currentChunk.frames) {
        this.chunkQueue.shift();
        this.currentFrameOffset = 0;
      }
    }

    // Fill any remaining frames with silence if buffer underran
    for (; outIndex < frameCount; outIndex++) {
      left[outIndex] = 0;
      if (right) {
        right[outIndex] = 0;
      }
    }

    this.quantaSinceStatus++;
    if (this.quantaSinceStatus >= 2 || this.bufferedFrames < 768) {
      this.quantaSinceStatus = 0;
      this.port.postMessage({
        type: 'status',
        epoch: this.currentEpoch,
        bufferedFrames: this.bufferedFrames,
        framesReceivedInEpoch: this.framesReceivedInEpoch,
      });
    }

    return true;
  }
}

registerProcessor('webgpu-audio-processor', WebGpuAudioProcessor);
