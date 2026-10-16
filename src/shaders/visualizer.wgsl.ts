import {AUDIO_CHUNK_FRAMES, NUM_KEYS} from '../constants';

export function createVisualizerShaderCode(
  textureFormat: 'bgra8unorm' | 'rgba8unorm',
): string {
  return /* wgsl */ `
struct KeyState {
  frequency: f32,
  isPressed: f32,
  pressTime: f32,
  releaseTime: f32,
};

struct AudioUniforms {
  sampleRate: f32,
  startTime: f32,
  bpm: f32,
  beatEnabled: f32,
  masterVolume: f32,
  synthMode: u32,
  chunkFrames: u32,
  activeKeyCount: u32,
  keys: array<KeyState, 32>,
};

@group(0) @binding(0) var outTex: texture_storage_2d<${textureFormat}, write>;
@group(0) @binding(1) var<uniform> params: AudioUniforms;
@group(0) @binding(2) var<storage, read> audioData: array<vec2f>;

const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;
const CHUNK_FRAMES: u32 = ${AUDIO_CHUNK_FRAMES}u;
const NUM_PIANO_KEYS: u32 = ${NUM_KEYS}u;

fn hsv2rgb(c: vec3f) -> vec3f {
  let k = vec4f(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  let p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
  return c.z * mix(k.xxx, clamp(p - k.xxx, vec3f(0.0), vec3f(1.0)), c.y);
}

fn sampleAudioSmooth(normX: f32) -> vec2f {
  let pos = clamp(normX, 0.0, 0.9999) * f32(CHUNK_FRAMES - 1u);
  let idx0 = u32(floor(pos));
  let idx1 = min(idx0 + 1u, CHUNK_FRAMES - 1u);
  let frac = fract(pos);
  return mix(audioData[idx0], audioData[idx1], frac);
}

fn keyActivity(key: KeyState, t: f32) -> f32 {
  if (key.isPressed > 0.5) {
    return 1.0;
  }
  if (key.pressTime < 0.0) {
    return 0.0;
  }
  let relAge = max(0.0, t - key.releaseTime);
  if (relAge > 0.65) {
    return 0.0;
  }
  return exp(-relAge * 8.0);
}

var<workgroup> sharedEnergy: f32;

@compute @workgroup_size(8, 8)
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_index) localIdx: u32,
) {
  if (localIdx == 0u) {
    var sum = 0.0;
    for (var i = 0u; i < CHUNK_FRAMES; i = i + 16u) {
      let s = audioData[i];
      sum += dot(s, s);
    }
    sharedEnergy = sqrt(sum / f32(CHUNK_FRAMES / 16u));
  }
  workgroupBarrier();

  let dims = textureDimensions(outTex);
  if (gid.x >= dims.x || gid.y >= dims.y) {
    return;
  }

  let energy = sharedEnergy;
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
  let aspect = f32(dims.x) / max(f32(dims.y), 1.0);
  let centered = vec2f((uv.x - 0.5) * aspect, 0.5 - uv.y);
  let t = params.startTime;

  // Deep space dark gradient background with subtle beat-synced glow
  let beatsPerSec = max(params.bpm, 40.0) / 60.0;
  let beatPhase = fract(t * beatsPerSec);
  let beatPulse = select(0.0, exp(-beatPhase * 6.0), params.beatEnabled > 0.5);

  var color = mix(
    vec3f(0.03, 0.04, 0.09),
    vec3f(0.07, 0.03, 0.14),
    uv.y
  );

  // Subtle synth grid
  let gridX = abs(fract(uv.x * f32(NUM_PIANO_KEYS)) - 0.5);
  let gridY = abs(fract(uv.y * 12.0) - 0.5);
  let gridLine = smoothstep(0.48, 0.5, max(gridX, gridY));
  color += vec3f(0.03, 0.05, 0.10) * gridLine;

  // Radial bass / beat ring in the center
  let radius = length(centered);
  let angle = atan2(centered.y, centered.x);
  let radialIndex = clamp((angle + PI) / TAU, 0.0, 0.9999);
  let radialSample = sampleAudioSmooth(radialIndex);
  let targetRadius = 0.20 + 0.07 * beatPulse + 0.14 * energy + 0.08 * radialSample.x;
  let ringDist = abs(radius - targetRadius);
  let ringGlow = 0.008 / (ringDist + 0.006);
  let ringColor = hsv2rgb(vec3f(fract(radialIndex + t * 0.08), 0.75, 1.0));
  color += ringColor * ringGlow * (0.28 + 0.72 * clamp(energy * 2.5, 0.0, 1.0));

  // Piano key columns aura across the x-axis
  let keyFloat = uv.x * f32(NUM_PIANO_KEYS);
  let keyIdx = min(u32(floor(keyFloat)), NUM_PIANO_KEYS - 1u);
  let keyLocalX = fract(keyFloat);
  let activity = keyActivity(params.keys[keyIdx], t);
  if (activity > 0.001) {
    let keyHue = f32(keyIdx) / f32(NUM_PIANO_KEYS);
    let keyCol = hsv2rgb(vec3f(keyHue, 0.8, 1.0));
    let edgeMask = smoothstep(0.0, 0.12, keyLocalX) * smoothstep(1.0, 0.88, keyLocalX);
    let heightMask = smoothstep(0.0, 0.85, uv.y);
    let waveMod = 0.5 + 0.5 * sin(uv.y * 20.0 - t * params.keys[keyIdx].frequency * 0.05);
    color += keyCol * activity * edgeMask * heightMask * (0.22 + 0.25 * waveMod);
  }

  // Stereo waveform oscilloscope ribbons
  let sampleHere = sampleAudioSmooth(uv.x);
  let sampleNext = sampleAudioSmooth(min(uv.x + 1.5 / f32(dims.x), 1.0));

  // Left channel waveform (cyan / electric blue)
  let waveYL = 0.45 - sampleHere.x * 0.30;
  let waveYLNext = 0.45 - sampleNext.x * 0.30;
  let slopeL = (waveYLNext - waveYL) * f32(dims.x);
  let distL = abs(uv.y - waveYL) / sqrt(1.0 + slopeL * slopeL * 0.15);
  let coreL = smoothstep(0.0045, 0.001, distL);
  let glowL = 0.004 / (distL + 0.004);
  color += vec3f(0.15, 0.92, 1.0) * coreL + vec3f(0.05, 0.55, 0.95) * glowL * 0.55;

  // Right channel waveform (magenta / neon coral)
  let waveYR = 0.55 - sampleHere.y * 0.30;
  let waveYRNext = 0.55 - sampleNext.y * 0.30;
  let slopeR = (waveYRNext - waveYR) * f32(dims.x);
  let distR = abs(uv.y - waveYR) / sqrt(1.0 + slopeR * slopeR * 0.15);
  let coreR = smoothstep(0.0045, 0.001, distR);
  let glowR = 0.004 / (distR + 0.004);
  color += vec3f(1.0, 0.25, 0.78) * coreR + vec3f(0.90, 0.12, 0.55) * glowR * 0.55;

  // 16-step beat indicator bar along the top
  if (params.beatEnabled > 0.5 && uv.y < 0.045) {
    let stepPos = t * beatsPerSec * 4.0;
    let activeStep = u32(floor(stepPos)) % 16u;
    let stepF = uv.x * 16.0;
    let cellStep = u32(floor(stepF));
    let cellX = fract(stepF);
    let cellMask = smoothstep(0.06, 0.12, cellX) * smoothstep(0.94, 0.88, cellX) *
                   smoothstep(0.008, 0.014, uv.y) * smoothstep(0.042, 0.036, uv.y);
    let isCurrent = cellStep == activeStep;
    let isDownbeat = (cellStep % 4u) == 0u;
    var stepCol = vec3f(0.12, 0.16, 0.25);
    if (isCurrent) {
      stepCol = select(vec3f(0.2, 0.95, 0.75), vec3f(1.0, 0.75, 0.2), isDownbeat);
    } else if (isDownbeat) {
      stepCol = vec3f(0.22, 0.28, 0.42);
    }
    color = mix(color, stepCol, cellMask * 0.9);
  }

  // Tone-map and gamma-friendly clamp
  let finalRgb = clamp(color / (1.0 + color * 0.35), vec3f(0.0), vec3f(1.0));
  textureStore(outTex, vec2i(gid.xy), vec4f(finalRgb, 1.0));
}
`;
}
