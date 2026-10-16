import {AUDIO_CHUNK_FRAMES, NUM_KEYS} from '../constants';

export function createAudioShaderCode(): string {
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

@group(0) @binding(0) var<uniform> params: AudioUniforms;
@group(0) @binding(1) var<storage, read_write> audioOut: array<vec2f>;

const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;
const NUM_PIANO_KEYS: u32 = ${NUM_KEYS}u;

fn hash11(p: f32) -> f32 {
  var p2 = fract(p * 0.1031);
  p2 *= p2 + 33.33;
  p2 *= p2 + p2;
  return fract(p2) * 2.0 - 1.0;
}

fn synthesizeBeat(t: f32, sampleRate: f32, bpm: f32) -> vec2f {
  let beatsPerSec = max(bpm, 40.0) / 60.0;
  let stepRate = beatsPerSec * 4.0; // 16th notes per second
  let stepPos = t * stepRate;
  let stepIndex = u32(floor(stepPos)) % 16u;
  let stepFrac = fract(stepPos);
  let stepTime = stepFrac / stepRate;
  let attack = clamp(stepTime / 0.002, 0.0, 1.0);

  // 1. Kick drum on steps 0, 4, 8, 10, 12
  var kick = 0.0;
  if (stepIndex == 0u || stepIndex == 4u || stepIndex == 8u || stepIndex == 10u || stepIndex == 12u) {
    let kickPhase = TAU * (48.0 * stepTime + (115.0 / 30.0) * (1.0 - exp(-stepTime * 30.0)));
    let kickEnv = exp(-stepTime * 11.5) * attack;
    let clickEnv = exp(-stepTime * 90.0) * 0.25;
    kick = tanh((sin(kickPhase) + clickEnv * sin(TAU * 420.0 * stepTime)) * 1.7) * kickEnv * 0.55;
  }

  // 2. Snare drum on steps 4, 12 and ghost snare on step 15
  var snare = 0.0;
  if (stepIndex == 4u || stepIndex == 12u || stepIndex == 15u) {
    let gain = select(0.38, 0.16, stepIndex == 15u);
    let bodyPhase = TAU * (175.0 * stepTime + (90.0 / 35.0) * (1.0 - exp(-stepTime * 35.0)));
    let body = sin(bodyPhase) * exp(-stepTime * 22.0);
    let noise = hash11(floor(t * sampleRate)) * exp(-stepTime * 16.0);
    snare = (body * 0.45 + noise * 0.75) * attack * gain;
  }

  // 3. Hi-hat on all 16th steps with open accent on steps 2, 6, 10, 14
  let isOpenHat = (stepIndex % 4u) == 2u;
  let hatDecay = select(48.0, 14.0, isOpenHat);
  let hatGain = select(0.09, 0.16, isOpenHat);
  let highNoise = hash11(floor(t * sampleRate) + 19.19) - hash11(floor(t * sampleRate) + 18.19);
  let hat = highNoise * exp(-stepTime * hatDecay) * attack * hatGain;

  // 4. Funky Synth Bassline
  var bassFreq = 65.41; // C2
  if (stepIndex == 2u || stepIndex == 3u) {
    bassFreq = 65.41; // C2
  } else if (stepIndex == 5u || stepIndex == 6u) {
    bassFreq = 77.78; // D#2 / Eb2
  } else if (stepIndex == 7u) {
    bassFreq = 87.31; // F2
  } else if (stepIndex == 10u || stepIndex == 11u) {
    bassFreq = 98.00; // G2
  } else if (stepIndex == 13u) {
    bassFreq = 87.31; // F2
  } else if (stepIndex == 14u || stepIndex == 15u) {
    bassFreq = 116.54; // A#2 / Bb2
  }

  let bassActive = (stepIndex != 4u && stepIndex != 12u);
  var bass = 0.0;
  if (bassActive) {
    let filtEnv = exp(-stepTime * 10.0);
    let h1 = sin(TAU * bassFreq * t);
    let h2 = 0.55 * sin(TAU * bassFreq * 2.0 * t) * filtEnv;
    let h3 = 0.35 * sin(TAU * bassFreq * 3.0 * t) * filtEnv * filtEnv;
    let h4 = 0.20 * sin(TAU * bassFreq * 4.0 * t) * filtEnv * filtEnv;
    let ampEnv = exp(-stepTime * 6.5) * attack;
    bass = tanh((h1 + h2 + h3 + h4) * 1.4) * ampEnv * 0.28;
  }

  let center = kick + snare + bass;
  return vec2f(center + hat * 0.85, center + hat * 1.15);
}

fn synthesizeKey(key: KeyState, keyIndex: u32, t: f32, synthMode: u32) -> vec2f {
  if (key.pressTime < 0.0) {
    return vec2f(0.0);
  }

  let isHeld = key.isPressed > 0.5;
  if (!isHeld && t < key.pressTime) {
    return vec2f(0.0);
  }

  let noteAge = max(0.0, t - key.pressTime);

  var releaseEnv = 1.0;
  if (!isHeld) {
    let relAge = max(0.0, t - key.releaseTime);
    if (relAge > 0.65) {
      return vec2f(0.0);
    }
    releaseEnv = exp(-relAge * 13.0);
  }

  let attackEnv = clamp(noteAge / 0.006, 0.0, 1.0);
  let sustainDecay = 0.35 + 0.65 * exp(-noteAge * 1.6);
  let env = attackEnv * sustainDecay * releaseEnv;
  if (env < 0.0005) {
    return vec2f(0.0);
  }

  let freq = key.frequency;
  var signal = 0.0;

  if (synthMode == 0u) {
    // Mode 0: Acoustic/Synth Grand Piano with decaying upper harmonics
    let p1 = sin(TAU * freq * t);
    let p2 = 0.45 * sin(TAU * freq * 2.001 * t) * exp(-noteAge * 2.2);
    let p3 = 0.22 * sin(TAU * freq * 3.003 * t) * exp(-noteAge * 3.8);
    let p4 = 0.12 * sin(TAU * freq * 4.006 * t) * exp(-noteAge * 5.5);
    let p5 = 0.06 * sin(TAU * freq * 5.010 * t) * exp(-noteAge * 8.0);
    let hammer = 0.15 * sin(TAU * freq * 7.0 * t) * exp(-noteAge * 32.0);
    signal = (p1 + p2 + p3 + p4 + p5 + hammer) * 0.32 * env;
  } else if (synthMode == 1u) {
    // Mode 1: Warm FM Electric Piano (Rhodes-style)
    let modIndex = 2.2 * exp(-noteAge * 4.5) + 0.35;
    let modulator = sin(TAU * freq * 2.0 * t) * modIndex;
    let tine = 0.25 * sin(TAU * freq * 7.0 * t) * exp(-noteAge * 18.0);
    signal = (sin(TAU * freq * t + modulator) + tine) * 0.30 * env;
  } else {
    // Mode 2: Cyber Synth Lead (detuned saw harmonics + sub)
    let fA = freq * 0.996;
    let fB = freq * 1.004;
    let sawA = sin(TAU * fA * t) + 0.5 * sin(TAU * fA * 2.0 * t) + 0.33 * sin(TAU * fA * 3.0 * t);
    let sawB = sin(TAU * fB * t) + 0.5 * sin(TAU * fB * 2.0 * t) + 0.33 * sin(TAU * fB * 3.0 * t);
    let sub = 0.4 * sin(TAU * (freq * 0.5) * t);
    signal = (sawA * 0.45 + sawB * 0.45 + sub) * 0.24 * env;
  }

  // Pan lower keys slightly left and higher keys slightly right
  let pan = (f32(keyIndex) / f32(NUM_PIANO_KEYS - 1u)) * 0.6 - 0.3;
  let leftGain = cos((pan + 1.0) * 0.25 * PI);
  let rightGain = sin((pan + 1.0) * 0.25 * PI);
  return vec2f(signal * leftGain, signal * rightGain);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let frameIndex = gid.x;
  if (frameIndex >= ${AUDIO_CHUNK_FRAMES}u) {
    return;
  }

  let t = params.startTime + f32(frameIndex) / params.sampleRate;
  var stereo = vec2f(0.0);

  if (params.beatEnabled > 0.5) {
    stereo += synthesizeBeat(t, params.sampleRate, params.bpm);
  }

  for (var i = 0u; i < NUM_PIANO_KEYS; i = i + 1u) {
    stereo += synthesizeKey(params.keys[i], i, t, params.synthMode);
  }

  // Master volume and smooth soft-clipper
  stereo = tanh(stereo * params.masterVolume);
  audioOut[frameIndex] = stereo;
}
`;
}
