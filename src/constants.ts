export const AUDIO_CHUNK_FRAMES = 256;
export const AUDIO_CHANNELS = 2;
export const AUDIO_CHUNK_SAMPLES = AUDIO_CHUNK_FRAMES * AUDIO_CHANNELS;
export const AUDIO_CHUNK_BYTES =
  AUDIO_CHUNK_SAMPLES * Float32Array.BYTES_PER_ELEMENT;

// Keep 6 chunks (~32ms at 48kHz) queued in steady state, and preempt unplayed chunks on input
export const TARGET_BUFFERED_FRAMES = AUDIO_CHUNK_FRAMES * 6;
export const INITIAL_PREMAPPED_BUFFERS = 8;

export const NUM_KEYS = 25; // C3 (MIDI 48) to C5 (MIDI 72)

export interface KeyDefinition {
  index: number;
  midi: number;
  note: string;
  frequency: number;
  isBlack: boolean;
  keyLabel: string;
  codes: string[];
}

const NOTE_NAMES = [
  'C',
  'C#',
  'D',
  'D#',
  'E',
  'F',
  'F#',
  'G',
  'G#',
  'A',
  'A#',
  'B',
];

const KEY_BINDINGS: Array<{label: string; codes: string[]}> = [
  // Octave 3 (C3 - B3)
  {label: 'Z', codes: ['KeyZ']}, // C3
  {label: 'S', codes: ['KeyS']}, // C#3
  {label: 'X', codes: ['KeyX']}, // D3
  {label: 'D', codes: ['KeyD']}, // D#3
  {label: 'C', codes: ['KeyC']}, // E3
  {label: 'V', codes: ['KeyV']}, // F3
  {label: 'G', codes: ['KeyG']}, // F#3
  {label: 'B', codes: ['KeyB']}, // G3
  {label: 'H', codes: ['KeyH']}, // G#3
  {label: 'N', codes: ['KeyN']}, // A3
  {label: 'J', codes: ['KeyJ']}, // A#3
  {label: 'M', codes: ['KeyM']}, // B3
  // Octave 4 (C4 - B4) + C5
  {label: 'Q', codes: ['KeyQ', 'Comma', 'KeyA']}, // C4
  {label: '2', codes: ['Digit2', 'KeyL']}, // C#4
  {label: 'W', codes: ['KeyW', 'Period']}, // D4
  {label: '3', codes: ['Digit3', 'Semicolon']}, // D#4
  {label: 'E', codes: ['KeyE', 'Slash']}, // E4
  {label: 'R', codes: ['KeyR', 'KeyF']}, // F4
  {label: '5', codes: ['Digit5']}, // F#4
  {label: 'T', codes: ['KeyT']}, // G4
  {label: '6', codes: ['Digit6']}, // G#4
  {label: 'Y', codes: ['KeyY']}, // A4
  {label: '7', codes: ['Digit7']}, // A#4
  {label: 'U', codes: ['KeyU']}, // B4
  {label: 'I', codes: ['KeyI', 'KeyK']}, // C5
];

export const PIANO_KEYS: KeyDefinition[] = Array.from(
  {length: NUM_KEYS},
  (_, i) => {
    const midi = 48 + i;
    const noteIndex = midi % 12;
    const octave = Math.floor(midi / 12) - 1;
    const noteName = NOTE_NAMES[noteIndex];
    const isBlack = noteName.includes('#');
    const frequency = 440 * Math.pow(2, (midi - 69) / 12);
    const binding = KEY_BINDINGS[i];
    return {
      index: i,
      midi,
      note: `${noteName}${octave}`,
      frequency,
      isBlack,
      keyLabel: binding.label,
      codes: binding.codes,
    };
  },
);
