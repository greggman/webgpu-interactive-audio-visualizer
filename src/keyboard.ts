import {KeyDefinition, NUM_KEYS, PIANO_KEYS} from './constants';

export interface NoteRuntimeState {
  frequency: number;
  isPressed: boolean;
  pressTime: number;
  releaseTime: number;
}

export class PianoKeyboardController {
  private readonly container: HTMLElement;
  private readonly prepareInputTime: () => number;
  private readonly onStateChanged: () => void;
  private readonly onUserInteract: () => void;
  private readonly keyElements: HTMLElement[] = [];
  private readonly noteStates: NoteRuntimeState[] = [];
  private readonly pointerToKey = new Map<number, number>();
  private readonly codeToKey = new Map<string, number>();
  private readonly heldCodes = new Set<string>();

  constructor(
    container: HTMLElement,
    prepareInputTime: () => number,
    onStateChanged: () => void,
    onUserInteract: () => void,
  ) {
    this.container = container;
    this.prepareInputTime = prepareInputTime;
    this.onStateChanged = onStateChanged;
    this.onUserInteract = onUserInteract;

    for (const def of PIANO_KEYS) {
      this.noteStates.push({
        frequency: def.frequency,
        isPressed: false,
        pressTime: -100,
        releaseTime: -100,
      });
      for (const code of def.codes) {
        this.codeToKey.set(code, def.index);
      }
    }

    this.buildDom();
    this.attachEvents();
  }

  private buildDom(): void {
    this.container.replaceChildren();

    const whiteKeys = PIANO_KEYS.filter(k => !k.isBlack);
    const totalWhite = whiteKeys.length;

    let whiteCountSoFar = 0;
    const whiteIndexBeforeKey: number[] = [];
    for (const def of PIANO_KEYS) {
      whiteIndexBeforeKey.push(whiteCountSoFar);
      if (!def.isBlack) {
        whiteCountSoFar++;
      }
    }

    for (const def of PIANO_KEYS) {
      const keyEl = this.createKeyElement(
        def,
        whiteIndexBeforeKey[def.index],
        totalWhite,
      );
      this.keyElements.push(keyEl);
      this.container.appendChild(keyEl);
    }
  }

  private createKeyElement(
    def: KeyDefinition,
    precedingWhiteCount: number,
    totalWhite: number,
  ): HTMLElement {
    const btn = document.createElement('div');
    btn.className = def.isBlack ? 'piano-key black-key' : 'piano-key white-key';
    btn.dataset.keyIndex = String(def.index);
    btn.setAttribute('role', 'button');
    btn.setAttribute('aria-label', `Piano key ${def.note}`);

    if (def.isBlack) {
      const whiteWidthPercent = 100 / totalWhite;
      const blackWidthPercent = whiteWidthPercent * 0.64;
      const leftPercent =
        precedingWhiteCount * whiteWidthPercent - blackWidthPercent * 0.5;
      btn.style.left = `${leftPercent.toFixed(4)}%`;
      btn.style.width = `${blackWidthPercent.toFixed(4)}%`;
    } else {
      const whiteWidthPercent = 100 / totalWhite;
      const leftPercent = precedingWhiteCount * whiteWidthPercent;
      btn.style.left = `${leftPercent.toFixed(4)}%`;
      btn.style.width = `${whiteWidthPercent.toFixed(4)}%`;
    }

    const shortcutSpan = document.createElement('span');
    shortcutSpan.className = 'key-shortcut';
    shortcutSpan.textContent = def.keyLabel;

    const noteSpan = document.createElement('span');
    noteSpan.className = 'key-note';
    noteSpan.textContent = def.note;

    btn.appendChild(shortcutSpan);
    btn.appendChild(noteSpan);
    return btn;
  }

  private attachEvents(): void {
    this.container.addEventListener('pointerdown', (e: PointerEvent) => {
      this.onUserInteract();
      try {
        this.container.setPointerCapture(e.pointerId);
      } catch {
        // Ignore if pointer capture is unsupported on synthetic events
      }
      const keyIndex = this.findKeyIndexAtPoint(e.clientX, e.clientY, e.target);
      if (keyIndex !== null) {
        this.pointerToKey.set(e.pointerId, keyIndex);
        if (this.recomputeKeyPressState(keyIndex)) {
          this.onStateChanged();
        }
      }
    });

    this.container.addEventListener('pointermove', (e: PointerEvent) => {
      if (!this.pointerToKey.has(e.pointerId)) {
        return;
      }
      const prevKey = this.pointerToKey.get(e.pointerId) ?? null;
      const nextKey = this.findKeyIndexAtPoint(e.clientX, e.clientY, null);
      if (nextKey !== prevKey) {
        if (nextKey !== null) {
          this.pointerToKey.set(e.pointerId, nextKey);
        } else {
          this.pointerToKey.delete(e.pointerId);
        }
        let changed = false;
        if (prevKey !== null) {
          changed = this.recomputeKeyPressState(prevKey) || changed;
        }
        if (nextKey !== null) {
          changed = this.recomputeKeyPressState(nextKey) || changed;
        }
        if (changed) {
          this.onStateChanged();
        }
      }
    });

    const handlePointerEnd = (e: PointerEvent) => {
      this.onUserInteract();
      const prevKey = this.pointerToKey.get(e.pointerId);
      if (prevKey !== undefined) {
        this.pointerToKey.delete(e.pointerId);
        if (this.recomputeKeyPressState(prevKey)) {
          this.onStateChanged();
        }
      }
    };

    this.container.addEventListener('pointerup', handlePointerEnd);
    this.container.addEventListener('pointercancel', handlePointerEnd);
    this.container.addEventListener('lostpointercapture', handlePointerEnd);

    window.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) {
        return;
      }
      const keyIndex = this.codeToKey.get(e.code);
      if (keyIndex === undefined) {
        return;
      }
      this.onUserInteract();
      this.heldCodes.add(e.code);
      if (this.recomputeKeyPressState(keyIndex)) {
        this.onStateChanged();
      }
    });

    window.addEventListener('keyup', (e: KeyboardEvent) => {
      const keyIndex = this.codeToKey.get(e.code);
      if (keyIndex === undefined) {
        return;
      }
      this.heldCodes.delete(e.code);
      if (this.recomputeKeyPressState(keyIndex)) {
        this.onStateChanged();
      }
    });

    window.addEventListener('blur', () => {
      this.pointerToKey.clear();
      this.heldCodes.clear();
      let changed = false;
      for (let i = 0; i < NUM_KEYS; i++) {
        changed = this.recomputeKeyPressState(i) || changed;
      }
      if (changed) {
        this.onStateChanged();
      }
    });
  }

  private findKeyIndexAtPoint(
    clientX: number,
    clientY: number,
    fallbackTarget: EventTarget | null,
  ): number | null {
    const hit = document.elementFromPoint(clientX, clientY);
    const target =
      hit ?? (fallbackTarget instanceof Element ? fallbackTarget : null);
    if (!target) {
      return null;
    }
    const keyEl = target.closest('[data-key-index]') as HTMLElement | null;
    if (!keyEl || !keyEl.dataset.keyIndex) {
      return null;
    }
    const idx = Number.parseInt(keyEl.dataset.keyIndex, 10);
    return Number.isNaN(idx) ? null : idx;
  }

  private isKeyHeldByInput(keyIndex: number): boolean {
    for (const heldIdx of this.pointerToKey.values()) {
      if (heldIdx === keyIndex) {
        return true;
      }
    }
    for (const code of this.heldCodes) {
      if (this.codeToKey.get(code) === keyIndex) {
        return true;
      }
    }
    return false;
  }

  private recomputeKeyPressState(keyIndex: number): boolean {
    const state = this.noteStates[keyIndex];
    const el = this.keyElements[keyIndex];
    if (!state || !el) {
      return false;
    }

    const shouldBePressed = this.isKeyHeldByInput(keyIndex);

    if (shouldBePressed && !state.isPressed) {
      const now = this.prepareInputTime();
      state.isPressed = true;
      state.pressTime = now;
      el.classList.add('active');
      return true;
    } else if (!shouldBePressed && state.isPressed) {
      const now = this.prepareInputTime();
      state.isPressed = false;
      state.releaseTime = Math.max(now, state.pressTime + 0.005);
      el.classList.remove('active');
      return true;
    }
    return false;
  }

  getNoteStates(): readonly NoteRuntimeState[] {
    return this.noteStates;
  }

  getActiveKeyCount(): number {
    let count = 0;
    for (const s of this.noteStates) {
      if (s.isPressed) {
        count++;
      }
    }
    return count;
  }
}
