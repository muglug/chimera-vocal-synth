/**
 * Plays one decoded track and keeps the one playhead every head reads: what the listener hears
 * now, in seconds into the track. Its own AudioContext (the voice engine doesn't run here).
 *
 * The track is the show's mix, or its parts: one mono track per voice (`loadParts`), started
 * together and each panned to its place, which sound as the mix does and can be muted one by one
 * (to sing a part yourself against the others).
 *
 * Paused, the playhead is the saved position. Playing, it is that position plus the audible time
 * since the source started (never less than none: the first moments after Play are still in the
 * output buffer): the context's output timestamp where the browser has one (it already allows for
 * output latency), else currentTime less the reported latency. A seek bumps `epoch` (so animation
 * history can be reset) and restarts a fresh source. The track ends when its end is heard, not when
 * the source finishes rendering. The context leaving "running" (an iOS interruption, the page
 * going to the background), or failing to resume after Play, pauses the transport where it was.
 */
export type TransportState = { playing: boolean; time: number; duration: number; epoch: number };

export class Transport {
  readonly ctx: AudioContext;
  /** what plays: the mix alone, or every part */
  private buffers: AudioBuffer[] = [];
  /** each part's way out (its gain, then its pan); none for a mix, which goes straight out */
  private outs: { gain: GainNode; pan: StereoPannerNode; muted: boolean }[] = [];
  private srcs: AudioBufferSourceNode[] = [];
  private offset = 0;
  private startedAt = 0;
  private playing = false;
  /** the source has rendered its last sample; the track ends once that is heard */
  private rendered = false;
  /** bumped by every play, pause, seek and dispose: a late resume result for an old one is ignored */
  private generation = 0;
  private wasRunning = false;
  epoch = 0;
  private readonly listeners = new Set<() => void>();

  constructor(ctx: AudioContext = new AudioContext({ latencyHint: "playback" })) {
    this.ctx = ctx;
    this.ctx.onstatechange = () => {
      const running = this.ctx.state === "running";
      if (this.wasRunning && !running && this.playing) this.pause();
      this.wasRunning = running;
    };
  }

  async load(data: ArrayBuffer): Promise<void> {
    this.buffers = [await this.ctx.decodeAudioData(data)];
    this.emit();
  }

  /** The show as its parts: one mono track per voice, each panned −1 (left) … 1 (right). */
  async loadParts(data: ArrayBuffer[], pans: number[]): Promise<void> {
    const buffers = await Promise.all(data.map((d) => this.ctx.decodeAudioData(d)));
    this.outs = buffers.map((_, i) => {
      const gain = this.ctx.createGain(), pan = this.ctx.createStereoPanner();
      pan.pan.value = Math.min(1, Math.max(-1, pans[i] ?? 0));
      gain.connect(pan);
      pan.connect(this.ctx.destination);
      return { gain, pan, muted: false };
    });
    this.buffers = buffers;
    this.emit();
  }

  /** whether parts can be muted (the show was loaded as its parts) */
  get canMute(): boolean {
    return this.outs.length > 0;
  }

  isMuted(part: number): boolean {
    return this.outs[part]?.muted ?? false;
  }

  /** Silence a part, or bring it back: at once, playing or not (over 10 ms, without a click). */
  setMuted(part: number, muted: boolean): void {
    const o = this.outs[part];
    if (!o || o.muted === muted) return;
    o.muted = muted;
    o.gain.gain.setTargetAtTime(muted ? 0 : 1, this.ctx.currentTime, 0.01);
    this.emit();
  }

  get duration(): number {
    return this.buffers.reduce((d, b) => Math.max(d, b.duration), 0);
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  get state(): TransportState {
    return { playing: this.playing, time: this.time(), duration: this.duration, epoch: this.epoch };
  }

  subscribe(f: () => void): () => void {
    this.listeners.add(f);
    return () => this.listeners.delete(f);
  }

  private emit(): void {
    for (const f of this.listeners) f();
  }

  /** Call from the tap itself: iOS only lets a user gesture start audio. */
  play(): void {
    if (!this.buffers.length || this.playing) return;
    const gen = ++this.generation;
    this.ctx.resume().catch(() => {
      // no audio (an interruption, a browser's refusal): back to paused, where it was
      if (gen === this.generation && this.playing) this.pause();
    });
    if (this.offset >= this.duration - 0.05) this.offset = 0;
    this.startSource();
    this.playing = true;
    this.emit();
  }

  pause(): void {
    if (!this.playing) return;
    this.generation++;
    this.offset = this.time();
    this.stopSource();
    this.playing = false;
    this.emit();
  }

  seek(t: number): void {
    this.generation++;
    this.offset = Math.min(this.duration, Math.max(0, t));
    this.epoch++;
    if (this.playing) {
      this.stopSource();
      this.startSource();
    }
    this.emit();
  }

  /** the audible playhead (s) */
  time(): number {
    if (!this.playing) return this.offset;
    const t = Math.min(this.duration, this.offset + Math.max(0, this.audibleNow() - this.startedAt));
    if (this.rendered && t >= this.duration) {
      // the end has been heard: stop here (listeners hear of it after the caller's frame)
      this.playing = false;
      this.offset = this.duration;
      queueMicrotask(() => this.emit());
    }
    return t;
  }

  dispose(): void {
    this.generation++;
    this.stopSource();
    for (const o of this.outs) {
      o.gain.disconnect();
      o.pan.disconnect();
    }
    this.listeners.clear();
    void this.ctx.close().catch(() => {});
  }

  /** One source per buffer, all started at the same context time and place in the track (the
   *  parts are as long as each other: the first one's end is the track's). */
  private startSource(): void {
    this.rendered = false;
    this.startedAt = this.ctx.currentTime;
    this.srcs = this.buffers.map((buffer, i) => {
      const src = this.ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(this.outs[i]?.gain ?? this.ctx.destination);
      return src;
    });
    const first = this.srcs[0];
    first.onended = () => {
      if (this.srcs[0] === first) this.rendered = true;
    };
    for (const src of this.srcs) src.start(this.startedAt, this.offset);
  }

  private stopSource(): void {
    const srcs = this.srcs;
    this.srcs = [];
    this.rendered = false;
    for (const src of srcs) {
      src.onended = null;
      try {
        src.stop();
      } catch {
        // never started (the context was closed)
      }
      src.disconnect();
    }
  }

  /** the context time being heard now */
  private audibleNow(): number {
    const c = this.ctx;
    const ts = typeof c.getOutputTimestamp === "function" ? c.getOutputTimestamp() : null;
    if (ts && ts.contextTime !== undefined && ts.performanceTime !== undefined && ts.contextTime > 0) {
      // the stamp is a moment ago; carry it to now, but never ahead of what has been rendered
      const t = ts.contextTime + Math.max(0, (performance.now() - ts.performanceTime) / 1000);
      return Math.min(t, c.currentTime);
    }
    return c.currentTime - (c.outputLatency || c.baseLatency || 0);
  }
}
