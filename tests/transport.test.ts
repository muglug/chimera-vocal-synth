/**
 * The duet page's transport on a fake AudioContext whose clock and output latency the test sets:
 * the playhead is what the listener hears, Play never moves it back, the end comes when it's heard,
 * and audio that can't resume leaves the transport paused.
 */
import { describe, expect, test } from "bun:test";
import { Transport } from "../src/show/transport.ts";

class FakeSource {
  onended: (() => void) | null = null;
  started: [number, number] | null = null;
  stopped = false;
  buffer: unknown = null;
  to: unknown = null;
  connect(n: unknown) {
    this.to = n;
  }
  disconnect() {}
  start(when: number, offset: number) {
    this.started = [when, offset];
  }
  stop() {
    this.stopped = true;
  }
}

class FakeParam {
  value: number;
  constructor(v: number) {
    this.value = v;
  }
  setTargetAtTime(v: number) {
    this.value = v;
  }
}

class FakeNode {
  gain = new FakeParam(1);
  pan = new FakeParam(0);
  to: unknown = null;
  connect(n: unknown) {
    this.to = n;
  }
  disconnect() {}
}

class FakeContext {
  currentTime = 0;
  nodes: FakeNode[] = [];
  createGain() {
    const n = new FakeNode();
    this.nodes.push(n);
    return n;
  }
  createStereoPanner() {
    return this.createGain();
  }
  outputLatency = 0.2;
  baseLatency = 0;
  state = "suspended";
  onstatechange: (() => void) | null = null;
  destination = {};
  sources: FakeSource[] = [];
  resumeResult: "ok" | "fail" = "ok";
  createBufferSource() {
    const s = new FakeSource();
    this.sources.push(s);
    return s;
  }
  async decodeAudioData() {
    return { duration: 100 };
  }
  async resume() {
    if (this.resumeResult === "fail") throw new Error("interrupted");
    this.setState("running");
  }
  async close() {
    this.setState("closed");
  }
  setState(s: string) {
    this.state = s;
    this.onstatechange?.();
  }
}

async function make() {
  const ctx = new FakeContext();
  const t = new Transport(ctx as unknown as AudioContext);
  await t.load(new ArrayBuffer(0));
  return { ctx, t };
}

describe("Transport", () => {
  test("the playhead is the audible time: still where Play was pressed until the output catches up", async () => {
    const { ctx, t } = await make();
    t.seek(60);
    ctx.currentTime = 5;
    t.play();
    expect(ctx.sources.at(-1)!.started).toEqual([5, 60]);
    expect(t.time()).toBe(60); // not 59.8: the first 0.2 s are still in the output buffer
    ctx.currentTime = 5.7;
    expect(t.time()).toBeCloseTo(60.5, 6);
  });

  test("tapping Play and Pause quickly never moves the playhead back", async () => {
    const { ctx, t } = await make();
    t.seek(60);
    for (let i = 0; i < 5; i++) {
      t.play();
      ctx.currentTime += 0.05;
      t.pause();
    }
    expect(t.time()).toBe(60);
  });

  test("the end comes when it is heard, not when the source finishes rendering", async () => {
    const { ctx, t } = await make();
    t.seek(99);
    t.play();
    ctx.currentTime = 1.0;
    ctx.sources.at(-1)!.onended!(); // rendered; 0.2 s still to come out of the speakers
    expect(t.isPlaying).toBe(true);
    expect(t.time()).toBeCloseTo(99.8, 6);
    ctx.currentTime = 1.25;
    expect(t.time()).toBe(100);
    expect(t.isPlaying).toBe(false);
    // Play at the end starts again from the top
    t.play();
    expect(ctx.sources.at(-1)!.started![1]).toBe(0);
  });

  test("a seek while playing restarts the source there and bumps the epoch", async () => {
    const { ctx, t } = await make();
    t.play();
    const first = ctx.sources.at(-1)!;
    const e = t.epoch;
    ctx.currentTime = 2;
    t.seek(30);
    expect(first.stopped).toBe(true);
    expect(ctx.sources.at(-1)!.started).toEqual([2, 30]);
    expect(t.epoch).toBe(e + 1);
    // the old source's end means nothing now
    first.onended?.();
    ctx.currentTime = 100;
    expect(t.isPlaying).toBe(true);
  });

  test("audio that can't resume leaves the transport paused where it was", async () => {
    const { ctx, t } = await make();
    ctx.resumeResult = "fail";
    t.seek(10);
    t.play();
    await Promise.resolve();
    await Promise.resolve();
    expect(t.isPlaying).toBe(false);
    expect(t.time()).toBe(10);
  });

  test("an interruption while playing pauses it", async () => {
    const { ctx, t } = await make();
    t.play();
    await Promise.resolve();
    expect(ctx.state).toBe("running");
    ctx.currentTime = 3;
    ctx.setState("interrupted");
    expect(t.isPlaying).toBe(false);
    expect(t.time()).toBeCloseTo(2.8, 6);
  });

  test("a show loaded as its parts starts them together, each through its own gain and pan, and mutes them one by one", async () => {
    const ctx = new FakeContext();
    const t = new Transport(ctx as unknown as AudioContext);
    expect(t.canMute).toBe(false);
    await t.loadParts([new ArrayBuffer(0), new ArrayBuffer(0), new ArrayBuffer(0)], [-0.75, 0, 2]);
    expect(t.canMute).toBe(true);
    expect(t.duration).toBe(100);
    // gain then pan per part, the pan clamped to the far right
    const [g0, p0, g1, , g2, p2] = ctx.nodes;
    expect([p0.pan.value, p2.pan.value]).toEqual([-0.75, 1]);
    expect([g0.to, p0.to]).toEqual([p0, ctx.destination]);

    t.seek(20);
    ctx.currentTime = 3;
    t.play();
    const playing = ctx.sources.slice(-3);
    expect(playing.map((s) => s.started)).toEqual([[3, 20], [3, 20], [3, 20]]);
    expect(playing.map((s) => s.to)).toEqual([g0, g1, g2]);

    // muting the middle part silences only it, without touching what plays; listeners hear of it
    let told = 0;
    t.subscribe(() => told++);
    t.setMuted(1, true);
    expect([g0.gain.value, g1.gain.value, g2.gain.value]).toEqual([1, 0, 1]);
    expect([t.isMuted(0), t.isMuted(1), told]).toEqual([false, true, 1]);
    expect(playing.every((s) => !s.stopped)).toBe(true);
    // a seek restarts every part at the new place, the muted one still muted
    ctx.currentTime = 4;
    t.seek(50);
    expect(playing.every((s) => s.stopped)).toBe(true);
    expect(ctx.sources.slice(-3).map((s) => s.started)).toEqual([[4, 50], [4, 50], [4, 50]]);
    expect(g1.gain.value).toBe(0);
    t.setMuted(1, false);
    expect(g1.gain.value).toBe(1);
    // the track ends when the first part's end has been heard
    t.seek(99);
    ctx.currentTime = 5;
    ctx.sources.at(-3)!.onended!();
    ctx.currentTime = 6.25;
    expect(t.time()).toBe(100);
    expect(t.isPlaying).toBe(false);
  });

  test("a mix has no parts to mute", async () => {
    const { ctx, t } = await make();
    expect(t.canMute).toBe(false);
    t.setMuted(0, true);
    expect(t.isMuted(0)).toBe(false);
    t.play();
    expect(ctx.sources.at(-1)!.to).toBe(ctx.destination);
  });
});
