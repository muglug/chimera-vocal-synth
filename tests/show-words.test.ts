/**
 * The words at a duet head's mouth: sung syllables over their notes, and the words of spoken lines
 * at the times the render wrote for them.
 */
import { describe, expect, test } from "bun:test";
import type { Syllable } from "../src/wasm/types.ts";
import { sungWords, wordAt, type Melody } from "../src/show/words.ts";

const syl = (text: string, wordEnd = true, melisma = false) => ({ text, wordEnd, melisma }) as unknown as Syllable;
// 120 bpm from 1 s: notes at 1-1.5 and 1.5-2 s, four beats' rest, a note at 4-5 s
const melody: Melody = { bpm: 120, t0: 1, lyrics: "", notes: [[60, 1], [62, 1], [0, 4], [64, 2]] };

describe("a show's words", () => {
  test("sung syllables stand over their notes", () => {
    const w = sungWords(melody, [syl("hel", false), syl("lo"), syl("there")]);
    expect(w.shown).toEqual(["hel-", "lo", "there"]);
    expect(wordAt(w, 1.2)).toEqual({ key: 0, text: "hel-" });
    expect(wordAt(w, 3)).toBeNull();
    expect(wordAt(w, 4.5)).toEqual({ key: 2, text: "there" });
  });

  test("spoken words sit among them at the render's times, each its own word", () => {
    const spoken: Melody["spoken"] = [[2.4, 2.7, "you"], [2.7, 3.1, "bet"]];
    const w = sungWords({ ...melody, spoken }, [syl("hel", false), syl("lo"), syl("there")]);
    expect(w.shown).toEqual(["hel-", "lo", "you", "bet", "there"]);
    expect(wordAt(w, 1.7)).toEqual({ key: 1, text: "lo" });
    expect(wordAt(w, 2.2)).toBeNull();
    expect(wordAt(w, 2.5)).toEqual({ key: -1, text: "you" });
    expect(wordAt(w, 2.9)).toEqual({ key: -2, text: "bet" });
    expect(wordAt(w, 3.5)).toBeNull();
    // the sung note after the line keeps its own key
    expect(wordAt(w, 4.5)).toEqual({ key: 2, text: "there" });
  });
});
