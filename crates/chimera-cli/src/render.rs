//! `chimera render …`: ports of `tools/render-{speech,text,song,beatbox,creature,vowels}.ts`.

use crate::js::{fixed, num, pad_end, pad_start};
use crate::{Ctx, FS, arg, body, body_mix, peak, repo_path, wav_name};
use chimera::analysis::lpc::{FormantOpts, estimate_formants};
use chimera::anatomy::human::{ArticKey, BodyMix, N_ARTIC};
use chimera::analysis::pitch::{YinOpts, yin_pitch};
use chimera::control::beat_player::BeatPlayer;
use chimera::control::score::Gesture;
use chimera::creatures::types::ParamRole;
use chimera::creatures::voice::CreatureVoice;
use chimera::creatures::{CREATURES, creature_by_id};
use chimera::math::{PI, cos, js_max, js_min, js_round, log10, sin};
use chimera::phonetics::beatbox::{BEAT_PRESETS, BEAT_SOUNDS, BeatStep, beat_ready, parse_beat};
use chimera::phonetics::english::EnglishG2P;
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{
    PhraseInput, PlanOptions, expand, plan_phrases, plan_utterance, vowel_table,
};
use chimera::phonetics::sing::{
    Diphthongs, LegatoFrom, NoteOnOptions, SongOptions, SungNote, midi_to_hz, plan_note_off,
    plan_note_on, plan_song,
};
use chimera::phonetics::songs::DEMO_SONGS;
use chimera::phonetics::syllables::lyrics_to_syllables;
use chimera::phonetics::{Intonation, Plan, Syllable};
use chimera::voice::human_voice::{DEFAULT_SOURCE, HumanVoice, SCOPE_POINTS, SourcePatch, VoiceTelemetry};
use std::io::Read;
use std::time::Instant;

/// Render `out.len()` samples in 128-sample blocks, as the tools did.
fn render_blocks(v: &mut HumanVoice, out: &mut [f32]) {
    for block in out.chunks_mut(128) {
        v.render(block);
    }
}

/// The CMU dictionary (`public/data/cmudict.txt.gz`) loaded into a G2P.
pub fn g2p() -> Result<EnglishG2P, String> {
    g2p_saying(None)
}

/// The dictionary with a melody's or show's own pronunciations over it: `"pronounce": {"word":
/// "W ER1 D"}` in the dictionary's notation. The dictionary is American and lists a word's
/// commonest form first: "worries" has the r-coloured vowel (British: "W AH1 R IY0 Z") and "and"
/// its weak form (sung: "AE1 N D").
fn g2p_saying(pronounce: Option<&serde_json::Value>) -> Result<EnglishG2P, String> {
    let path = repo_path("public/data/cmudict.txt.gz");
    let gz = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
    let mut text = String::new();
    flate2::read::GzDecoder::new(&gz[..])
        .read_to_string(&mut text)
        .map_err(|e| format!("{path}: {e}"))?;
    match pronounce {
        None | Some(serde_json::Value::Null) => {}
        Some(serde_json::Value::Object(words)) => {
            // later lines replace earlier ones
            for (word, phones) in words {
                let phones = phones.as_str().ok_or(format!("pronounce: {word:?} is not a string of phones"))?;
                text.push_str(&format!("\n{}\t{}", word.to_lowercase(), phones.trim()));
            }
        }
        Some(_) => return Err("pronounce: not an object of word: phones".into()),
    }
    let mut g = EnglishG2P::new();
    g.load_dict(&text);
    Ok(g)
}

/// `name.toLowerCase().replace(/[^a-z]+/g, "-")`
fn slug(name: &str) -> String {
    let mut out = String::new();
    let mut run = false;
    for c in name.to_lowercase().chars() {
        if c.is_ascii_lowercase() {
            out.push(c);
            run = false;
        } else if !run {
            out.push('-');
            run = true;
        }
    }
    out
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Mark list `ipa@t` as render-speech printed it.
fn mark_line(marks: &[(f64, String)]) -> String {
    marks
        .iter()
        .map(|(t, l)| format!("{l}@{}", fixed(*t, 3)))
        .collect::<Vec<_>>()
        .join(" ")
}

/// IPA utterances through the planner. `render speech <out> ["ˈɑpɑ / ˈɑbɑ"] [body]`
pub fn speech(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let name = arg(a, 0, "out/speech");
    let text = arg(a, 1, "ˈɑpɑ");
    let body_name = arg(a, 2, "male");
    let b = body(body_name)?;
    let f0 = match body_name {
        "female" => 200.0,
        "child" => 250.0,
        _ => 110.0,
    };
    let mut voice = HumanVoice::new(FS, b, 2);
    voice.set_source(&SourcePatch::f0(f0));
    let mut marks: Vec<(f64, String)> = Vec::new();
    let mut t0 = 0.05;
    for u in text.split('/').map(str::trim).filter(|u| !u.is_empty()) {
        let segs = tokenize_ipa(u);
        let plan = plan_utterance(
            &segs,
            &PlanOptions {
                body: body_name.into(),
                t0,
                f0: Some(f0),
                ..Default::default()
            },
        );
        voice.schedule(&plan.gestures);
        for s in &plan.segments {
            marks.push((s.t0, s.ipa.clone()));
        }
        t0 = plan.end + 0.35;
    }
    let total = js_round(FS * (t0 + 0.2)) as usize;
    let mut out = vec![0f32; total];
    render_blocks(&mut voice, &mut out);
    let file = wav_name(name);
    ctx.write_wav(&file, &out)?;
    println!(
        "wrote {file}  {} s  peak {}",
        fixed(total as f64 / FS, 2),
        fixed(peak(&out), 3)
    );
    println!("{}", mark_line(&marks));
    Ok(())
}

/// A sentence said by the text path: the samples and the plan.
pub struct Spoken {
    pub out: Vec<f32>,
    pub plan: Plan,
    /// the IPA `render text` prints, phrases joined by " | "
    pub shown: String,
}

/// The f0 `render text` speaks at in each body.
pub fn text_f0(body_name: &str) -> f64 {
    match body_name {
        "female" => 205.0,
        "child" => 250.0,
        _ => 112.0,
    }
}

/// Say `text` the way `render text` does: CMUdict G2P, the phrase planner from t0 = 50 ms, the
/// body's speaking f0, 300 ms of tail. `chimera compare tsp` measures exactly these renders.
pub fn say(g: &EnglishG2P, text: &str, body_name: &str) -> Result<Spoken, String> {
    let b = body(body_name)?;
    let f0 = text_f0(body_name);
    let phrases = g.to_phrases(text);
    let shown: Vec<String> = phrases
        .iter()
        .map(|p| {
            let s: String = p
                .segments
                .iter()
                .map(|s| format!("{}{}", if s.stress == Some(1) { "ˈ" } else { "" }, s.ipa))
                .collect();
            s + if p.intonation == Intonation::Question {
                "?"
            } else {
                ""
            }
        })
        .collect();
    let mut voice = HumanVoice::new(FS, b, 2);
    voice.set_source(&SourcePatch::f0(f0));
    let inputs: Vec<PhraseInput> = phrases
        .iter()
        .map(|p| PhraseInput {
            segments: p.segments.clone(),
            intonation: Some(p.intonation),
            pause: Some(p.pause),
        })
        .collect();
    let plan = plan_phrases(
        &inputs,
        &PlanOptions {
            body: body_name.into(),
            t0: 0.05,
            f0: Some(f0),
            ..Default::default()
        },
    );
    voice.schedule(&plan.gestures);
    let total = js_round(FS * (plan.end + 0.3)) as usize;
    let mut out = vec![0f32; total];
    render_blocks(&mut voice, &mut out);
    Ok(Spoken { out, plan, shown: shown.join(" | ") })
}

/// English text through G2P and the phrase planner. `render text <out> ["Text."] [body]`
pub fn text(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let name = arg(a, 0, "out/text");
    let text = arg(a, 1, "Hello world.");
    let body_name = arg(a, 2, "male");
    let g = g2p()?;
    let t = Instant::now();
    let s = say(&g, text, body_name)?;
    let ms = ms_since(t);
    println!("{}", s.shown);
    let file = wav_name(name);
    ctx.write_wav(&file, &s.out)?;
    println!(
        "wrote {file}  {} s, rendered in {} ms",
        fixed(s.out.len() as f64 / FS, 2),
        fixed(ms, 0)
    );
    Ok(())
}

/// The demo songs, and a live-style key sequence. `render song [outdir] [body]`
pub fn song(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let out_dir = arg(a, 0, "out/song");
    let body_name = arg(a, 1, "male");
    let b = body(body_name)?;
    let g = g2p()?;
    let syl_text = |syl: &[Syllable]| {
        syl.iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    };

    for song in DEMO_SONGS {
        let syl = lyrics_to_syllables(song.lyrics, &g);
        let octave = match body_name {
            "female" => song.octave.female,
            "child" => song.octave.child,
            _ => song.octave.male,
        };
        let shift = octave * 12.0;
        let notes: Vec<SungNote> = song
            .notes
            .iter()
            .enumerate()
            .map(|(i, n)| SungNote {
                midi: n[0] + shift,
                beats: n[1],
                syl: syl[i % syl.len()].clone(),
            })
            .collect();
        let mut v = HumanVoice::new(FS, b, 2);
        v.set_source(&SourcePatch {
            f0: Some(midi_to_hz(notes[0].midi)),
            voiced: Some(true),
            pressure: Some(9.0),
            tension: Some(0.55),
            ..Default::default()
        });
        let plan = plan_song(
            &notes,
            &SongOptions {
                body: body_name.into(),
                bpm: song.bpm,
                t0: 0.3,
                vibrato: 35.0,
                ..Default::default()
            },
        );
        v.schedule(&plan.gestures);
        let n = js_round(FS * (plan.end + 0.4)) as usize;
        let mut out = vec![0f32; n];
        render_blocks(&mut v, &mut out);
        ctx.write_wav(
            &format!("{out_dir}/{}-{body_name}.wav", slug(song.name)),
            &out,
        )?;
        println!(
            "{}: {} s, peak {}, syllables {}",
            song.name,
            fixed(n as f64 / FS, 1),
            fixed(peak(&out), 3),
            syl_text(&syl)
        );
    }

    // live-style: key presses with gaps and one legato pair, as the keyboard would send them
    let syl = lyrics_to_syllables("hello there my friend", &g);
    let mut v = HumanVoice::new(FS, b, 2);
    v.set_source(&SourcePatch {
        f0: Some(130.0),
        voiced: Some(true),
        pressure: Some(9.0),
        ..Default::default()
    });
    let keys: [(f64, f64, f64); 4] = [
        (0.2, 0.55, 57.0),
        (0.6, 1.0, 59.0),
        (1.1, 1.8, 62.0),
        (1.75, 2.6, 60.0),
    ]; // (down, up, midi)
    let n = js_round(FS * 3.0) as usize;
    let mut out = vec![0f32; n];
    let mut tag = 2.0;
    struct Held {
        syl: Syllable,
        midi: f64,
        tag: f64,
    }
    let mut held: Option<Held> = None;
    // (t, on, midi, key index), sorted by time (stable, as Array.sort)
    let mut events: Vec<(f64, bool, f64, usize)> = keys
        .iter()
        .enumerate()
        .flat_map(|(i, &(d, u, m))| [(d, true, m, i), (u, false, m, i)])
        .collect();
    events.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut e = 0;
    for block in out.chunks_mut(128) {
        let now = v.clock;
        while e < events.len() && events[e].0 <= now {
            let (_, on, m, i) = events[e];
            e += 1;
            let shift = |gs: Vec<Gesture>| -> Vec<Gesture> {
                gs.into_iter()
                    .map(|g| Gesture {
                        t0: g.t0 + now + 0.006,
                        t1: g.t1 + now + 0.006,
                        ..g
                    })
                    .collect()
            };
            if on {
                let s = &syl[i];
                if let Some(h) = &held {
                    v.release(h.tag);
                    v.release(h.tag + 1.0);
                }
                let legato_from = held.as_ref().map(|h| LegatoFrom {
                    syl: h.syl.clone(),
                    midi: h.midi,
                });
                let gs = shift(plan_note_on(
                    s,
                    m,
                    &NoteOnOptions {
                        body: body_name.into(),
                        vibrato: 35.0,
                        tag,
                        legato_from,
                    },
                ));
                v.schedule(&gs);
                held = Some(Held {
                    syl: s.clone(),
                    midi: m,
                    tag,
                });
                tag += 2.0;
            } else if let Some(h) = held.take_if(|h| h.midi == m) {
                v.release(h.tag);
                v.release(h.tag + 1.0);
                let gs = shift(plan_note_off(&h.syl, h.midi, body_name.into()));
                v.schedule(&gs);
            }
        }
        v.render(block);
    }
    ctx.write_wav(&format!("{out_dir}/live-{body_name}.wav"), &out)?;
    println!("live: syllables {}", syl_text(&syl));
    Ok(())
}

/// Articulator tracks are sampled at this rate (Hz): 480 samples at 48 kHz.
const ARTIC_RATE: usize = 100;

/// Telemetry frames (`--tele`) are taken every `TELE_HOP` samples: the live app's cadence (the
/// worklet posts one every 6 render quanta of 128), so each frame's RMS levels cover as long.
pub const TELE_HOP: usize = 768;
/// Points kept of the 256-point glottal scope in a telemetry frame (every 8th).
pub const TELE_SCOPE: usize = 32;
/// `--tele` file version (see `TeleRecorder`).
pub const TELE_VERSION: u32 = 1;

/// The voice's telemetry over time, for replaying a render in the app (two heads singing a duet).
///
/// File layout, little-endian: `b"CHTL"`, then u32 version, sampleRate, hop, frames, frameLen,
/// nArtic, nPh, nOr, nScope, then f32 bodyFemale, bodyChild (48 bytes in all); then `frames`
/// frames of `frameLen` f32s. A frame is the wasm telemetry block's fields in its order
/// (crates/chimera-wasm `engine_telemetry`): phLen, orLen, velumPort, f0, voicing, pLung, outRms,
/// noiseLevel, noiseNode, gate, clock, then artic, phAreas, orAreas, pressureRms (nPh+nOr),
/// flow (nPh+nOr), glottalArea (nScope), glottalFlow (nScope); no gesture labels.
/// Frame i is the state at sample i·hop (frame 0 before any sound); its RMS levels cover the hop
/// before it. A last partial hop gets no frame.
struct TeleRecorder {
    t: VoiceTelemetry,
    data: Vec<f32>,
    frames: u32,
    n_ph: usize,
    n_or: usize,
}

impl TeleRecorder {
    fn new(v: &mut HumanVoice) -> Self {
        let mut r = TeleRecorder { t: VoiceTelemetry::default(), data: Vec::new(), frames: 0, n_ph: 0, n_or: 0 };
        v.telemetry(&mut r.t);
        r.n_ph = r.t.ph_areas.len();
        r.n_or = r.t.or_areas.len();
        r.push_current();
        r
    }

    fn frame_len(&self) -> usize {
        11 + N_ARTIC + 3 * (self.n_ph + self.n_or) + 2 * TELE_SCOPE
    }

    fn snap(&mut self, v: &mut HumanVoice) -> Result<(), String> {
        v.telemetry(&mut self.t);
        if self.t.ph_areas.len() != self.n_ph || self.t.or_areas.len() != self.n_or {
            return Err("the airway's section count changed during the render".into());
        }
        self.push_current();
        Ok(())
    }

    fn push_current(&mut self) {
        let t = &self.t;
        let d = &mut self.data;
        d.extend([t.ph_len, t.or_len, t.velum_port, t.f0, t.voicing, t.p_lung, t.out_rms, t.noise_level, t.noise_node as f64, t.gate, t.clock].map(|x| x as f32));
        for arr in [&t.artic, &t.ph_areas, &t.or_areas, &t.pressure_rms, &t.flow] {
            d.extend_from_slice(arr);
        }
        let step = SCOPE_POINTS / TELE_SCOPE;
        for arr in [&t.glottal_area, &t.glottal_flow] {
            d.extend((0..TELE_SCOPE).map(|i| arr[i * step]));
        }
        self.frames += 1;
        debug_assert_eq!(self.data.len(), self.frames as usize * self.frame_len());
    }

    fn bytes(&self, body: BodyMix) -> Vec<u8> {
        let mut b = Vec::with_capacity(48 + 4 * self.data.len());
        b.extend_from_slice(b"CHTL");
        for x in [TELE_VERSION, FS as u32, TELE_HOP as u32, self.frames, self.frame_len() as u32, N_ARTIC as u32, self.n_ph as u32, self.n_or as u32, TELE_SCOPE as u32] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        for x in [body.female as f32, body.child as f32] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        for x in &self.data {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }
}

/// Render like `render_blocks`, recording the articulators every `ARTIC_RATE`th of a second
/// (`artic`: frame i is the state at i / ARTIC_RATE s, plus one at the end) and/or the telemetry
/// every `TELE_HOP` samples (`tele`). The two cadences are independent; taking telemetry is the
/// only thing that resets its RMS levels. The voice runs sample by sample, so the sound is the
/// same either way. `sources` change the vocal folds at given samples (in order): a melody's
/// spoken lines.
fn render_tracked(v: &mut HumanVoice, out: &mut [f32], artic: bool, tele: bool, sources: &[(usize, SourcePatch)]) -> Result<(Vec<Vec<f64>>, Option<TeleRecorder>), String> {
    let a_hop = FS as usize / ARTIC_RATE;
    let frame = |v: &HumanVoice| ArticKey::ALL.iter().map(|k| js_round(v.artic.x.get(*k) * 1e4) / 1e4).collect();
    let mut frames = if artic { vec![frame(v)] } else { Vec::new() };
    let mut rec = if tele { Some(TeleRecorder::new(v)) } else { None };
    let n = out.len();
    let (mut pos, mut next_a, mut next_t) = (0usize, a_hop, TELE_HOP);
    let mut changes = sources.iter().peekable();
    while pos < n {
        while let Some((_, patch)) = changes.next_if(|c| c.0 <= pos) {
            v.set_source(patch);
        }
        let mut end = n;
        if let Some(c) = changes.peek() {
            end = end.min(c.0);
        }
        if artic {
            end = end.min(next_a);
        }
        if tele {
            end = end.min(next_t);
        }
        render_blocks(v, &mut out[pos..end]);
        pos = end;
        if artic && (pos == next_a || pos == n) {
            frames.push(frame(v));
            next_a += a_hop;
        }
        if let Some(r) = rec.as_mut()
            && pos == next_t
        {
            r.snap(v)?;
            next_t += TELE_HOP;
        }
    }
    Ok((frames, rec))
}

/// A melody sung: the samples, and what was recorded along the way.
struct Sung {
    out: Vec<f32>,
    /// articulator frames (`--artic`)
    frames: Vec<Vec<f64>>,
    rec: Option<TeleRecorder>,
    sounding: usize,
    /// the words of the spoken lines: start, end (s), word
    spoken: Vec<(f64, f64, String)>,
}

/// A spoken line never slows below this rate to fill its time (1 is the planner's own pace).
const SLOWEST_SPEECH: f64 = 0.8;
/// The planner starts an utterance's first sound this long after its t0 (its breath's lead).
const SPEECH_LEAD: f64 = 0.06;

/// One spoken line of a melody, planned: when it sounds and its gestures.
struct Line {
    start: f64,
    end: f64,
    /// its speaking pitch (Hz)
    f0: f64,
    gestures: Vec<Gesture>,
    words: Vec<(f64, f64, String)>,
}

/// Plan a melody's spoken line: `{"at": beats from the melody's start, "beats": the time it has,
/// "text": "English words." | "ipa": "tp", "f0"?: Hz, "show"?: the word shown for an "ipa" line}`.
/// It starts where it is written and is said in the rhythm of its words, not the beat's: at the
/// planner's own pace when that fills the time, a little slower (never under
/// [`SLOWEST_SPEECH`]) when the time is longer, and as much faster as it takes to finish in time.
fn plan_line(j: &serde_json::Value, g: &EnglishG2P, body_name: &str, t0: f64, beat: f64) -> Result<Line, String> {
    let at = j["at"].as_f64().ok_or("speech: at missing")?;
    let beats = j["beats"].as_f64().filter(|b| *b > 0.0).ok_or("speech: beats missing")?;
    let f0 = j["f0"].as_f64().unwrap_or(text_f0(body_name));
    let (start, window) = (t0 + at * beat, beats * beat);
    let (inputs, said): (Vec<PhraseInput>, Vec<String>) = if let Some(ipa) = j["ipa"].as_str() {
        let mut segments = tokenize_ipa(ipa);
        if let Some(last) = segments.last_mut() {
            last.word_end = Some(true);
        }
        (vec![PhraseInput { segments, intonation: None, pause: None }], vec![j["show"].as_str().unwrap_or(ipa).to_string()])
    } else {
        let phrases = g.to_phrases(j["text"].as_str().ok_or("speech: text or ipa missing")?);
        (
            phrases.iter().map(|p| PhraseInput { segments: p.segments.clone(), intonation: Some(p.intonation), pause: Some(p.pause) }).collect(),
            phrases.iter().flat_map(|p| p.words.iter().cloned()).collect(),
        )
    };
    if inputs.iter().all(|p| p.segments.is_empty()) {
        return Err(format!("speech: nothing to say in {j}"));
    }
    let plan_at = |rate: f64| plan_phrases(&inputs, &PlanOptions { body: body_name.into(), t0: start - SPEECH_LEAD, rate: Some(rate), f0: Some(f0), ..Default::default() });
    let mut plan = plan_at(1.0);
    let natural = plan.end - start;
    if natural > window {
        // faster, until it fits (the pauses between phrases don't shrink with the rate)
        let mut rate = 1.0;
        for _ in 0..8 {
            rate *= 1.01 * (plan.end - start) / window;
            plan = plan_at(rate);
            if plan.end - start <= window {
                break;
            }
        }
    } else {
        plan = plan_at(js_max(SLOWEST_SPEECH, natural / window));
    }
    // each word's time: its sounds in the plan (a diphthong is planned as its two vowels)
    let mut words = Vec::new();
    let (mut k, mut w, mut from) = (0usize, 0usize, None);
    for seg in inputs.iter().flat_map(|p| &p.segments) {
        let n = expand(&seg.ipa).map_or(1, |parts| parts.len());
        let (Some(first), Some(last)) = (plan.segments.get(k), plan.segments.get(k + n - 1)) else { break };
        let t = *from.get_or_insert(first.t0);
        k += n;
        if seg.word_end == Some(true) {
            if let Some(word) = said.get(w) {
                words.push((t, last.t1, word.clone()));
            }
            w += 1;
            from = None;
        }
    }
    Ok(Line { start, end: plan.end, f0, gestures: plan.gestures, words })
}

/// Sing a melody: `{"bpm", "lyrics", "notes": [[midi, beats], ...], "t0"?, "vibrato"?, "source"?,
/// "speech"?, "speechSource"?, "diphthongs"?, "pitchLead"?, "pitchFollow"?, "pronounce"?}` (`pronounce`: see
/// [`g2p_saying`]; the caller loads it). One lyric syllable per sounding note; a note with midi <= 0 is a rest, which
/// splits the melody into phrases planned separately on one voice. `source` sets the vocal folds
/// for the singing (a `SourcePatch`: tension, breathiness, brightness, pressure, roughness...),
/// over the sung default (pressure 9, tension 0.55): a lax, breathy, dark source on the male body
/// is a falsetto (`{"tension": 0.2, "breathiness": 0.45, "brightness": 0.1}`).
/// `speech` is lines the voice says in rests of its notes (see [`plan_line`]), in its speaking
/// voice: the vocal folds go back to the engine's default for a line, with `speechSource` over
/// it (`{"pressure": 13}` projects the line over other singers: at the default 8 cmH2O speech is
/// some 9 dB under the singing), and to the sung source after it.
/// `diphthongs` (`{"glide": 0.5, "open": "ɑ"}`) and `pitchLead` (seconds) are the song
/// planner's options of those names (`SongOptions`): diphthongs that glide over the last part of
/// their note, from a darker vowel, and notes whose pitch is there as they start. `pitchFollow`
/// (seconds) is how closely the voice follows the written pitch (`HumanVoice::pitch_follow`, 0.03
/// unless set). Together, less of a swoop between notes: a lead of 0.05 and a follow of 0.012.
fn sing(j: &serde_json::Value, body_name: &str, g: &EnglishG2P, artic: bool, tele: bool) -> Result<Sung, String> {
    let b = body(body_name)?;
    let bpm = j["bpm"].as_f64().ok_or("bpm missing")?;
    let vibrato = j["vibrato"].as_f64().unwrap_or(35.0);
    let raw: Vec<[f64; 2]> = serde_json::from_value(j["notes"].clone()).map_err(|e| format!("notes: {e}"))?;
    let source: SourcePatch = match j.get("source") {
        Some(s) if !s.is_null() => serde_json::from_value(s.clone()).map_err(|e| format!("source: {e}"))?,
        _ => SourcePatch::default(),
    };
    let diphthongs: Option<Diphthongs> = match j.get("diphthongs") {
        Some(d) if !d.is_null() => Some(serde_json::from_value(d.clone()).map_err(|e| format!("diphthongs: {e}"))?),
        _ => None,
    };
    let pitch_lead = j["pitchLead"].as_f64();
    let syl = lyrics_to_syllables(j["lyrics"].as_str().ok_or("lyrics missing")?, g);
    let sounding = raw.iter().filter(|n| n[0] > 0.0).count();
    if syl.len() != sounding {
        return Err(format!("{} syllables for {sounding} notes: {}", syl.len(),
            syl.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ")));
    }
    let t0 = j["t0"].as_f64().unwrap_or(0.3);
    let beat = 60.0 / bpm;
    let mut v = HumanVoice::new(FS, b, 2);
    let first = raw.iter().find(|n| n[0] > 0.0).ok_or("no notes")?;
    v.set_source(&SourcePatch {
        f0: Some(midi_to_hz(first[0])),
        voiced: Some(true),
        pressure: Some(9.0),
        tension: Some(0.55),
        ..Default::default()
    });
    v.set_source(&SourcePatch { f0: None, voiced: None, ..source });
    if let Some(follow) = j["pitchFollow"].as_f64() {
        v.pitch_follow = follow;
    }
    // the spoken lines, and where the vocal folds change for them (lines close together share
    // one stretch of the speaking voice)
    let mut lines: Vec<Line> = match j.get("speech") {
        Some(serde_json::Value::Array(a)) => a.iter().map(|l| plan_line(l, g, body_name, t0, beat)).collect::<Result<_, _>>()?,
        Some(serde_json::Value::Null) | None => Vec::new(),
        Some(_) => return Err("speech: not a list".into()),
    };
    lines.sort_by(|a, b| a.start.total_cmp(&b.start));
    let folds = |s: &chimera::voice::human_voice::VoiceSource| SourcePatch { f0: None, glide: None, voiced: None, vibrato_depth: None, ..SourcePatch::from_source(s) };
    let (singing, mut speaking) = (folds(&v.source), folds(&DEFAULT_SOURCE));
    if let Some(x) = j.get("speechSource").filter(|x| !x.is_null()) {
        let patch: SourcePatch = serde_json::from_value(x.clone()).map_err(|e| format!("speechSource: {e}"))?;
        speaking.merge(&SourcePatch { f0: None, glide: None, voiced: None, vibrato_depth: None, ..patch });
    }
    let mut sources: Vec<(usize, SourcePatch)> = Vec::new();
    let mut spoken = Vec::new();
    let mut until = f64::NEG_INFINITY;
    for l in &lines {
        v.schedule(&l.gestures);
        spoken.extend(l.words.iter().cloned());
        let at = |t: f64| js_round(FS * js_max(0.0, t)) as usize;
        if l.start - 0.03 > until + 0.25 {
            // with the voice's resting pitch at the line's: the planner's pitch gestures blend
            // with it wherever they are weak (between words, at a phrase's edges), and the
            // resting pitch of a sung melody is its first note's, which for a high voice pulled
            // every gap in a spoken line up towards it
            // The resting pitch goes first, in the silence before the line (after the last
            // change, 0.3 s ahead at most): it glides, and gliding from a sung note's pitch as
            // the line began made each line start with a swoop down.
            let early = sources.last().map_or(0, |c| c.0).max(at(l.start - 0.3));
            sources.push((early, SourcePatch { f0: Some(l.f0), ..Default::default() }));
            sources.push((at(l.start - 0.03).max(early), speaking));
            sources.push((at(l.end + 0.05), SourcePatch { f0: Some(v.source.f0), ..singing }));
        } else if let Some(last) = sources.last_mut() {
            last.0 = at(l.end + 0.05);
        }
        until = js_max(until, l.end + 0.05);
    }
    let (mut t, mut k, mut end) = (t0, 0usize, lines.iter().fold(0.0f64, |e, l| js_max(e, l.end)));
    let mut phrase: Vec<SungNote> = Vec::new();
    let mut start = t;
    let flush = |phrase: &mut Vec<SungNote>, start: f64, v: &mut HumanVoice, end: &mut f64| {
        if phrase.is_empty() {
            return;
        }
        let plan = plan_song(phrase, &SongOptions { body: body_name.into(), bpm, t0: start, vibrato, diphthongs: diphthongs.clone(), pitch_lead });
        v.schedule(&plan.gestures);
        *end = js_max(*end, plan.end);
        phrase.clear();
    };
    for n in &raw {
        if n[0] > 0.0 {
            if phrase.is_empty() {
                start = t;
            }
            phrase.push(SungNote { midi: n[0], beats: n[1], syl: syl[k].clone() });
            k += 1;
        } else {
            flush(&mut phrase, start, &mut v, &mut end);
        }
        t += n[1] * beat;
    }
    flush(&mut phrase, start, &mut v, &mut end);
    let n = js_round(FS * (end + 0.4)) as usize;
    let mut out = vec![0f32; n];
    let (frames, rec) = if artic || tele || !sources.is_empty() {
        render_tracked(&mut v, &mut out, artic, tele, &sources)?
    } else {
        render_blocks(&mut v, &mut out);
        (Vec::new(), None)
    };
    Ok(Sung { out, frames, rec, sounding, spoken })
}

/// `render notes <in.json> <out> [body] [--artic tracks.json] [--tele voice.tele]`: sing a melody
/// file (see [`sing`]). `--artic` also writes the articulators over time (`{"rate", "body", "keys",
/// "frames"}`), to animate a face in sync; `--tele` writes the voice's whole telemetry over time
/// (see `TeleRecorder`), which the app's duet page replays on a 3D head.
pub fn notes(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let mut artic_out: Option<String> = None;
    let mut tele_out: Option<String> = None;
    let mut rest = Vec::new();
    let mut it = a.iter();
    while let Some(x) = it.next() {
        if x == "--artic" {
            artic_out = Some(it.next().ok_or("--artic needs a file")?.clone());
        } else if x == "--tele" {
            tele_out = Some(it.next().ok_or("--tele needs a file")?.clone());
        } else {
            rest.push(x.clone());
        }
    }
    let a = &rest[..];
    let input = a.first().ok_or("render notes <in.json> <out> [male|female|child] [--artic tracks.json] [--tele voice.tele]")?;
    let out_name = wav_name(arg(a, 1, "out/notes"));
    let body_name = arg(a, 2, "male");
    let text = std::fs::read_to_string(input).map_err(|e| format!("{input}: {e}"))?;
    let j: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{input}: {e}"))?;
    let Sung { out, frames, rec, sounding, spoken } = sing(&j, body_name, &g2p_saying(j.get("pronounce"))?, artic_out.is_some(), tele_out.is_some())?;
    ctx.write_wav(&out_name, &out)?;
    println!("{out_name}: {} s, peak {}, {} notes{}", fixed(out.len() as f64 / FS, 2), fixed(peak(&out), 3), sounding, if spoken.is_empty() { String::new() } else { format!(", {} spoken words", spoken.len()) });
    if let Some(path) = &artic_out {
        let j = serde_json::json!({"rate": ARTIC_RATE, "body": body_name, "keys": ArticKey::NAMES, "frames": frames});
        std::fs::write(path, j.to_string()).map_err(|e| format!("{path}: {e}"))?;
        println!("{path}: {} frames at {ARTIC_RATE} Hz", frames.len());
    }
    if let (Some(path), Some(r)) = (&tele_out, rec) {
        std::fs::write(path, r.bytes(body_mix(body_name)?)).map_err(|e| format!("{path}: {e}"))?;
        println!("{path}: {} telemetry frames every {TELE_HOP} samples", r.frames);
    }
    Ok(())
}

/// The level of a voice while it sounds (dB re full scale): the RMS of its 20 ms frames within
/// 40 dB of the loudest, so rests don't count.
fn sounding_db(x: &[f32]) -> f64 {
    let frames: Vec<f64> = x.chunks(FS as usize / 50).map(|c| c.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / c.len() as f64).collect();
    let top = frames.iter().fold(0.0f64, |p, &v| js_max(p, v));
    let loud: Vec<f64> = frames.into_iter().filter(|&v| v > top * 1e-4).collect();
    if loud.is_empty() {
        return f64::NEG_INFINITY;
    }
    10.0 * log10(loud.iter().sum::<f64>() / loud.len() as f64)
}

/// `render show <show.json> <outdir>`: a whole show for the app's duet page from one file, any
/// number of voices. With `<outdir>` = `public/shows/<id>` it plays at `?show=<id>`.
///
///   { "title": "…", "bpm": 120, "t0"?: 0.3, "downbeat"?: s, "spell"?: {"dum": "dm"}, "bowTie"?: true,
///     "pronounce"?: {"worries": "W AH1 R IY0 Z"},
///     "voices": [{ "name": "Bass", "lyrics": "…", "notes": [[midi, beats], …],
///                  "body"?: "male", "source"?: {…}, "speech"?: […], "vibrato"?: 35, "gain"?: 1, "pan"?: 0, "levels"?: […] }, …] }
///
/// Each voice is a melody as `render notes` sings it (see [`sing`]), all on the show's `bpm` and
/// `t0`, so they share sample 0. Written: per voice `<name>.json` (its melody, for the words at
/// its mouth) and `<name>.tele`; `mix.wav` (stereo: each voice at its `gain`, panned −1 left … 1
/// right, the sum scaled to a −1 dB peak); and `show.json`. The voices render side by side.
/// A voice's `"levels": [[beat, level], …]` moves it in the mix as the piece goes (straight lines
/// between the breakpoints, on top of `gain`): the voice with the tune up, the ones humming under
/// it down.
pub fn show(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    const USAGE: &str = "render show <show.json> <outdir>";
    let input = a.first().ok_or(USAGE)?;
    let out_dir = a.get(1).ok_or(USAGE)?;
    let text = std::fs::read_to_string(input).map_err(|e| format!("{input}: {e}"))?;
    let j: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{input}: {e}"))?;
    let title = j["title"].as_str().ok_or("title missing")?;
    let bpm = j["bpm"].as_f64().ok_or("bpm missing")?;
    let t0 = j["t0"].as_f64().unwrap_or(0.3);
    struct Voice {
        name: String,
        id: String,
        body: String,
        melody: serde_json::Value,
        gain: f64,
        pan: f64,
        /// (time s, level) breakpoints
        levels: Vec<(f64, f64)>,
    }
    let mut voices: Vec<Voice> = Vec::new();
    for (i, v) in j["voices"].as_array().ok_or("voices missing")?.iter().enumerate() {
        let name = v["name"].as_str().ok_or(format!("voice {}: name missing", i + 1))?;
        // the files' name: the voice's, in lower-case letters and digits ("Alto 1" → alto-1)
        let id = name.to_lowercase().split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).collect::<Vec<_>>().join("-");
        if id.is_empty() || matches!(id.as_str(), "show" | "mix") || voices.iter().any(|o| o.id == id) {
            return Err(format!("voice {name:?}: its files would be named {id:?}, which is taken or empty"));
        }
        let body_name = v["body"].as_str().unwrap_or("male");
        body(body_name)?;
        let mut melody = serde_json::json!({"bpm": bpm, "t0": t0, "lyrics": v["lyrics"], "notes": v["notes"]});
        for k in ["vibrato", "source", "speech", "speechSource", "diphthongs", "pitchLead", "pitchFollow"] {
            if let Some(x) = v.get(k) {
                melody[k] = x.clone();
            }
        }
        // the show's pronunciations go with each melody, so `render notes` sings it the same
        if let Some(x) = j.get("pronounce") {
            melody["pronounce"] = x.clone();
        }
        let levels: Vec<[f64; 2]> = match v.get("levels") {
            Some(x) if !x.is_null() => serde_json::from_value(x.clone()).map_err(|e| format!("{name}: levels: {e}"))?,
            _ => Vec::new(),
        };
        if levels.windows(2).any(|w| w[1][0] < w[0][0]) {
            return Err(format!("{name}: levels: the beats must not go back"));
        }
        let levels = levels.iter().map(|l| (t0 + l[0] * 60.0 / bpm, l[1])).collect();
        voices.push(Voice { name: name.into(), id, body: body_name.into(), melody, gain: v["gain"].as_f64().unwrap_or(1.0), pan: v["pan"].as_f64().unwrap_or(0.0).clamp(-1.0, 1.0), levels });
    }
    if voices.is_empty() {
        return Err("a show needs a voice".into());
    }
    let g = g2p_saying(j.get("pronounce"))?;
    let t_start = Instant::now();
    let sung: Vec<Result<Sung, String>> = std::thread::scope(|s| {
        let jobs: Vec<_> = voices.iter().map(|v| s.spawn(|| sing(&v.melody, &v.body, &g, false, true))).collect();
        jobs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("the render panicked".into()))).collect()
    });
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{out_dir}: {e}"))?;
    let mut parts: Vec<Sung> = Vec::new();
    for (v, r) in voices.iter().zip(sung) {
        let r = r.map_err(|e| format!("{}: {e}", v.name))?;
        let (tele, melody) = (format!("{out_dir}/{}.tele", v.id), format!("{out_dir}/{}.json", v.id));
        let rec = r.rec.as_ref().ok_or("no telemetry")?;
        std::fs::write(&tele, rec.bytes(body_mix(&v.body)?)).map_err(|e| format!("{tele}: {e}"))?;
        // with the times its spoken words took, which only the render knows: the page writes them
        // at the mouth as it does the sung syllables
        let mut written = v.melody.clone();
        if !r.spoken.is_empty() {
            let ms = |t: f64| js_round(t * 1000.0) / 1000.0;
            written["spoken"] = r.spoken.iter().map(|(a, b, w)| serde_json::json!([ms(*a), ms(*b), w])).collect();
        }
        std::fs::write(&melody, written.to_string()).map_err(|e| format!("{melody}: {e}"))?;
        println!("{}  {} notes  {} s  {} dB while sounding (peak {})", pad_end(&v.name, 12), pad_start(&r.sounding.to_string(), 5), fixed(r.out.len() as f64 / FS, 1), fixed(sounding_db(&r.out), 1), fixed(peak(&r.out), 3));
        parts.push(r);
    }
    // the mix: constant-power panning, then one scale for the whole track
    let n = parts.iter().map(|p| p.out.len()).max().unwrap_or(0);
    let mut mix = vec![0f64; 2 * n];
    for (v, p) in voices.iter().zip(&parts) {
        let a = (v.pan + 1.0) * PI / 4.0;
        let (l, r) = (v.gain * cos(a), v.gain * sin(a));
        // the voice's level over time: straight lines between its breakpoints, held outside them
        let mut at = 0usize;
        for (i, &x) in p.out.iter().enumerate() {
            let t = i as f64 / FS;
            while at < v.levels.len() && v.levels[at].0 <= t {
                at += 1;
            }
            let level = match (at.checked_sub(1).map(|k| v.levels[k]), v.levels.get(at)) {
                (Some((t0, g0)), Some(&(t1, g1))) => g0 + (g1 - g0) * (t - t0) / (t1 - t0),
                (Some((_, g)), None) | (None, Some(&(_, g))) => g,
                (None, None) => 1.0,
            };
            mix[2 * i] += l * level * x as f64;
            mix[2 * i + 1] += r * level * x as f64;
        }
    }
    let top = mix.iter().fold(0.0f64, |p, &v| js_max(p, v.abs()));
    let k = if top > 0.0 { 0.891 / top } else { 1.0 };
    let mix: Vec<f32> = mix.iter().map(|&v| (v * k) as f32).collect();
    let audio = format!("{out_dir}/mix.wav");
    ctx.write_wav_stereo(&audio, &mix)?;
    let mut show = serde_json::json!({
        "title": title,
        "audio": "mix.wav",
        "bpm": bpm,
        "downbeat": j["downbeat"].as_f64().unwrap_or(t0),
        "voices": voices.iter().map(|v| serde_json::json!({"name": v.name, "tele": format!("{}.tele", v.id), "melody": format!("{}.json", v.id)})).collect::<Vec<_>>(),
    });
    // what only the page reads goes through as it is
    for k in ["spell", "bowTie"] {
        if let Some(x) = j.get(k) {
            show[k] = x.clone();
        }
    }
    let path = format!("{out_dir}/show.json");
    std::fs::write(&path, show.to_string()).map_err(|e| format!("{path}: {e}"))?;
    println!("{audio}: {} voices, {} s, mixed {} dB, rendered in {} s", voices.len(), fixed(n as f64 / FS, 1), fixed(20.0 * log10(k), 1), fixed(ms_since(t_start) / 1000.0, 1));
    Ok(())
}

/// The beatbox kit, each sound alone, and a preset pattern through the sequencer.
/// `render beatbox [outdir] [preset name | pattern] [body] [bpm] [bars]`
pub fn beatbox(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let out_dir = arg(a, 0, "out/beatbox");
    let preset_name = arg(a, 1, "Boots and cats");
    let body_name = arg(a, 2, "male");
    let b = body(body_name)?;
    let voice = || {
        let mut v = HumanVoice::new(FS, b, 2);
        v.set_articulation(&beat_ready(body_name.into()), None);
        v.set_source(&SourcePatch {
            f0: Some(100.0),
            voiced: Some(true),
            pressure: Some(8.0),
            ..Default::default()
        });
        v.set_gate(0.0);
        v
    };
    let at = |t: f64| js_round(t * FS) as usize;

    // ---- each sound alone
    let mut player = BeatPlayer::new();
    player.body = body_name.into();
    let mut rows: Vec<String> = Vec::new();
    let n = js_round(FS * 0.5) as usize;
    let mut strip = vec![0f32; js_round(FS * 0.5 * BEAT_SOUNDS.len() as f64) as usize];
    for (idx, s) in BEAT_SOUNDS.iter().enumerate() {
        let mut v = voice();
        let hit = 0.25;
        let mut gs = Vec::new();
        player.gestures(&BeatStep::new(s.id), hit, 0.15, &mut gs);
        v.schedule(&gs);
        let mut out = vec![0f32; n];
        render_blocks(&mut v, &mut out);
        let abs = |i: usize| (out[i] as f64).abs();
        // the event: its peak within [hit − 30 ms, hit + 80 ms]; onset = first sample from
        // hit − 40 ms that reaches a quarter of that peak
        let mut pk = 0.0;
        for i in at(hit - 0.03)..at(hit + 0.08) {
            pk = js_max(pk, abs(i));
        }
        let mut on = at(hit);
        for i in at(hit - 0.04)..at(hit + 0.08) {
            if abs(i) > 0.25 * pk {
                on = i;
                break;
            }
        }
        // energy before the hit (set-up noise) relative to the hit
        let mut pre = 0.0;
        for i in at(hit - 0.15)..at(hit - 0.04) {
            pre = js_max(pre, abs(i));
        }
        // what follows: level 60–150 ms after the hit (unwanted voicing shows up here)
        let mut tail = 0.0;
        for i in at(hit + 0.06)..at(hit + 0.15) {
            tail = js_max(tail, abs(i));
        }
        rows.push(format!(
            "{} {} onset {} ms  peak {}  set-up {} dB  tail {} dB  lead-in {} ms",
            pad_end(s.id, 4),
            pad_end(s.name, 13),
            pad_start(&fixed((on as f64 / FS - hit) * 1000.0, 1), 6),
            fixed(pk, 3),
            fixed(20.0 * log10(js_max(1e-6, pre) / pk), 1),
            fixed(20.0 * log10(js_max(1e-6, tail) / pk), 1),
            fixed(BeatPlayer::lead_in(s.id, body_name.into()) * 1000.0, 0),
        ));
        strip[idx * n..(idx + 1) * n].copy_from_slice(&out);
        let file: String =
            s.id.chars()
                .map(|c| if c.is_ascii_alphabetic() { c } else { '_' })
                .collect();
        ctx.write_wav(&format!("{out_dir}/{file}.wav"), &out)?;
    }
    println!("{}", rows.join("\n"));
    ctx.write_wav(&format!("{out_dir}/kit.wav"), &strip)?;

    // ---- a pattern, two bars, through the sequencer exactly as the worklet ran it
    // a preset by name, or a pattern in beat notation (with optional bpm and bar count)
    let preset = BEAT_PRESETS
        .iter()
        .find(|p| p.name.to_lowercase() == preset_name.to_lowercase());
    let (name, text, mut bpm, swing) = match preset {
        Some(p) => (p.name, p.text, p.bpm, p.swing),
        None if parse_beat(preset_name).iter().any(Option::is_some) => ("pattern", preset_name, 100.0, 0.0),
        None => (BEAT_PRESETS[0].name, BEAT_PRESETS[0].text, BEAT_PRESETS[0].bpm, BEAT_PRESETS[0].swing),
    };
    if let Some(b) = a.get(3) {
        bpm = b.parse().map_err(|_| format!("bad bpm {b:?}"))?;
    }
    let bars: f64 = match a.get(4) {
        Some(n) => n.parse().map_err(|_| format!("bad bar count {n:?}"))?,
        None => 2.0,
    };
    let mut v = voice();
    let mut seq = BeatPlayer::new();
    seq.body = body_name.into();
    seq.set_steps(&parse_beat(text));
    seq.set_tempo(bpm, swing);
    seq.start(v.clock);
    let total = js_round(
        FS * (BeatPlayer::HORIZON + bars * seq.steps().len() as f64 * seq.step_dur() + 0.3),
    ) as usize;
    let mut out = vec![0f32; total];
    for block in out.chunks_mut(128) {
        let clock = v.clock;
        seq.tick(clock, &mut |gs: &[Gesture]| v.schedule(gs));
        v.render(block);
    }
    let file = format!("{out_dir}/{}.wav", slug(name));
    ctx.write_wav(&file, &out)?;
    println!(
        "pattern \"{}\": {} s, peak {} -> {file}",
        name,
        fixed(total as f64 / FS, 2),
        fixed(peak(&out), 3)
    );
    Ok(())
}

/// A creature's sustained note and calls; level, stability and pitch.
/// `render creature [creatureId] [outdir]`
pub fn creature(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let id = arg(a, 0, "songbird");
    let out_dir = arg(a, 1, "out/creatures");
    let def = creature_by_id(id).ok_or_else(|| {
        format!(
            "unknown creature {id}; have {}",
            CREATURES
                .iter()
                .map(|c| c.id)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;
    let mut v = CreatureVoice::new(def, FS, &[]);
    let mut out: Vec<f32> = Vec::new();
    let t = Instant::now();
    let render_for =
        |v: &mut CreatureVoice, out: &mut Vec<f32>, sec: f64| -> std::ops::Range<usize> {
            let n = js_round(sec * FS) as usize;
            let start = out.len();
            out.resize(start + n, 0.0);
            for block in out[start..].chunks_mut(128) {
                v.render(block);
            }
            start..start + n
        };
    // sustained note at the default pitch
    let pitch_param = def.params.iter().find(|p| p.role == Some(ParamRole::Pitch));
    let voices = def.param_index("voices");
    if let Some(k) = voices {
        v.set_params(&[(k, 0.0)]); // single voice for the pitch check
    }
    v.set_gate(1.0);
    let sus = render_for(&mut v, &mut out, 1.2);
    v.set_gate(0.0);
    render_for(&mut v, &mut out, 0.4);
    if let Some(k) = voices {
        v.set_params(&[(k, 0.5)]);
    }
    let mut pitch_report = String::new();
    if let Some(p) = pitch_param {
        let s = sus.start + js_round(0.6 * FS) as usize;
        let frame: Vec<f64> = out[s..s + 4096].iter().map(|&x| x as f64).collect();
        let est = yin_pitch(
            &frame,
            FS,
            YinOpts {
                min_f0: Some(js_max(20.0, p.min * 0.5)),
                max_f0: Some(js_min(12000.0, p.max * 2.0)),
                ..Default::default()
            },
        );
        pitch_report = format!(
            "target {} Hz, measured {} Hz (aperiodicity {})",
            num(p.default),
            fixed(est.f0, 1),
            fixed(est.aperiodicity, 2)
        );
    }
    for call in &def.calls {
        v.play_call_id(call.id, None, 1.0);
        let dur = call.keys[call.keys.len() - 1].t + 0.5;
        render_for(&mut v, &mut out, dur);
    }
    let ms = ms_since(t);
    let total = out.len() as f64;
    let bad = out.iter().filter(|x| !x.is_finite()).count();
    let file = format!("{out_dir}/{id}.wav");
    ctx.write_wav(&file, &out)?;
    println!(
        "{}: {} s in {} ms ({}% RT, internal {} Hz)  peak {}  non-finite {bad}",
        def.name,
        fixed(total / FS, 1),
        fixed(ms, 0),
        fixed(ms / 1000.0 / (total / FS) * 100.0, 1),
        num(v.fs),
        fixed(peak(&out), 3)
    );
    println!("  {pitch_report}");
    Ok(())
}

/// The calibrated vowels through the real-time voice; LPC formants and YIN f0 from the audio.
/// `render vowels [body] [out]`
pub fn vowels(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let body_name = arg(a, 0, "male");
    let b = body(body_name)?;
    let default_out = format!("out/vowels-{body_name}");
    let name = arg(a, 1, &default_out);
    let f0 = match body_name {
        "female" => 205.0,
        "child" => 260.0,
        _ => 110.0,
    };
    let mut voice = HumanVoice::new(FS, b, 2);
    voice.set_source(&SourcePatch::f0(f0));
    let list = ["i", "e", "ɛ", "æ", "ɑ", "ɔ", "o", "u", "ʌ", "ɝ"];
    let seg = 0.6;
    let mut out = vec![0f32; js_round(FS * seg * list.len() as f64) as usize];
    let table = &vowel_table()[body_name];
    let t = Instant::now();
    let block = 128;
    let mut pos = 0;
    'vowels: for ipa in list {
        voice.set_articulation(&table[ipa].artic, Some(true));
        let n = js_round(FS * seg) as usize;
        let mut i = 0;
        while i < n {
            let local = i as f64 / FS;
            voice.set_gate(if local < seg - 0.12 { 1.0 } else { 0.0 });
            let len = block.min(out.len() - pos);
            voice.render(&mut out[pos..pos + len]);
            pos += block;
            if pos >= out.len() {
                break 'vowels;
            }
            i += block;
        }
    }
    let ms = ms_since(t);
    let secs = out.len() as f64 / FS;
    println!(
        "{body_name}: rendered {} s in {} ms → {}% of real time; peak {}",
        fixed(secs, 1),
        fixed(ms, 0),
        fixed(ms / 1000.0 / secs * 100.0, 1),
        fixed(peak(&out), 3)
    );
    for (k, ipa) in list.iter().enumerate() {
        let c = js_round(FS * (k as f64 * seg + 0.3)) as usize;
        let frame = &out[c..c + 1536];
        let ds: Vec<f64> = (0..384)
            .map(|i| {
                (frame[4 * i] as f64
                    + frame[4 * i + 1] as f64
                    + frame[4 * i + 2] as f64
                    + frame[4 * i + 3] as f64)
                    / 4.0
            })
            .collect();
        let fm = estimate_formants(
            &ds,
            FS / 4.0,
            FormantOpts {
                order: Some(12),
                ..Default::default()
            },
        );
        let y: Vec<f64> = out[c..c + 2400].iter().map(|&x| x as f64).collect();
        let est = yin_pitch(
            &y,
            FS,
            YinOpts {
                min_f0: Some(60.0),
                ..Default::default()
            },
        );
        let reference = &table[*ipa].formants;
        let lpc: String = fm
            .iter()
            .take(3)
            .map(|f| pad_start(&fixed(f.freq, 0), 5))
            .collect();
        let model: String = reference
            .iter()
            .take(3)
            .map(|f| pad_start(&num(*f), 5))
            .collect();
        println!(
            "  {} f0={}  LPC {lpc}   model {model}",
            pad_end(ipa, 2),
            fixed(est.f0, 1)
        );
    }
    ctx.write_wav(&wav_name(name), &out)?;
    Ok(())
}
