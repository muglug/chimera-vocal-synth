//! Singing: syllables on notes, through the same gestural planner as speech.
//!  - A note's vowel lands on the beat. Its onset consonants are sung just before it (they take
//!    their time from the previous note, as singers do), its coda just before the next onset.
//!  - Diphthongs sustain their first vowel and glide to the second at the end of the note.
//!  - A melisma ("-") carries the vowel onto further notes; the coda waits for the last of them.
//!  - Vibrato blooms a quarter of a second into a held vowel.
//!
//! Live playing uses open-ended notes: a key press sings the onset and starts a vowel that lasts
//! until the key is released (tagged gestures, ended with GestureScore.release), and the release
//! sings the glide and the coda.
//!
//! These planners allocate (they run off the audio thread).

use super::inventory::{Manner, Voicing, phoneme};
use crate::anatomy::human::BodyMix;
use super::planner::{PlanOptions, artic_targets, diphthong_parts, is_vowel, plan_utterance, vowel_artic};
use crate::anatomy::human::ArticPatch;
use std::collections::BTreeMap;
use super::{Intonation, Plan, PlannedSegment, Segment, Syllable};
use crate::control::score::{ControlKey, Gesture, Targets};
use crate::math::{js_max, js_min};
use serde::{Deserialize, Serialize};

/// `440 · 2^((m − 69)/12)`, the same formula and operation order as `sing.ts`.
pub use crate::math::midi_to_hz;

/// sung consonants are a little shorter than spoken ones
fn cdur(ipa: &str, body: BodyMix, k: f64) -> f64 {
    phoneme(ipa, body).map_or(0.07, |d| d.dur) * k
}

fn consonants(segs: &[Segment], body: BodyMix, k: f64) -> Vec<Segment> {
    segs.iter().map(|c| Segment { ipa: c.ipa.clone(), dur: Some(cdur(&c.ipa, body, k)), ..Default::default() }).collect()
}

/// Sung consonants at their note's pitch: an onset consonant moves the pitch to the new note
/// before its vowel (as singers do), a coda holds the old one.
fn consonants_at(segs: &[Segment], body: BodyMix, k: f64, hz: f64) -> Vec<Segment> {
    let mut out = consonants(segs, body, k);
    for c in out.iter_mut() {
        c.f0 = Some(hz);
    }
    out
}

/// a sung vowel keeps at least this long when a short note squeezes its consonants (s)
const MIN_VOWEL: f64 = 0.05;
/// a squeezed consonant keeps at least this long (s); a flap is 16 ms
const MIN_CONSONANT: f64 = 0.012;

/// Fit the consonants between a vowel and the next one into `span`, the time from the vowel's
/// beat to the next vowel's (or to the end of a phrase's last note): when they leave the vowel
/// less than [`MIN_VOWEL`], they shrink in proportion (each down to [`MIN_CONSONANT`]), so the
/// next vowel still lands on its beat and a fast run doesn't drift later note by note.
fn fit(groups: [&mut [Segment]; 2], span: f64) {
    let [a, b] = groups;
    let cons = total(a) + total(b);
    let room = span - MIN_VOWEL;
    if cons <= room || cons <= 0.0 {
        return;
    }
    let scale = js_max(0.0, room) / cons;
    for c in a.iter_mut().chain(b.iter_mut()) {
        c.dur = Some(js_max(MIN_CONSONANT, c.dur.unwrap_or(0.0) * scale));
    }
}

fn total(segs: &[Segment]) -> f64 {
    segs.iter().fold(0.0, |s, c| s + c.dur.unwrap_or(0.0))
}

fn vibrato_gesture(t0: f64, t1: f64, depth: f64, tag: Option<f64>) -> Gesture {
    let mut g = Gesture::new(t0 + 0.22, t1, 0.3, 0.06, 4.0, Targets::default().with(ControlKey::VIBRATO, depth)).label("vibrato");
    g.tag = tag;
    g
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SungNote {
    pub midi: f64,
    /// length in beats
    pub beats: f64,
    pub syl: Syllable,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SongOptions {
    pub body: BodyMix,
    pub bpm: f64,
    /// plan time of the first note's beat (its onset consonants come before)
    pub t0: f64,
    /// vibrato depth (cents) on held vowels
    pub vibrato: f64,
    /// how diphthongs move between their vowels. None: the first is held and the second takes the
    /// note's last fifth, with the planner's quick step between them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diphthongs: Option<Diphthongs>,
    /// a note's pitch is moved to in at most [`PITCH_MOVE`] seconds each way, and where the move
    /// would be heard in the note (it starts on its vowel, or on a consonant that sounds its
    /// pitch: a nasal, r, l, w, j, a voiced fricative) this long (s) before its first sound, so
    /// that sound starts on the pitch. None: the old note's pitch lets go only as the new note begins, over
    /// up to 60 ms, and the voice follows that late. A voiced onset then swoops into its note
    /// ("r-recipes" up a fifth is 120 cents flat on the beat), and a short slurred note inside a
    /// word (a quaver's neighbour note at a fast tempo) is over before its pitch is heard.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch_lead: Option<f64>,
    /// vowels sung as another colour: a vowel → the vowel it is shaded towards and how far
    /// (0 itself … 1 the other), its articulation the weighted mean of the two. A choir's
    /// rounded, covered vowels are the front vowels most of the way to their rounded
    /// neighbours: `{"i": ["y", 0.9], "ɛ": ["œ", 0.9]}`. The keys are the vowels as planned
    /// (a diphthong's two parts, after `diphthongs.open`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vowels: Option<BTreeMap<String, (String, f64)>>,
    /// vibrato that grows through the note, as a singer's does once the vowel has opened. None:
    /// it is at its full depth from the note's start (the blend lets any gesture's target
    /// through at once), on vowels over 0.3 s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vibrato_grows: Option<VibratoGrowth>,
    /// how long sung consonants are, as a share of their spoken lengths. None: 0.8. At a fast
    /// tempo that leaves a short note mostly consonant (an f is 88 ms, and a swung quaver at 168
    /// to the minute is 119): a singer's are nearer 0.5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub consonant_length: Option<f64>,
}

/// How a held vowel's vibrato comes in: none for `delay` seconds, then growing over `over`
/// seconds to the song's depth, which it keeps to the note's end. A shorter note gets as far as
/// it lasts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VibratoGrowth {
    pub delay: f64,
    pub over: f64,
}

/// the growing vibrato is made of gestures this long (s), each a little deeper than the last
const VIBRATO_STEP: f64 = 0.12;

/// A vowel's growing vibrato from t0 to t1: a run of overlapping gestures, each on the depth the
/// growth has reached at its middle and cross-fading with its neighbours over a whole step, so
/// the depth rises smoothly. (One gesture cannot grow: the voice takes a gesture's target nearly
/// whole as soon as the gesture has any weight.)
fn growing_vibrato(t0: f64, t1: f64, depth: f64, grow: VibratoGrowth, out: &mut Vec<Gesture>) {
    let (from, over) = (t0 + js_max(0.0, grow.delay), js_max(VIBRATO_STEP, grow.over));
    let reached = |t: f64| {
        let x = js_min(1.0, js_max(0.0, (t - from) / over));
        depth * (0.5 - 0.5 * crate::math::cos(crate::math::PI * x))
    };
    let mut a = from;
    while a + VIBRATO_STEP <= t1 {
        // the last step holds what has been reached to the end of the note
        let last = a + 2.0 * VIBRATO_STEP > t1 || a + VIBRATO_STEP >= from + over;
        let b = if last { t1 } else { a + VIBRATO_STEP };
        let at = if last { js_min(t1, from + over) } else { a + 0.5 * VIBRATO_STEP };
        out.push(Gesture::new(a, b, js_min(VIBRATO_STEP, js_max(0.01, a - t0)), if last { 0.06 } else { VIBRATO_STEP }, 4.0, Targets::default().with(ControlKey::VIBRATO, reached(at))).label("vibrato"));
        if last {
            break;
        }
        a = b;
    }
}

/// A sung diphthong that glides: "life" as laaa-iife rather than lah-if.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Diphthongs {
    /// the last part of the vowel's length (0..1) given to its second vowel: four fifths of it
    /// one smooth move from the first vowel, the last fifth the second held (at most
    /// [`MAX_GLIDE`] seconds in all)
    pub glide: f64,
    /// "aɪ" and "aʊ" set out from this vowel instead of the front "a" ("ɑ" is darker)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open: Option<String>,
}

/// a diphthong's glide takes at most this long (s), however long the note
pub const MAX_GLIDE: f64 = 0.8;
/// with [`SongOptions::pitch_lead`], the old note's pitch lets go and the new one's comes in
/// over at most this long each (s), at the same time
pub const PITCH_MOVE: f64 = 0.03;

/// A hum (a syllable whose core is a nasal: "dm", "mm") after consonants opens for this long (s)
/// on the beat before the lips close, at most 0.3 of the note: the consonant's release needs an
/// open mouth to be heard. A d let go behind closed lips has no attack.
pub const HUM_RELEASE: f64 = 0.05;
/// A phrase that ends on a hum keeps the mouth in the hum this long (s) after it: the breath
/// stops (its 50 ms fall, and the folds parting until 100 ms) behind closed lips. Opening with
/// the hum's end let the last of the voice and the breath out through the mouth: "mmm-uh".
pub const HUM_HOLD: f64 = 0.13;

fn melisma(s: &Syllable) -> bool {
    s.melisma == Some(true)
}

/// Plan a whole melody: one utterance per phrase, so each phrase gets its own breath.
pub fn plan_song(notes: &[SungNote], o: &SongOptions) -> Plan {
    let beat = 60.0 / o.bpm;
    let mut gestures: Vec<Gesture> = Vec::new();
    let mut segments: Vec<PlannedSegment> = Vec::new();
    let mut times: Vec<f64> = Vec::with_capacity(notes.len());
    let mut t = o.t0;
    for n in notes {
        times.push(t);
        t += n.beats * beat;
    }
    let mut end = t;
    let mut i = 0;
    while i < notes.len() {
        let mut j = i;
        while j < notes.len() - 1 && !notes[j].syl.phrase_end {
            j += 1;
        }
        let mut segs: Vec<Segment> = Vec::new();
        let mut vowel_spans: Vec<(f64, f64)> = Vec::new();
        // segments to reshape once planned: a gliding diphthong's first vowel and how long the
        // move to its second takes, and the first segment of each note whose pitch leads it
        let mut glides: Vec<(usize, f64)> = Vec::new();
        let mut led: Vec<(usize, bool, f64, f64)> = Vec::new();
        // and the hums, which hold their closure a little past their span
        let mut hums: Vec<usize> = Vec::new();
        // each note's onset and coda consonants, at the note's pitch (a coda at the pitch of the
        // note whose syllable it ends; a melisma has neither)
        let hz_of = |k: usize| midi_to_hz(notes[k].midi);
        let clen = o.consonant_length.unwrap_or(0.8);
        let mut onsets: Vec<Vec<Segment>> = (i..=j).map(|k| if melisma(&notes[k].syl) { Vec::new() } else { consonants_at(&notes[k].syl.onset, o.body, clen, hz_of(k)) }).collect();
        let mut codas: Vec<Vec<Segment>> = (i..=j)
            .map(|k| if k < j && melisma(&notes[k + 1].syl) { Vec::new() } else { consonants_at(&last_of_chain(notes, k).coda, o.body, clen, hz_of(k)) })
            .collect();
        // the phrase's last vowel leaves a short breath before the next phrase. The call's last
        // phrase is followed by a rest (callers split phrases at rests), so the breath comes from
        // the rest: hold that note to its end, less the breath's release.
        let reserve = if j == notes.len() - 1 { 0.05 } else { 0.12 };
        let last_end = times[j] + notes[j].beats * beat - js_min(reserve, 0.3 * notes[j].beats * beat);
        for k in i..=j {
            let (span, next) = if k < j { (times[k + 1] - times[k], &mut onsets[k + 1 - i][..]) } else { (last_end - times[k], &mut [][..]) };
            fit([&mut codas[k - i][..], next], span);
        }
        let first_onset = total(&onsets[0]);
        for k in i..=j {
            let n = &notes[k];
            let hz = hz_of(k);
            let next_melisma = k < j && melisma(&notes[k + 1].syl);
            let coda = std::mem::take(&mut codas[k - i]);
            let onset = std::mem::take(&mut onsets[k - i]);
            // vowel from this note's beat to the next note's onset (less this note's coda)
            let note_end = if k < j { times[k + 1] - total(&onsets[k + 1 - i]) } else { last_end };
            // only consonants squeezed to their floor can leave less than MIN_VOWEL
            let vdur = js_max(0.02, note_end - times[k] - total(&coda));
            let hum = !is_vowel(&n.syl.nucleus.ipa, o.body);
            let released = hum && !onset.is_empty();
            if k > i && o.pitch_lead.is_some() && notes[k].midi != notes[k - 1].midi {
                // is the move heard in the note's first sound? (its vowel, or a consonant that
                // sounds its pitch); and the old note's length
                let heard = onset.first().is_none_or(|c| phoneme(&c.ipa, o.body).is_some_and(|d| d.voicing == Voicing::Voiced && !matches!(d.manner, Manner::Stop | Manner::Affricate)));
                // into a consonant, the lead is for leaps: a step may slide in the consonant,
                // as a singer's does (a tone down into an r takes all of the r), where a leap
                // sliding there is a swoop; a third or more is led in full
                let far = if onset.is_empty() { 1.0 } else { js_min(1.0, (notes[k].midi - notes[k - 1].midi).abs() / 4.0) };
                led.push((segs.len(), heard, far, times[k] - times[k - 1]));
            }
            segs.extend(onset);
            let parts = diphthong_parts(&n.syl.nucleus.ipa);
            // the vowel a diphthong sets out from
            let first = |a: &str| match &o.diphthongs {
                Some(Diphthongs { open: Some(open), .. }) if a == "a" => open.clone(),
                _ => a.to_string(),
            };
            match parts {
                Some((a, b)) if !next_melisma => {
                    // the second vowel's share of the note: its last fifth, or the last fifth of
                    // the glide (the move itself takes the four fifths before it, see below)
                    let tail = match &o.diphthongs {
                        Some(d) => {
                            let glide = js_min(MAX_GLIDE, vdur * js_min(1.0, js_max(0.0, d.glide)));
                            glides.push((segs.len(), 0.8 * glide));
                            0.2 * glide
                        }
                        None => vdur * 0.2,
                    };
                    // (the default's 0.8 and 0.2 are kept as products: the plans are golden)
                    let head = if o.diphthongs.is_some() { vdur - tail } else { vdur * 0.8 };
                    segs.push(Segment { ipa: first(a), dur: Some(head), f0: Some(hz), stress: Some(1), word_end: None });
                    segs.push(Segment { ipa: b.to_string(), dur: Some(tail), f0: Some(hz), stress: Some(1), word_end: None });
                }
                _ => {
                    let ipa = parts.map_or(n.syl.nucleus.ipa.clone(), |p| first(p.0));
                    // a hum's consonants are let go into a moment of open mouth, on the beat
                    let open = if released { js_min(HUM_RELEASE, 0.3 * vdur) } else { 0.0 };
                    if released {
                        segs.push(Segment { ipa: "ə".to_string(), dur: Some(open), f0: Some(hz), stress: Some(0), word_end: None });
                    }
                    if hum {
                        hums.push(segs.len());
                    }
                    segs.push(Segment { ipa, dur: Some(vdur - open), f0: Some(hz), stress: Some(1), word_end: None });
                }
            }
            segs.extend(coda);
            vowel_spans.push((times[k], times[k] + vdur));
        }
        if !segs.is_empty() {
            // the planner lays segments out from t0 + 0.06: start so the first vowel lands on its beat
            let mut plan = plan_utterance(&segs, &PlanOptions { body: o.body, t0: times[i] - first_onset - 0.06, f0: Some(hz_of(i)), intonation: Some(Intonation::Flat), sung: Some(true), ..Default::default() });
            // The planner gives each segment one gesture over its span (labelled with its vowel,
            // or "pitch"), ramped in before the span and out after it; overlapping gestures
            // blend by weight. So:
            let same = |a: f64, b: f64| (a - b).abs() < 1e-9;
            let of = |g: &Gesture, label: &str, s: &PlannedSegment| g.label.as_ref().is_some_and(|l| l.as_str() == label) && same(g.t0, s.t0) && same(g.t1, s.t1);
            // shaded vowels: the vowel's gesture aims between its own articulation and the
            // other vowel's (first, while every gesture is still on its segment's span)
            if let Some(shades) = &o.vowels {
                for s in plan.segments.iter().filter(|s| s.kind == "vowel") {
                    let Some((to, w)) = shades.get(&s.ipa) else { continue };
                    let (Some(own), Some(other)) = (vowel_artic(&s.ipa, o.body), vowel_artic(to, o.body)) else { continue };
                    let w = js_min(1.0, js_max(0.0, *w));
                    let targets = artic_targets(&ArticPatch::blend(&[(1.0 - w, own), (w, other)]));
                    for g in plan.gestures.iter_mut().filter(|g| of(g, &s.ipa, s)) {
                        g.targets = targets;
                    }
                }
            }
            // a hum keeps its closure (the gestures labelled with it that end with its span)
            // while the breath stops at the end of a phrase, and into a consonant after it, so
            // the mouth does not open on the voice between the two
            for &at in &hums {
                let seg = plan.segments[at].clone();
                let hold = match plan.segments.get(at + 1) {
                    None => HUM_HOLD,
                    Some(next) if next.kind != "vowel" => js_min(0.04, 0.6 * (next.t1 - next.t0)),
                    Some(_) => 0.0,
                };
                for g in plan.gestures.iter_mut() {
                    if g.label.as_ref().is_some_and(|l| l.as_str() == seg.ipa) && g.t0 >= seg.t0 - 0.05 && (g.t1 - seg.t1).abs() < 0.05 {
                        g.t1 += hold;
                    }
                }
            }
            // A note's pitch: the old note lets go of its pitch and the new note takes its own
            // over a short ramp each, so the move is quick; and where the move would be heard in
            // the new note, it is made early, in the last of the old one, to be over as the new
            // note's first sound starts. That is a note with no onset consonant (a melisma, a
            // word that starts on its vowel) and one whose first consonant carries its pitch
            // out loud: a nasal, r, l, w, j, a voiced fricative. Left to move in such a
            // consonant the pitch is halfway as it starts and slides through all of it ("r-re"
            // up a fifth starts near the old note). A voiceless consonant or a stop hides the
            // move, so those notes are not led: leading them bends the old note's vowel for
            // nothing, and took a short slurred note's pitch away early.
            if let Some(lead) = o.pitch_lead {
                let pitch = |g: &Gesture| g.label.as_ref().is_some_and(|l| l.as_str() == "pitch");
                for &(at, heard, far, old_note) in &led {
                    let edge = plan.segments[at].t0;
                    // at most three tenths of the old note
                    let cut = edge - if heard { js_min(lead * far, 0.3 * old_note) } else { 0.0 };
                    // the old note's pitch ends at the cut: what lies wholly after it (a short
                    // coda) goes, and the new note's pitch covers it
                    plan.gestures.retain(|g| !(pitch(g) && g.t0 >= cut - 1e-9 && g.t1 <= edge + 1e-9 && g.t0 < edge - 1e-9 && cut < edge - 1e-9));
                    for g in plan.gestures.iter_mut().filter(|g| pitch(g)) {
                        if same(g.t0, edge) && g.t1 > edge {
                            g.t0 = cut;
                            g.ramp_in = js_min(g.ramp_in, PITCH_MOVE);
                        } else if g.t0 < cut && g.t1 <= edge + 1e-9 && g.t1 + g.ramp_out > cut {
                            g.t1 = js_min(g.t1, cut);
                            g.ramp_out = js_min(g.ramp_out, js_max(0.005, cut + PITCH_MOVE - g.t1));
                        }
                    }
                }
            }
            // a gliding diphthong: the first vowel lets go early, by the length of the move, and
            // ramps out over it while the second ramps in over the same stretch. Their weights
            // add to one throughout, so the blend is one smooth move from the first vowel to the
            // second, which then holds for its own span.
            for &(at, mv) in &glides {
                let (a, b) = (plan.segments[at].clone(), plan.segments[at + 1].clone());
                for g in plan.gestures.iter_mut() {
                    if of(g, &a.ipa, &a) {
                        g.t1 -= mv;
                        g.ramp_out = mv;
                    } else if of(g, &b.ipa, &b) {
                        g.ramp_in = mv;
                    }
                }
                // the timeline shows the vowels changing over at the middle of the move
                plan.segments[at].t1 -= 0.5 * mv;
                plan.segments[at + 1].t0 -= 0.5 * mv;
            }
            gestures.extend_from_slice(&plan.gestures);
            segments.extend(plan.segments);
            for (a, b) in vowel_spans {
                match o.vibrato_grows {
                    Some(grow) => growing_vibrato(a, b, o.vibrato, grow, &mut gestures),
                    None if b - a > 0.3 => gestures.push(vibrato_gesture(a, b, o.vibrato, None)),
                    None => {}
                }
            }
            end = js_max(end, plan.end);
        }
        i = j + 1;
    }
    Plan { gestures, segments, end }
}

/// The syllable whose coda ends a melisma chain starting at note k.
fn last_of_chain(notes: &[SungNote], k: usize) -> &Syllable {
    let mut b = k;
    while b > 0 && melisma(&notes[b].syl) {
        b -= 1;
    }
    &notes[b].syl
}

/// Live: gestures relative to "now" (the planner's 60 ms lead is mostly taken back).
const LIVE_T0: f64 = -0.045;
/// a held vowel lasts at most this long (s)
const HOLD: f64 = 30.0;

/// The previous note of a legato pair.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LegatoFrom {
    pub syl: Syllable,
    pub midi: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NoteOnOptions {
    pub body: BodyMix,
    pub vibrato: f64,
    pub tag: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legato_from: Option<LegatoFrom>,
}

/// Key pressed: sing the previous syllable's glide and coda if it is still sounding (legato),
/// then this syllable's onset, then its vowel held open-ended. Every gesture gets `tag`; the
/// pitch and vibrato get `tag + 1`, so a melisma can change pitch without ending the vowel.
pub fn plan_note_on(syl: &Syllable, midi: f64, o: &NoteOnOptions) -> Vec<Gesture> {
    let hz = midi_to_hz(midi);
    let mut segs: Vec<Segment> = Vec::new();
    if let Some(l) = &o.legato_from {
        segs.extend(tail(&l.syl, o.body, midi_to_hz(l.midi)));
    }
    segs.extend(consonants(&syl.onset, o.body, 0.7));
    let parts = diphthong_parts(&syl.nucleus.ipa);
    segs.push(Segment { ipa: parts.map_or(syl.nucleus.ipa.clone(), |p| p.0.to_string()), dur: Some(HOLD), f0: Some(hz), stress: Some(1), word_end: None });
    let plan = plan_utterance(&segs, &PlanOptions { body: o.body, t0: LIVE_T0, f0: Some(hz), intonation: Some(Intonation::Flat), open_end: Some(true), ..Default::default() });
    let vowel = &plan.segments[plan.segments.len() - 1];
    let mut out: Vec<Gesture> = plan.gestures.iter().map(|g| g.tag(if g.targets.has(ControlKey::F0) { o.tag + 1.0 } else { o.tag })).collect();
    out.push(vibrato_gesture(vowel.t0, vowel.t1, o.vibrato, Some(o.tag + 1.0)));
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MelismaOptions {
    pub vibrato: f64,
    pub tag: f64,
}

/// Melisma, live: the held vowel moves to a new pitch (release tag + 1 first).
pub fn plan_melisma(midi: f64, o: &MelismaOptions) -> Vec<Gesture> {
    let MelismaOptions { vibrato, tag } = *o;
    vec![
        Gesture::new(0.0, HOLD, 0.05, 0.06, 6.0, Targets::default().with(ControlKey::F0, midi_to_hz(midi))).label("pitch").tag(tag + 1.0),
        vibrato_gesture(0.0, HOLD, vibrato, Some(tag + 1.0)),
    ]
}

/// Key released: the diphthong's glide and the coda (release the note's tags first).
pub fn plan_note_off(syl: &Syllable, midi: f64, body: BodyMix) -> Vec<Gesture> {
    let hz = midi_to_hz(midi);
    let segs = tail(syl, body, hz);
    if segs.is_empty() {
        return Vec::new();
    }
    plan_utterance(&segs, &PlanOptions { body, t0: LIVE_T0, f0: Some(hz), intonation: Some(Intonation::Flat), ..Default::default() }).gestures
}

/// What follows a held vowel: the diphthong's glide (at the note's pitch) and the coda.
fn tail(syl: &Syllable, body: BodyMix, hz: f64) -> Vec<Segment> {
    let parts = diphthong_parts(&syl.nucleus.ipa);
    let mut out: Vec<Segment> = Vec::new();
    if let Some((_, b)) = parts {
        out.push(Segment { ipa: b.to_string(), dur: Some(0.07), stress: Some(1), f0: Some(hz), word_end: None });
    }
    out.extend(consonants(&syl.coda, body, 0.7));
    out
}
