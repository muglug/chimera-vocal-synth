//! Fast sung runs (scat): every vowel lands on its beat however short the note, and the pitch
//! moves on the onset consonant, so it has arrived when the vowel starts.
use chimera::control::score::ControlKey;
use chimera::phonetics::english::EnglishG2P;
use chimera::phonetics::sing::{SongOptions, SungNote, midi_to_hz, plan_song};
use chimera::phonetics::syllables::lyrics_to_syllables;
use chimera::phonetics::{Plan, PlannedSegment};

fn g2p() -> EnglishG2P {
    let mut g = EnglishG2P::new();
    g.load_dict("bah\tB AA1\nbee\tB IY1\nbop\tB AA1 P\ndee\tD IY1\ndoo\tD UW1\nscoo\tS K UW1\nskee\tS K IY1\nstrut\tS T R AH1 T\n");
    g
}

/// The melody alternates C4 and E4, `beats` long each.
fn sing(lyrics: &str, beats: f64, bpm: f64) -> (Vec<SungNote>, Vec<f64>, Plan) {
    let syl = lyrics_to_syllables(lyrics, &g2p());
    let notes: Vec<SungNote> = syl.iter().enumerate().map(|(i, s)| SungNote { midi: if i % 2 == 0 { 60.0 } else { 64.0 }, beats, syl: s.clone() }).collect();
    let times = (0..notes.len()).map(|i| 0.5 + i as f64 * beats * 60.0 / bpm).collect();
    let plan = plan_song(&notes, &SongOptions { body: "male".into(), bpm, t0: 0.5, vibrato: 0.0, ..Default::default() });
    (notes, times, plan)
}

/// The first vowel segment of each syllable (a diphthong's second part follows its first).
fn vowel_onsets(segs: &[PlannedSegment]) -> Vec<f64> {
    segs.iter().enumerate().filter(|(i, s)| s.kind == "vowel" && (*i == 0 || segs[i - 1].kind != "vowel")).map(|(_, s)| s.t0).collect()
}

#[test]
fn fast_runs_land_on_the_beat() {
    for lyrics in ["scoo bah dee bop scoo bah dee bop", "skee dee bop bah skee dee bop bah", "bah bee doo bop bah bee doo bop", "strut strut strut strut"] {
        for (bpm, beats) in [(184.0, 0.25), (184.0, 1.0 / 3.0), (240.0, 0.25), (100.0, 0.5)] {
            if lyrics.starts_with("strut") && bpm == 240.0 {
                continue; // four consonants in a 62 ms note: see too_many_consonants_degrade_gently
            }
            let (_, times, plan) = sing(lyrics, beats, bpm);
            let got = vowel_onsets(&plan.segments);
            assert_eq!(got.len(), times.len(), "{lyrics:?} at {bpm} bpm");
            for (k, (g, want)) in got.iter().zip(&times).enumerate() {
                assert!((g - want).abs() < 1e-6, "{lyrics:?}, {beats} beats at {bpm} bpm: vowel {k} at {g:.4} s, its beat is {want:.4} s");
            }
            // and every vowel still sounds for a while
            for s in plan.segments.iter().filter(|s| s.kind == "vowel") {
                assert!(s.t1 - s.t0 >= 0.02 - 1e-9, "{lyrics:?}: a {:.3} s vowel", s.t1 - s.t0);
            }
        }
    }
}

#[test]
fn too_many_consonants_degrade_gently() {
    // "t | s t r" between vowels can't shrink below its floor in a 62 ms note: the vowels keep
    // their 20 ms and the run slips a few milliseconds a note instead of a whole note
    let (_, times, plan) = sing("strut strut strut strut", 0.25, 240.0);
    for (k, (g, want)) in vowel_onsets(&plan.segments).iter().zip(&times).enumerate() {
        assert!(g - want < 0.008 * k as f64 + 1e-6, "vowel {k} {:.1} ms late", (g - want) * 1000.0);
    }
}

#[test]
fn notes_with_room_keep_their_consonants() {
    // at a walking pace nothing is squeezed: the onset is the consonant's sung length
    let (_, _, plan) = sing("bah dee bah dee", 1.0, 100.0);
    let d = plan.segments.iter().find(|s| s.ipa == "d").unwrap();
    assert!(d.t1 - d.t0 > 0.04, "d squeezed to {:.3} s", d.t1 - d.t0);
}

#[test]
fn onset_consonants_carry_the_new_pitch() {
    let (notes, _, plan) = sing("bah dee bah dee", 1.0 / 3.0, 184.0);
    let d = plan.segments.iter().find(|s| s.ipa == "d").unwrap();
    let hz = midi_to_hz(notes[1].midi);
    let on_d = plan.gestures.iter().any(|g| (g.t0 - d.t0).abs() < 1e-9 && g.targets.get(ControlKey::F0) == Some(hz));
    assert!(on_d, "no pitch gesture for {hz:.1} Hz starting on the 'd' at {:.4} s", d.t0);
}

#[test]
fn a_delayed_cluster_closure_stays_inside_a_short_sung_stop() {
    // In "harped" the /t/ closes behind the /p/ with its tip already raised by the /ɹ/, so its
    // closure is aimed later (the pop on "sharp tree"). In a quarter-beat note at 180 bpm the /t/
    // lasts only a few milliseconds: a fixed 25 ms delay pushed its closure 23 ms into the next
    // vowel and reversed its secondary gesture.
    let mut g = EnglishG2P::new();
    g.load_dict("harped\tHH AA1 R P T\nah\tAA1\n");
    let syl = lyrics_to_syllables("harped ah harped ah", &g);
    let notes: Vec<SungNote> = syl.iter().map(|s| SungNote { midi: 60.0, beats: 0.25, syl: s.clone() }).collect();
    let plan = plan_song(&notes, &SongOptions { body: "male".into(), bpm: 180.0, t0: 0.5, vibrato: 0.0, ..Default::default() });
    let ts: Vec<_> = plan.segments.iter().enumerate().filter(|(i, s)| s.ipa == "t" && *i > 0 && plan.segments[i - 1].ipa == "p").map(|(_, s)| s.clone()).collect();
    assert!(!ts.is_empty());
    for t in ts {
        for gst in plan.gestures.iter().filter(|g| g.label.as_ref().is_some_and(|l| l.as_str() == "t") && g.t0 >= t.t0 - 0.05 && g.t0 <= t.t1) {
            assert!(gst.t0 <= gst.t1, "a /t/ gesture runs backwards: {:.4} → {:.4}", gst.t0, gst.t1);
            assert!(gst.t1 <= t.t1 + 1e-9, "a /t/ gesture ends at {:.4} s, after its /t/ ({:.4}–{:.4} s)", gst.t1, t.t0, t.t1);
        }
    }
}

/// "dm" has no vowel: the d is let go on the beat through a moment of open mouth into an m that
/// is held for the rest of the note, on its pitch, where a vowel would be; and at the end of
/// the phrase the lips stay closed while the breath stops.
#[test]
fn a_hummed_syllable_holds_its_nasal_for_the_note() {
    use chimera::phonetics::sing::{HUM_HOLD, HUM_RELEASE};
    use chimera::phonetics::syllables::syllabify_segments;
    use chimera::phonetics::{Segment, Syllable};
    let parts = syllabify_segments(&[Segment::new("d"), Segment::new("m")]).remove(0);
    let syl = Syllable { text: "dm".into(), onset: parts.onset, nucleus: parts.nucleus, coda: parts.coda, melisma: None, word_end: true, phrase_end: false };
    // two one-second notes
    let notes: Vec<SungNote> = [45.0, 52.0].iter().map(|&midi| SungNote { midi, beats: 1.0, syl: syl.clone() }).collect();
    let plan = plan_song(&notes, &SongOptions { body: "male".into(), bpm: 60.0, t0: 1.0, vibrato: 0.0, ..Default::default() });
    let ipa: String = plan.segments.iter().map(|s| s.ipa.as_str()).collect();
    assert_eq!(ipa, "dəmdəm");
    let of = |ipa: &str| plan.segments.iter().filter(|s| s.ipa == ipa).collect::<Vec<&PlannedSegment>>();
    let (open, m) = (of("ə"), of("m"));
    // the release is on the beat, the m follows it and lasts until the next note's d (or the
    // phrase's breath)
    for (k, beat) in [1.0, 2.0].iter().enumerate() {
        assert!((open[k].t0 - beat).abs() < 1e-9, "release at {}", open[k].t0);
        assert!((m[k].t0 - (beat + HUM_RELEASE)).abs() < 1e-9, "m at {}", m[k].t0);
        assert!(m[k].t1 - m[k].t0 > 0.85, "m lasts {}", m[k].t1 - m[k].t0);
    }
    // the lips' closure: the first m's runs into the d after it, the last one's past the end
    // of the breath
    let closed_until = |seg: &PlannedSegment| plan.gestures.iter().filter(|g| g.label.as_ref().is_some_and(|l| l.as_str() == "m") && g.t0 >= seg.t0 - 0.05 && g.t0 < seg.t1).map(|g| g.t1).fold(f64::MIN, f64::max);
    assert!(closed_until(m[0]) > m[0].t1 + 0.005 && closed_until(m[0]) < m[0].t1 + 0.05, "first m closed until {} (ends {})", closed_until(m[0]), m[0].t1);
    assert!(closed_until(m[1]) >= m[1].t1 + HUM_HOLD - 0.03, "last m closed until {} (ends {})", closed_until(m[1]), m[1].t1);
    let breath = plan.gestures.iter().find(|g| g.label.as_ref().is_some_and(|l| l.as_str() == "breath")).unwrap();
    assert!(closed_until(m[1]) > breath.t1 + breath.ramp_out, "the lips open at {} with the breath still falling until {}", closed_until(m[1]), breath.t1 + breath.ramp_out);
}

/// The options that change how a song is sung leave it alone when unset, and when set: a
/// diphthong sets out from a darker vowel and moves to its second in one smooth glide over the
/// last part of the note; a note's pitch is moved to quickly, before the note's first sound.
#[test]
fn diphthongs_glide_and_melismas_lead_only_when_asked() {
    use chimera::phonetics::sing::Diphthongs;
    use chimera::phonetics::{Segment, Syllable};
    let seg = |x: &str| Segment::new(x);
    let life = Syllable { text: "life".into(), onset: vec![seg("l")], nucleus: seg("aɪ"), coda: vec![seg("f")], melisma: None, word_end: true, phrase_end: false };
    let on = Syllable { text: "~".into(), onset: vec![], nucleus: seg("ɑ"), coda: vec![], melisma: Some(true), word_end: true, phrase_end: false };
    let wa = Syllable { text: "wa".into(), onset: vec![seg("w")], nucleus: seg("ɑ"), coda: vec![], melisma: None, word_end: true, phrase_end: false };
    // "wa" on F#4 for half a second, slurred down to E#4, then "life" for two seconds
    let notes = vec![SungNote { midi: 66.0, beats: 0.5, syl: wa }, SungNote { midi: 65.0, beats: 0.5, syl: on }, SungNote { midi: 66.0, beats: 2.0, syl: life }];
    let base = SongOptions { body: "male".into(), bpm: 60.0, t0: 1.0, vibrato: 0.0, ..Default::default() };
    let plain = plan_song(&notes, &base);
    let styled = plan_song(&notes, &SongOptions { diphthongs: Some(Diphthongs { glide: 0.5, open: Some("ɑ".into()) }), pitch_lead: Some(0.06), ..base.clone() });
    let ipa = |p: &Plan| p.segments.iter().map(|s| s.ipa.as_str()).collect::<String>();
    assert_eq!((ipa(&plain).as_str(), ipa(&styled).as_str()), ("wɑɑlaɪf", "wɑɑlɑɪf"));
    let vowel = |p: &Plan, v: &str, after: f64| p.gestures.iter().find(|g| g.label.as_ref().is_some_and(|l| l.as_str() == v) && g.t0 >= after).cloned().unwrap();

    // plain: "a" for four fifths of the vowel, then a quick step to "ɪ"
    let (a, i) = (vowel(&plain, "a", 1.9), vowel(&plain, "ɪ", 1.9));
    assert!((a.t1 - i.t0).abs() < 1e-9 && a.ramp_out < 0.06 && i.ramp_in < 0.06, "plain: a to {} (+{}), ɪ from {} (-{})", a.t1, a.ramp_out, i.t0, i.ramp_in);
    // styled: the first vowel lets go as the second starts to come in, and both take the same
    // long stretch over it; the second then holds to the end of the vowel
    let (a, i) = (vowel(&styled, "ɑ", 1.9), vowel(&styled, "ɪ", 1.9));
    let mv = i.ramp_in;
    assert!(mv > 0.5 && mv < 0.8 && (a.ramp_out - mv).abs() < 1e-9, "the move takes {mv}");
    assert!((a.t1 - (i.t0 - mv)).abs() < 1e-9, "a lets go at {}, ɪ comes in from {}", a.t1, i.t0 - mv);
    assert!((i.t1 - i.t0 - 0.25 * mv).abs() < 1e-6, "ɪ holds {}", i.t1 - i.t0);

    // the slur's pitch: plain changes over at the beat (1.5 s), styled 60 ms before it
    let pitches = |p: &Plan| p.gestures.iter().filter(|g| g.label.as_ref().is_some_and(|l| l.as_str() == "pitch") && g.t0 < 1.9 && g.t1 > 1.2).map(|g| (g.t0, g.t1)).collect::<Vec<_>>();
    let change = |p: &Plan| pitches(p).iter().map(|x| x.0).filter(|t| *t > 1.2).fold(f64::MAX, f64::min);
    assert!((change(&plain) - 1.5).abs() < 1e-9 && (change(&styled) - 1.44).abs() < 1e-9, "plain {} styled {}", change(&plain), change(&styled));
    assert!(pitches(&styled).iter().any(|x| (x.1 - 1.44).abs() < 1e-9), "the old note's pitch should end at 1.44: {:?}", pitches(&styled));
    // a consonant that sounds its pitch is led too, by the size of the move: the l of "life",
    // a semitone back up to F#4, has its pitch a quarter of the 60 ms before it starts (a third
    // or more would have all of it); and both sides of each move are quick
    let l = |p: &Plan| p.segments.iter().find(|s| s.ipa == "l").cloned().unwrap();
    let early = 0.015;
    let starts = |p: &Plan, t: f64| p.gestures.iter().any(|g| g.label.as_ref().is_some_and(|x| x.as_str() == "pitch") && (g.t0 - t).abs() < 1e-9);
    assert!(starts(&plain, l(&plain).t0) && starts(&styled, l(&styled).t0 - early) && !starts(&styled, l(&styled).t0));
    // a stop or a voiceless consonant hides the move and is not led: "tie" in place of "life"
    let tie = Syllable { text: "tie".into(), onset: vec![seg("t")], nucleus: seg("aɪ"), coda: vec![], melisma: None, word_end: true, phrase_end: false };
    let mut hidden = notes.clone();
    hidden[2].syl = tie;
    let p = plan_song(&hidden, &SongOptions { pitch_lead: Some(0.06), ..base.clone() });
    let t = p.segments.iter().find(|s| s.ipa == "t").unwrap().t0;
    assert!(starts(&p, t), "the t's pitch should start with it");
    for g in styled.gestures.iter().filter(|g| g.label.as_ref().is_some_and(|x| x.as_str() == "pitch")) {
        if (g.t0 - 1.44).abs() < 1e-9 || (g.t0 - (l(&styled).t0 - early)).abs() < 1e-9 {
            assert!(g.ramp_in <= 0.03 + 1e-9, "comes in over {}", g.ramp_in);
        }
        if (g.t1 - 1.44).abs() < 1e-9 || (g.t1 - (l(&styled).t0 - early)).abs() < 1e-9 {
            assert!(g.ramp_out <= 0.03 + 1e-9, "lets go over {}", g.ramp_out);
        }
    }
}

/// Shaded vowels: the vowel's own gesture aims between its articulation and the other vowel's,
/// all the way at 1, and nothing else in the plan changes.
#[test]
fn a_shaded_vowel_is_sung_towards_another() {
    use chimera::anatomy::human::ArticKey;
    use chimera::phonetics::planner::vowel_artic;
    use std::collections::BTreeMap;
    let base = SongOptions { body: "male".into(), bpm: 60.0, t0: 1.0, vibrato: 0.0, ..Default::default() };
    let (notes, _, plain) = sing("dee bah", 1.0, 60.0);
    let shade = |w: f64| plan_song(&notes, &SongOptions { t0: 0.5, vowels: Some(BTreeMap::from([("i".to_string(), ("y".to_string(), w))])), ..base.clone() });
    let lips = |p: &Plan, v: &str| p.gestures.iter().find(|g| g.label.as_ref().is_some_and(|l| l.as_str() == v)).unwrap().targets.get(ArticKey::LipProt).unwrap();
    let (i, y) = (vowel_artic("i", "male".into()).unwrap(), vowel_artic("y", "male".into()).unwrap());
    let (i, y) = (i.get(ArticKey::LipProt).unwrap(), y.get(ArticKey::LipProt).unwrap());
    assert!(y > i + 0.1, "y should be rounder than i: {y} against {i}");
    assert!((lips(&plain, "i") - i).abs() < 1e-12);
    assert!((lips(&shade(1.0), "i") - y).abs() < 1e-12);
    assert!((lips(&shade(0.5), "i") - 0.5 * (i + y)).abs() < 1e-12);
    // the other vowel, the consonants and every time are as they were
    let (a, b) = (shade(1.0), plain);
    assert_eq!(a.segments, b.segments);
    assert_eq!(a.gestures.len(), b.gestures.len());
    for (g, h) in a.gestures.iter().zip(&b.gestures) {
        assert_eq!((g.t0, g.t1, g.ramp_in, g.ramp_out), (h.t0, h.t1, h.ramp_in, h.ramp_out));
        if g.label.as_ref().is_some_and(|l| l.as_str() != "i") {
            assert_eq!(g.targets.0.map(f64::to_bits), h.targets.0.map(f64::to_bits));
        }
    }
}

/// Vibrato that grows: nothing for the delay, then steps of rising depth up to the song's, held
/// to the end of the note; a short note only gets as far as it lasts.
#[test]
fn vibrato_grows_through_a_held_vowel_when_asked() {
    use chimera::phonetics::sing::VibratoGrowth;
    let depth = |p: &Plan| p.gestures.iter().filter(|g| g.label.as_ref().is_some_and(|l| l.as_str() == "vibrato")).map(|g| (g.t0, g.t1, g.targets.get(ControlKey::VIBRATO).unwrap())).collect::<Vec<_>>();
    // one "bah" for two seconds from 0.5 s (its vowel starts on the beat), then one for 0.4 s
    let (notes, _, plain) = sing("bah", 2.0, 60.0);
    assert_eq!(depth(&plain).len(), 1, "plain: one vibrato gesture at the full depth");
    let grows = SongOptions { body: "male".into(), bpm: 60.0, t0: 0.5, vibrato: 40.0, vibrato_grows: Some(VibratoGrowth { delay: 0.1, over: 0.8 }), ..Default::default() };
    let steps = depth(&plan_song(&notes, &grows));
    assert!(steps.len() >= 6, "{steps:?}");
    // from the delay on, without a gap, each deeper than the last, ending on the full depth
    assert!((steps[0].0 - 0.6).abs() < 1e-9, "starts at {}", steps[0].0);
    assert!(steps.windows(2).all(|w| (w[1].0 - w[0].1).abs() < 1e-9 && w[1].2 > w[0].2), "{steps:?}");
    assert!(steps[0].2 < 4.0 && (steps.last().unwrap().2 - 40.0).abs() < 1e-9, "{steps:?}");
    assert!(steps.last().unwrap().1 > 2.3, "held to the end of the vowel: {steps:?}");
    // 0.4 s: it gets a third of the way
    let short = depth(&plan_song(&[SungNote { beats: 0.4, ..notes[0].clone() }], &grows));
    let top = short.last().unwrap().2;
    assert!(!short.is_empty() && top > 4.0 && top < 20.0, "{short:?}");
}

/// Sung consonants are 0.8 of their spoken lengths unless the song says otherwise.
#[test]
fn consonant_length_scales_the_sung_consonants() {
    let (notes, _, plain) = sing("strut strut", 2.0, 60.0);
    let half = plan_song(&notes, &SongOptions { body: "male".into(), bpm: 60.0, t0: 0.5, vibrato: 0.0, consonant_length: Some(0.4), ..Default::default() });
    let cons = |p: &Plan| p.segments.iter().filter(|s| s.kind != "vowel").map(|s| s.t1 - s.t0).collect::<Vec<_>>();
    let (a, b) = (cons(&plain), cons(&half));
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert!((y / x - 0.5).abs() < 1e-9, "{x} became {y}");
    }
    // and the vowels still land on their beats
    for (x, y) in vowel_onsets(&half.segments).iter().zip(vowel_onsets(&plain.segments)) {
        assert!((x - y).abs() < 1e-9, "{x} against {y}");
    }
    // coda_length shortens only the consonants that close a syllable: the t of "strut", not s, t, r
    let clipped = plan_song(&notes, &SongOptions { body: "male".into(), bpm: 60.0, t0: 0.5, vibrato: 0.0, coda_length: Some(0.5), ..Default::default() });
    let ratios: Vec<f64> = cons(&clipped).iter().zip(&a).map(|(y, x)| y / x).collect();
    assert_eq!(ratios.iter().map(|r| (r * 100.0).round() as i32).collect::<Vec<_>>(), [100, 100, 100, 50, 100, 100, 100, 50]);
}

/// A consonant's posture: a gesture over each such consonant holding the named articulators,
/// and nothing for the others.
#[test]
fn a_consonant_can_hold_a_posture_of_its_own() {
    use chimera::anatomy::human::ArticKey;
    use std::collections::BTreeMap;
    let (notes, _, plain) = sing("strut bah", 1.0, 60.0);
    let shape = BTreeMap::from([("s".to_string(), BTreeMap::from([("lipProt".to_string(), 0.0), ("lipWidth".to_string(), 1.0)]))]);
    let p = plan_song(&notes, &SongOptions { body: "male".into(), bpm: 60.0, t0: 0.5, vibrato: 0.0, consonant_posture: Some(shape), ..Default::default() });
    let held: Vec<_> = p.gestures.iter().filter(|g| g.label.as_ref().is_some_and(|l| l.as_str() == "posture")).collect();
    let s = p.segments.iter().find(|s| s.ipa == "s").unwrap();
    assert_eq!(held.len(), 1);
    assert_eq!((held[0].t0, held[0].t1), (s.t0, s.t1));
    assert_eq!((held[0].targets.get(ArticKey::LipProt), held[0].targets.get(ArticKey::LipWidth), held[0].targets.get(ArticKey::Jaw)), (Some(0.0), Some(1.0), None));
    assert_eq!(p.gestures.len(), plain.gestures.len() + 1);
}

/// A vowel into a voiceless fricative: unless asked, the folds have parted before the consonant
/// starts; asked, they part across its start, gently.
#[test]
fn devoicing_can_straddle_the_start_of_a_fricative() {
    let (notes, _, plain) = sing("bah see", 1.0, 60.0);
    let folds = |p: &Plan| p.gestures.iter().find(|g| g.label.as_ref().is_some_and(|l| l.as_str() == "voiceless")).cloned().unwrap();
    let s = plain.segments.iter().find(|s| s.ipa == "s").unwrap().clone();
    let g = folds(&plain);
    assert!(g.t0 < s.t0 && (g.strength - 6.0).abs() < 1e-9, "plain: apart from {} (the s starts at {})", g.t0, s.t0);
    let over = plan_song(&notes, &SongOptions { body: "male".into(), bpm: 60.0, t0: 0.5, vibrato: 0.0, devoice_over: Some(0.08), ..Default::default() });
    let g = folds(&over);
    // the ramp runs from 40 ms before the s to 40 ms into it
    assert!((g.t0 - (s.t0 + 0.04)).abs() < 1e-9 && (g.ramp_in - 0.08).abs() < 1e-9 && (g.strength - 2.0).abs() < 1e-9, "from {} over {} at {}", g.t0, g.ramp_in, g.strength);
}
