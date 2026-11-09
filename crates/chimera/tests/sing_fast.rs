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
    // an onset consonant is not led (the pitch moves in it): the l of "life", back up to F#4,
    // takes its pitch as it starts, as before; but both sides of each move are quick
    let l = |p: &Plan| p.segments.iter().find(|s| s.ipa == "l").cloned().unwrap();
    let early = 0.0;
    let starts = |p: &Plan, t: f64| p.gestures.iter().any(|g| g.label.as_ref().is_some_and(|x| x.as_str() == "pitch") && (g.t0 - t).abs() < 1e-9);
    assert!(starts(&plain, l(&plain).t0) && starts(&styled, l(&styled).t0));
    for g in styled.gestures.iter().filter(|g| g.label.as_ref().is_some_and(|x| x.as_str() == "pitch")) {
        if (g.t0 - 1.44).abs() < 1e-9 || (g.t0 - (l(&styled).t0 - early)).abs() < 1e-9 {
            assert!(g.ramp_in <= 0.03 + 1e-9, "comes in over {}", g.ramp_in);
        }
        if (g.t1 - 1.44).abs() < 1e-9 || (g.t1 - (l(&styled).t0 - early)).abs() < 1e-9 {
            assert!(g.ramp_out <= 0.03 + 1e-9, "lets go over {}", g.ramp_out);
        }
    }
}
