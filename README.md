# GestureScroll

A touchless interaction prototype for sterile-field review
of ultrasound image sequences.

Webcam only. Two gestures. Open source.

[![GestureScroll Demo](https://img.youtube.com/vi/VCTbd_2xli0/maxresdefault.jpg)](https://youtu.be/VCTbd_2xli0)

---

## Why this exists

Touchless image navigation in sterile and near-sterile environments is a
documented clinical need with a decade of research behind it. Most
implementations solve for the full OR — depth sensors, proprietary hardware,
complex gesture vocabularies mapped to PACS systems. That's the right solution
for a well-equipped surgical suite.

It's the wrong solution for a simulation center, a training lab, or a
point-of-care environment where the need is simpler: review a captured
ultrasound sequence without touching anything.

This prototype explores the opposite end of the design space. One continuous
gesture scrubs frames. One deliberate hold gesture switches between sequences.
A standard webcam is the only hardware required. The gesture vocabulary is
intentionally minimal — two interactions, chosen to reduce cognitive load and
false triggers in a procedural context.

---

## Gestures

| Gesture | Action |
|---|---|
| Open hand · swipe left/right | Scrub through frames |
| Hold fist · ~0.6s | Switch between sequences A and B |

Keyboard fallbacks: `A` / `B` switch sequences, `,` / `.` step frames,
`[` / `]` calibrate swipe range, `Q` quit.

---

## Stack

Python · OpenCV · MediaPipe · single file · no config

Hand landmarker model downloads automatically on first run (~1MB).

---

## Setup

```bash
pip install opencv-python mediapipe numpy
python gesturescroll.py
```

Place image sequences in `set_a/` and `set_b/` as numbered PNGs
(`000.png`, `001.png` ...). Press `[` with hand at the left edge of your
intended sweep range and `]` at the right edge to calibrate.

---

## Demo data

Sample sequences are real B-mode lumbar spine ultrasound from a robotic
phantom study. 200 frames per sequence, Clarius HD3 probe, 10 MHz, 8 cm depth.

**Dataset:** ISMR 2024 Lumbar Spine Phantom  
**License:** CC BY 4.0  
**Source:** https://zenodo.org/records/11455227

---

## Findings

Informal observations from building and using the prototype:

- Continuous lateral scrubbing produced fewer accidental activations
  than discrete swipe or pose recognition approaches considered
  during design. Mapping scrub position to wrist x-coordinate
  felt natural and required no gesture training.
- Hold-based sequence switching was more reliable than a discrete
  fist pose trigger. A sustained hold of ~0.6s eliminated false
  switches from momentary finger curl during scrubbing.
- A two-gesture vocabulary was sufficient for the core review task.
  Adding more gestures was considered and rejected — each additional
  gesture increases both cognitive load and false trigger surface.
- Webcam-only tracking at typical workstation distances (0.5–1m)
  was sufficient for reliable hand detection without depth sensing
  or specialized hardware.

---

## Future directions

Areas where this prototype could extend meaningfully:

- DICOM support — load sequences directly from clinical imaging
  files rather than pre-exported PNGs
- PACS integration — gesture navigation layer over existing
  hospital imaging systems
- Multi-sequence review — more than two sequences, gesture-indexed
- Temporal AI-assisted indexing — automatic landmark detection
  to jump to clinically relevant frames
- Expanded gesture vocabulary — zoom, annotate, bookmark,
  evaluated against false-trigger cost

---

## Design notes

Prior touchless interaction research identifies two consistent failure modes:
false triggers from incidental hand movement, and gesture fatigue from large
or unnatural motion. GestureScroll addresses both — the scrub gesture maps to
natural lateral hand movement, and the switch gesture requires a sustained
deliberate hold rather than a discrete pose. The result is a vocabulary small
enough to be used reliably without training.

This is an interaction design prototype, not a medical device.

---

## Related work

Gesture-based sterile field navigation has been studied since the Kinect era
(Wachs et al., 2012; Jacob et al., 2013) and remains an active research area.
GestureScroll is distinguished by its minimal hardware requirement (standard
webcam, no depth sensor) and narrow scope — sequence review in training
contexts rather than full intraoperative PACS navigation.
