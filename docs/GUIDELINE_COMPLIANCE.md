# Subtitle guideline compliance - what the tool can and cannot check

PlanetRead's Hindi subtitle guidelines set presentation rules for editors:
font, sizes, spacing, character and line limits, colour, stroke, shadow. The
checker only sees the finished, burned-in video - pixels, not the project that
made them. So it verifies the rules a rendered frame can actually show, and is
explicit about the ones it cannot.

## What the tool checks

- **Characters on screen: up to 70** (two lines of 35). Counted from the OCR
  text of each subtitle. This is the pass/fail gate - a reliable signal that also
  caps how much text is on screen, so a caption carrying too much gets caught
  here.
- **Lines on screen** are also measured from the subtitle's shape, but only for
  information. Counting lines from burned pixels is not dependable across every
  render, so a caption is never failed on the line count - the character budget
  above already limits how much text can sit on screen.

The result is shown as a **Guideline compliance** score in the HTML report,
next to the legibility grade: "X of Y lines follow the guidelines".

## Why the other rules cannot be checked

Because a burned-in video only gives us pixels. Once the subtitle is baked into
the frame, the original settings are gone. Rule by rule:

Font Name (Noto Sans Devanagari) - to know the font you would have to recognise
the exact letter shapes and match them to one font out of thousands. Two fonts
can look nearly identical at video size. No reliable way to tell from pixels.

Font Type (Regular) - same problem, even harder. Regular vs medium is a tiny
weight difference, lost after the video is compressed and scaled.

Font Size (60) - "60" is a point size in the editing software. It only means
something on the original canvas. In a finished video we just see pixels, and
the size in pixels depends on the video resolution and how it was scaled. We can
measure how TALL the text is, but we cannot turn that back into "60".

Word Spacing (2) and Line Spacing (20) - these are also point values set at
authoring time. The gaps in the final video depend on resolution and scaling, so
the exact "2" or "20" cannot be recovered.

Text Colour (White) - this one we technically could read, but we skipped it on
purpose. SLS subtitles use a yellow highlight that moves word by word. A strict
"must be white" check would wrongly fail those. Legibility (contrast) already
covers whether the text stands out.

Text Stroke (2 black) and Shadow - the exact stroke width in points, or the
shadow's angle and blur, cannot be measured precisely from a small, compressed
frame. But their EFFECT (making text readable) is what legibility measures.

Short version: the ones we skipped are all authoring-tool settings measured in
points, plus the font itself. The finished video does not carry those numbers.
So instead of guessing them and being wrong, the tool checks what the pixels
honestly show (characters and rough line/text amount) and measures the effect of
the rest through legibility.
