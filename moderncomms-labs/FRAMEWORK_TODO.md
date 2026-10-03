# studio framework follow-ups (collect from lab authors)

Done in studio 1.1 (see studio/README.md): cmap lists; fixed-width controls column with
wrapping/eliding labels and a scroll area sized to the current page; `Text` control (labs 13,
19, 24, 34 migrated); `on_click`; `Plot(axes=False)` / `Canvas`; `legend_swatch` and coloured
bar legends; bars on log axes; NaN readouts ("—") and all-NaN traces allowed by the self-test;
`Readout(floor=)`; polygon `fill_between`; `text(y=None)` re-pins on range changes; joint
x/y ranges and `set_range` (aspect plots); `Button(starts_play=True)`; `on_reset`; challenges
re-checked on pause and reset between self-test settings; ASCII `_slug`; reserved-name errors;
`line(downsample=False)`; non-string `Choice` options (+ `labels=`); `set_control(refresh=False)`;
`theory`/`sim` on every plot; multi-line x tick labels; no SI prefixes; plain LogSlider values;
legend KeyError on key reuse; draggable `handles` + `on_drag` (lab 19 water-filling); book
titles strip `$…$`; book link opens the PDF at the page; catalog descriptions for all 36 labs.

## Still open
- commlib cpm.LaurentReceiver.calibrate ref phase depends on first precoded bit
- commlib.ofdm.dft_s_ofdm_modulate maps non-contiguously around DC -> inflated PAPR (check vs book ch17 figure)
- BOOK: final pass on all labboxes/tryit to match new lab experiment names (ch05 labbox "companion notebook... 75 minutes"; ch04 "Section 6" of lab15 -> experiment 7 "Zero-IF vs low-IF")
- BOOK: ch08 MLT-3 "DC null" claim (lab measures only ~4.5 dB below peak) -> soften
- selftest timings use warm imports (real first launch is slower); text(y=None) with the default
  anchor (0, 1) sits partly above the view top: pass anchor=(…, 0) for labels pinned to the top
- retire notebook tooling: delete labs/lab00_index.py, labs/lab00_index.ipynb,
  tests/build_notebooks.py, tests/build_all.sh, tests/dump_figs.py (deletion needs the owner's OK)
- lab28 OFDM radar range-profile legend truncated 'at +14 m…' may mislabel ~40 m target
