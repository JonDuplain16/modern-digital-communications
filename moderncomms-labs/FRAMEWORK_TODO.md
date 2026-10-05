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
- (done, opt-in) commlib cpm: the GMSK waveform's absolute phase depended on its first bit (the
  Gaussian precursor of bit 0 was dropped), so a LaurentReceiver calibration only fitted data
  starting with the same bit (~19 deg mismatch at BT 0.3). `gmsk_baseband(..., laurent_ref=True)`
  now keeps the precursor (phase 0 before any pulse = Laurent's reference) and the calibration
  fits every waveform (tested). Default unchanged (the book's ch09 figures use their own copy of
  the generator, and lab 21's phase plots start at 0); lab 21 still uses its c[0] = 0 workaround
  and could switch to laurent_ref=True.
- (done, opt-in) commlib.ofdm.dft_s_ofdm_modulate/demodulate(..., contiguous=True): the true
  localized LTE/NR mapping (DC included); used by lab 27's new PAPR experiment, tested. The
  default stays the legacy DC-hole mapping because book/figscripts/ch17_figs.py (ch17_papr, panel
  b) calls it: the legacy mapping overstates DFT-s-OFDM PAPR at 1e-3 by about 0.3 dB (QPSK 8.0 vs
  7.7 dB, 16-QAM 9.1 vs 8.9 dB, 300 subcarriers, 4x). OWNER: when figures are next regenerated,
  pass contiguous=True there (ch17 text quotes no DFT-s number; check the caption).
- BOOK: final pass on all labboxes/tryit to match new lab experiment names (ch05 labbox "companion notebook... 75 minutes"; ch04 "Section 6" of lab15 -> experiment 7 "Zero-IF vs low-IF")
- BOOK: ch08 MLT-3 "DC null" claim (lab measures only ~4.5 dB below peak) -> soften
- selftest timings use warm imports (real first launch is slower); text(y=None) with the default
  anchor (0, 1) sits partly above the view top: pass anchor=(…, 0) for labels pinned to the top
- retire notebook tooling: delete labs/lab00_index.py, labs/lab00_index.ipynb,
  tests/build_notebooks.py, tests/build_all.sh, tests/dump_figs.py (deletion needs the owner's OK)
- (done) lab28 OFDM radar: cut legends replaced by titles naming the cut cell ("Range cut at
  v = +14.1 m/s", "Doppler cut at R = 39.0 m") and T1/T2 labels on the true-target lines;
  Doppler cut range matches the map. Book screenshot ch17_lab28_radar re-captured.
