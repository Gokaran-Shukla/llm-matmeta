# Changelog

## 0.2.0 (2026-11-19)

### Added
- `matmeta file`: LLM-only extraction for files without a parser, with **quote verification**: every value
  must come with the source line it was read from; the line must be in the file and contain the value
  (numbers to 1e-4, Ry/Ha/Bohr conversions recognised). `--strict` removes values that fail.
- Record types `wannier` (Wannier90 `.wout`) and `jobscript` (SLURM/PBS/LSF/SGE), with auto-detection.
- `matmeta calc --extras`: attaches `.wout` and job-script records to the calculation record; exported as
  `hasPart` in JSON-LD.
- Long files are condensed (head + tail + keyword windows) before being sent to the model.
- New result fields `energy_sigma0_eV` and `magnetization_vector_muB`.
- `docs/llm-matmeta-formalism.pdf`: full formalism of the pipeline.
- Example `examples/vasp_Fe_soc` (SOC, moment along z, Wannier90 output, SLURM script); 19 new tests.

### Fixed
- VASP task now follows VASP's defaults (no IBRION + NSW=0 -> scf) and the effective parameters echoed in
  the OUTCAR, instead of requiring IBRION in the INCAR.
- Functional read from LEXCH (POTCAR) when the INCAR has no GGA tag; QE functional read from the output.
- Noncollinear runs: total moment is |m| of (mx, my, mz) instead of mx only.
- Relaxations and MD report the CONTCAR (final) structure.
- An unreachable or failing LLM no longer aborts the run: the parser record is written with `llm_error`.
- Lattice vectors and other arrays are compared with numerical tolerance.
- `spin_polarized` is `false` (not `null`) for known non-spin-polarised runs.

### Changed
- OUTCAR and pw.x output are streamed line by line (constant memory); `OUTCAR.gz` is accepted.
- The model now sees the POSCAR header verbatim and the OUTCAR parameter block, making its reading of the
  settings independent of the parser.
- Schema descriptions fix conventions sent to the model (full-cell formula, standard functional names,
  smearing names), removing spurious conflicts.
