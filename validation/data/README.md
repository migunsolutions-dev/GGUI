UFC 3-340-02 reference tables for the Validation engines.

Runtime code loads these JSON files only. Excel is not opened at runtime.

Sources
-------
- UFC Calc.xlsx (user-supplied). Sheets DataSpherical and DataHemiSpherical.
- DPlot GRF files 02_007.grf, 02_009.grf, 02_010.grf, 02_013.grf, 02_015 (1).grf.

Figures
-------
2-7  Spherical free-air positive-phase parameters. Workbook DataSpherical in SI,
     verified against 02_007.grf after unit conversion (endpoint Ps0 relative
     residual ~1e-9). Independent variable z = R/W^(1/3) in m/kg^(1/3).

2-9  Reflected pressure vs angle of incidence at a reflecting surface. From
     02_009.grf. Family: scaled height of charge Hc/W^(1/3) (ft/lb^(1/3)).
     Independent variable reconstructed as uniform samples on [0, α_max] deg.
     First GRF series has no Hc label and is not used for Hc interpolation.

2-10 Scaled reflected impulse vs angle of incidence. From 02_010.grf. Same
     applicability as Figure 2-9. Y-axis title in the GRF does not print a unit
     string; treated as UFC English psi-ms/lb^(1/3).

2-13 Scaled height of triple point HT/W^(1/3) vs scaled horizontal distance
     R/W^(1/3), both ft/lb^(1/3). From 02_013.grf. Published Hc curves 1, 1.5,
     2, 2.5, 3, 3.5, 4, 5, 6. Hc=7 is annotated but has no Y samples.

2-15 Hemispherical surface-burst positive-phase parameters. Workbook
     DataHemiSpherical, verified against 02_015 (1).grf.

Not in these files
------------------
Figures 2-11 and 2-12 (air-burst environment / pressure-time schematics) are
not quantitative HT(R) sources. CONWEP is a separate family; the workbook
Friedlander history is labeled UFC Calc, not CONWEP.

Decay parameter b
-----------------
Columns "b incident" and "b reflected" in the workbook satisfy the modified
Friedlander identity I = P t0 (b-1+exp(-b))/b^2 to ~1e-8 relative residual.
They are derived, not UFC figure series.

Rebuild
-------
From the repo root, with the supplied files in Downloads:

    python _ref_src/build_ufc_json.py
