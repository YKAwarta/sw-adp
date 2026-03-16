# OPSSAT-AD stands alone for space weather–telemetry fusion

**OPSSAT-AD is indeed the only publicly available satellite telemetry anomaly detection dataset that combines real UTC timestamps, expert anomaly annotations, and open access—making it uniquely suitable for alignment with space weather indices.** An exhaustive search across Zenodo, Kaggle, IEEE DataPort, NASA open data, ESA archives, GitHub, and academic supplementary materials (2020–2026) found no dataset that simultaneously satisfies all three requirements. Several candidates come tantalizingly close—ESA's large-scale ADB benchmark anonymizes its timelines, Mars Express thermal data lacks anomaly labels, and GOCE housekeeping telemetry has timestamps but no formal annotations—but none crosses the finish line. The claim is defensible, though the landscape contains important near-misses worth acknowledging.

---

## The three-criteria filter eliminates every alternative

The search evaluated every publicly known satellite/spacecraft telemetry dataset against three non-negotiable requirements: (1) real, non-anonymized UTC timestamps enabling temporal alignment with NOAA Kp, Dst, F10.7, or OMNI2 solar wind data; (2) expert-curated anomaly labels or annotations; and (3) unrestricted public availability for download without institutional approval.

**OPSSAT-AD** (Zenodo DOI: 10.5281/zenodo.12588358) meets all three. Published in *Nature Scientific Data* (2025), it contains **2,123 telemetry segments** from ESA's OPS-SAT 3U CubeSat across 9 channels (3 magnetometer, 6 photodiode), with 434 anomalous segments (~20% anomaly rate) labeled by ESA operations engineers and ML experts. The timestamps are in ISO date format from real mission operations (launched December 2019, reentered May 2024). The temporal coverage of the labeled dataset spans approximately January–June 2022, during which the maximum observed Kp was **6.7**—moderate storm activity but not extreme.

No other dataset passed all three filters. The table below summarizes all candidates evaluated:

| Dataset | Real UTC timestamps | Anomaly labels | Publicly available | Verdict |
|---|---|---|---|---|
| **OPSSAT-AD** | ✅ ISO dates | ✅ Expert-curated | ✅ Zenodo CC-BY-4.0 | **Only qualifying dataset** |
| ESA-ADB (3 missions) | ❌ Anonymized/shifted | ✅ Rich annotations | ✅ Zenodo | Fails timestamp criterion |
| NASA SMAP/MSL | ❌ No timestamps at all | ✅ Expert-labeled | ✅ GitHub/S3 | Fails timestamp criterion |
| Mars Express Thermal | ✅ Real dates | ❌ None | ✅ Figshare/Zenodo | Fails annotation criterion |
| GOCE Housekeeping | ✅ Real dates | ❌ No formal labels | ✅ ESA Dissemination | Fails annotation criterion |
| SatNOGS Database | ✅ Reception timestamps | ❌ None | ✅ API CC-BY-SA-4.0 | Fails annotation criterion |
| EgyptSat-1 (Elfiky 2025) | ✅ Real dates | ✅ Derived labels | ❌ Upon request only | Fails availability criterion |
| MisrSat-2 (Mostafa 2025) | ✅ Real dates | ✅ Statistical labels | ❌ Upon request only | Fails availability criterion |
| CATS | ❌ Synthetic integer indices | ✅ Injected anomalies | ✅ Zenodo | Synthetic, not real spacecraft |
| NASA ADAPT | ⚠️ Testbed timestamps | ✅ Injected faults | ✅ data.nasa.gov | Ground testbed, not orbital |
| ISS Live Telemetry | ✅ Real-time UTC | ❌ None | ⚠️ Stream only, no archive | No archive, no labels |
| LASP WebTCAD | ⚠️ Possibly real | ❌ None | ✅ GitHub | No labels |

---

## ESA's flagship benchmark deliberately blocks space weather alignment

The **ESA Anomaly Detection Benchmark (ESA-ADB)** is the largest and most carefully annotated satellite telemetry anomaly dataset ever published—approximately **31 GB** of data from three ESA missions with **1,430 annotated events** curated by ESOC spacecraft operations engineers. Yet it was explicitly designed to prevent temporal alignment with real-world events. The accompanying paper (Kotowski et al., arXiv:2406.17826) states: *"Datasets and results are anonymised to avoid disclosing sensitive mission-specific information, such as channel names, timelines, or types of measured values."* More pointedly, the authors acknowledge this *"does prevent from using physics-informed approaches or domain-specific knowledge to design algorithms (for example, to match telecommands and channels by names or to expect anomalies in specific times, e.g. during increased solar activity)."*

This was a deliberate design choice to enforce "universal data-driven approaches." The timestamps in the data files exist (the data is irregularly sampled time-series), but they are **shifted and anonymized** so that the true observation dates cannot be recovered. Mission identities are concealed as "ESA-Mission1," "ESA-Mission2," and "ESA-Mission3." Channel names are replaced with anonymous codes. This makes ESA-ADB entirely unsuitable for space weather–telemetry fusion, despite being the gold standard for general-purpose spacecraft anomaly detection benchmarking.

---

## NASA SMAP/MSL datasets contain no temporal information whatsoever

The NASA SMAP and MSL datasets (Hundman et al., KDD 2018) are **confirmed fully anonymized** with zero possibility of recovering real dates. The repository README explicitly states: *"All data has been anonymized with regard to time"* and *"no identifying information related to the timing or nature of commands is included in the data."* The data files are NumPy arrays indexed by sequential integers—**no timestamp column exists in any form**. Telemetry values are pre-scaled to (−1, 1), channel IDs are anonymized (only type prefixes preserved), and the sampling rate is unspecified. These datasets are fundamentally incompatible with space weather alignment. Additionally, recent work (Wu & Keogh 2023, IEEE TKDE) has flagged these datasets as problematic even for general benchmarking due to "unrealistic anomaly density, many trivial anomalies, mislabelled ground truth, and distributional shifts."

---

## No mission released housekeeping telemetry from major Solar Cycle 25 storms

A targeted search for public telemetry data covering the **May 2024 G5 superstorm** (Dst ≈ −412 nT, first G5 since 2003), **October 2024 storms** (Dst ≈ −335 to −390 nT), and the **February 2022 Starlink loss event** found no actual spacecraft housekeeping telemetry releases. What exists instead falls into three categories:

- **Scientific instrument data** (publicly available but measures space environment, not spacecraft health): NASA CIRBE particle flux measurements, ESA Swarm magnetic field and thermospheric density data, Macau Science Satellite-1 radio occultation observations.
- **Orbital element data** (publicly available from Space-Track): TLE-based drag analysis used in multiple papers (e.g., Starlink-1089 premature reentry analysis by Oliveira et al. 2025), but these are orbital parameters, not onboard telemetry.
- **Reports without data**: GOES-16 stopped transmitting all data for ~2 hours during the May 2024 storm; SpaceX lost 38 Starlink satellites in February 2022; the Binar CubeSat constellation experienced premature reentry. In each case, only operational impact reports or model simulations were published—**no internal housekeeping telemetry was released**.

A Copernicus paper (Guarnieri et al. 2025) explicitly lamented this gap: *"The scientific community would greatly benefit if companies operating satellites in these low orbits could make a 'scientific dataset' derived from their telemetry data public."*

The NARSS group's two papers are particularly relevant near-misses. **Elfiky et al. (2025)** studied EgyptSat-1 power telemetry (2007–2010) fused with 23 space weather parameters, while **Mostafa et al. (2025)** analyzed MisrSat-2 power subsystem data during the May 2024 G5 storm at 5-minute resolution—exactly the type of space weather–telemetry fusion dataset needed. However, neither released their satellite telemetry. Mostafa et al.'s data availability statement reads: *"The datasets generated and/or analyzed during the current study are available from the corresponding author upon reasonable request."* A GitHub repository (github.com/Hamada224/MisrSat2-Reproducibility) contains only analysis code with no data files.

---

## Three near-miss datasets deserve acknowledgment

While no alternative fully qualifies, three datasets warrant discussion as they partially address the gap:

**GOCE Housekeeping Telemetry Database** is the most notable near-miss. ESA published **all GOCE platform and instrument telemetry parameters** for the satellite's entire operational lifetime (March 2009–November 2013) through the GOCE Online Dissemination Service. This includes spacecraft bus parameters—temperatures, voltages, currents—with real UTC timestamps and free access. However, it lacks formal anomaly annotations. Known anomaly events (a 2010 computer communication malfunction, safe-mode entries, and the 2013 controlled reentry phase) are documented in ESA mission reports but have not been curated into machine-readable labels. A researcher could potentially construct an annotated dataset by cross-referencing these mission reports with the telemetry archive, but this requires substantial effort.

**Mars Express Thermal Power Dataset** (Petković et al., *Nature Scientific Data* 2022; Figshare DOI: 10.6084/m9.figshare.c.5360420.v1) provides ~8 years of real thermal power consumption telemetry (33 power line current measurements plus context features) with real timestamps at multiple resolutions (1-min to 60-min). It covers August 2008–March 2016 (Figshare) and January 2015–October 2020 (Zenodo orbit-level aggregation). Designed for power prediction, it has no anomaly annotations, though conjunction periods and known spacecraft events could serve as proxies.

**SatNOGS Telemetry Database** offers the broadest temporal and mission coverage—crowdsourced decoded telemetry from hundreds of amateur and university CubeSats with UTC reception timestamps, fully accessible via REST API. Many satellites in the database were operational during the May 2024 and October 2024 storms. However, the data has no anomaly labels, variable decode quality, and would require significant curation effort to create a usable anomaly detection dataset.

---

## Conclusion

The claim that OPSSAT-AD is the only publicly available, annotated satellite telemetry dataset with real timestamps suitable for space weather alignment is **well-supported by the evidence**. The dataset occupies a unique position at the intersection of three requirements that no other public resource satisfies simultaneously. The ESA-ADB benchmark—despite being far larger and more richly annotated—deliberately anonymizes its timelines specifically to prevent the kind of temporal correlation with space weather events that fusion research requires. NASA's SMAP/MSL datasets strip all temporal information entirely. The most promising storm-period data (NARSS group's EgyptSat-1 and MisrSat-2 analyses) remains locked behind "available upon request" policies.

The most actionable path to expanding beyond OPSSAT-AD would be constructing new annotated datasets from **GOCE housekeeping telemetry** (by cross-referencing ESA mission anomaly reports), from **SatNOGS data** (by systematically collecting CubeSat telemetry during known storm periods and applying expert or semi-automated labeling), or by formally requesting data from the NARSS group. The gap is real, widely recognized in the community, and unlikely to close soon given the security-driven anonymization practices of major space agencies.