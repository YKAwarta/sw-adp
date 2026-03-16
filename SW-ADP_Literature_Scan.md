# SW-ADP literature scan: your novelty is narrowed but defensible

**The space weather + telemetry fusion idea for ML-based anomaly detection has been partially published** — two papers by a NARSS (Egypt) research group appeared in late 2025 in *Scientific Reports*, combining space weather indices with satellite telemetry and applying ML. However, neither performs the systematic ablation study that is SW-ADP's core experiment, neither works on CubeSat data, and neither uses the same anomaly detection paradigms (Isolation Forest, Matrix Profile, LSTM/TCN autoencoders with conformal prediction calibration). The novelty claim survives but must be carefully reframed. The OPSSAT-AD benchmark — SW-ADP's primary dataset — explicitly lacks space weather features, creating a recognized gap the project can fill. Established methods for spacecraft anomaly detection are mature (LSTM-NDT, TranAD, Anomaly Transformer all exceed **90% F1** on SMAP/MSL benchmarks), and TinyML deployment on CubeSats is now proven with TensorFlow Lite running on ESA's OPS-SAT.

---

## Question 1: Spacecraft anomaly detection is a well-benchmarked field with clear leaders

The field of ML-based spacecraft telemetry anomaly detection crystallized around one seminal paper and two benchmark datasets, with two dominant paradigms now competing for state-of-the-art.

**The foundational benchmark: Hundman et al. (2018).** Kyle Hundman and colleagues at NASA JPL published "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding" at ACM KDD 2018. They trained individual LSTM models per telemetry channel on data from NASA's SMAP satellite (55 channels) and Mars Science Laboratory Curiosity rover (27 channels), predicting next telemetry values and flagging anomalies when prediction residuals exceeded a dynamically computed threshold. Their nonparametric dynamic thresholding (NDT) method selects the threshold that maximizes the reduction in mean and variance of smoothed errors — requiring no distributional assumptions. On 82 expert-labeled channels combined, they reported **precision of 87.5% and recall of 80.0%**, with point anomaly recall at 90.3% and contextual anomaly recall at 69.0%. The code was open-sourced as "Telemanom." This paper established SMAP/MSL as THE standard benchmarks and the prediction-error paradigm as the dominant approach for years.

**Transformer architectures now lead SMAP/MSL benchmarks.** Two papers have pushed state-of-the-art significantly:

- **TranAD** (Tuli, Casale & Jennings, VLDB 2022) uses a deep encoder-decoder transformer with adversarial two-phase training to amplify reconstruction errors for subtle anomalies. On SMAP it achieves **F1 = 0.892** with near-perfect recall (~1.00), and training completes in ~3.2 seconds (5 epochs) — up to **99% faster** than baselines. The adversarial training and model-agnostic meta-learning (MAML) for few-shot adaptation are particularly relevant for space applications where labeled anomalies are scarce.

- **Anomaly Transformer** (Xu et al., ICLR 2022 Spotlight) introduces "Association Discrepancy" — the insight that anomalous points exhibit only local, adjacent-concentrated attention patterns while normal points have broad associations across the series. On SMAP/MSL combined, it achieves **precision of 90.75% and recall of 95.30%**. A 2025 *Acta Astronautica* paper by an independent team systematically compared TranAD, Anomaly Transformer, and MEMTO on **ESA OPS-SAT telemetry**, finding MEMTO identified 97.99% and Anomaly Transformer 97.52% of true anomalies — confirming these methods transfer well to real CubeSat data.

**TCN and autoencoder approaches fill important niches.** Wang et al. (2021, IEEE I2MTC) demonstrated TCN-based prediction for spacecraft telemetry anomaly detection achieving **96.45% accuracy** with smaller inference times than LSTMs. Liu et al. (2023, *Neural Computing and Applications*) combined dynamic graph attention with TCN, outperforming LSTM-NDT on SMAP/MSL benchmarks. For autoencoder-based reconstruction approaches, USAD (Audibert et al., KDD 2020) uses two adversarially-trained autoencoders sharing a single encoder, offering very fast training and a tunable sensitivity parameter — valuable for operational deployment. Jin et al. (2019, *Sensors*) applied stage-training denoising autoencoders directly to satellite power subsystem telemetry, one of the subsystems most affected by space weather.

**The OPSSAT-AD benchmark is SW-ADP's most important reference.** Ruszczak, Kotowski, Evans & Nalepa (2025, *Scientific Data*) introduced the first publicly available, annotated CubeSat telemetry dataset: **2,123 single-channel segments** from 9 channels (3 magnetometer, 6 photodiode) of ESA's OPS-SAT 3U CubeSat, with 20% anomalous segments annotated by ESA operations engineers. Thirty algorithms were benchmarked, with Random Forest achieving ~95.7% accuracy using 18 handcrafted features. The paper explicitly notes SMAP/MSL limitations (unrealistic anomaly density, trivial anomalies, mislabeled ground truth). Critically, **OPSSAT-AD does not include space weather features** — this is the gap SW-ADP fills.

**Two additional gaps favor SW-ADP's approach.** No published work applies conformal prediction to spacecraft anomaly detection — this calibration layer providing distribution-free coverage guarantees is novel in this domain. Xu & Xie (2021, ICML) demonstrated conformal prediction intervals for time-series anomaly detection but not in spacecraft contexts. Similarly, Matrix Profile has been applied to space-adjacent telemetry (EDEN ISS greenhouse) but not directly to satellite anomaly detection despite its interpretability advantages.

### Key papers for Question 1

| Paper | Method | Venue/Year | Key Result | SW-ADP Relevance |
|-------|--------|------------|------------|-----------------|
| Hundman et al. | LSTM + NDT | KDD 2018 | P=87.5%, R=80.0% (SMAP+MSL) | Foundational benchmark |
| Tuli et al. (TranAD) | Adversarial Transformer | VLDB 2022 | F1=0.892 (SMAP), 99% faster training | SOTA comparison |
| Xu et al. (Anomaly Transformer) | Association Discrepancy | ICLR 2022 | P=90.75%, R=95.30% (SMAP+MSL) | SOTA comparison |
| Ruszczak et al. (OPSSAT-AD) | 30 algorithms benchmarked | Scientific Data 2025 | RF ~95.7% acc on CubeSat data | Primary dataset paper |
| Acta Astronautica 2025 | TranAD/AT/MEMTO on OPS-SAT | Acta Astronautica 2025 | MEMTO 97.99% anomaly recall | Transformer on CubeSat |
| Su et al. (OmniAnomaly) | GRU-VAE + POT | KDD 2019 | Competitive on SMAP/MSL | Reconstruction paradigm |
| Zhao et al. (MTAD-GAT) | Graph Attention + GRU | ICDM 2020 | ~1% improvement over SOTA | Inter-variable modeling |
| Audibert et al. (USAD) | Adversarial dual AE | KDD 2020 | Fast, tunable sensitivity | Lightweight AE baseline |
| Wang et al. | TCN prediction | IEEE I2MTC 2021 | 96.45% accuracy, fast inference | TCN for spacecraft |
| Liu et al. (ATCN) | Graph Attention + TCN | Neural Comput. Appl. 2023 | Outperforms LSTM-NDT | TCN+graph on SMAP/MSL |

---

## Question 2: The novelty claim survives but needs careful repositioning

This is the most consequential finding for the SW-ADP project. The landscape has three tiers: direct prior work that partially overlaps, statistical correlation studies that motivate the approach, and context-aware methods from other domains that provide methodological precedent.

**Two papers constitute direct prior work.** Both come from the Egyptian National Authority for Remote Sensing and Space Sciences (NARSS):

Elfiky, Ali, Mostafa, Hesham, Ahmed & Ibrahim (2025, *Scientific Reports* 15:43830) fuse space weather indices — proton flux (PF60, PF30, PF10), AL index, galactic cosmic rays, solar wind density, plus a broader set including SSN, F10.7, Kp, Ap, Dst, and IMF components — with EgyptSat-1 power subsystem telemetry (voltage, current, temperature). They apply CNN, LSTM, Random Forest, AdaBoost, and Gradient Boosting with a two-stage nonlinear feature selection (unsupervised RBM + supervised Mutual Information). Their best model (Random Forest) achieves **MSE = 0.00147, MAPE = 0.09%** for voltage prediction. Anomaly detection is performed on prediction residuals, with temporal coincidence rates of **31% with GCR disturbances and 27% with P10 proton events** (p < 10⁻³). This paper does ALMOST what SW-ADP proposes.

Mostafa, Ali, Hesham, Yassin, Ahmed & Elfiky (2025, *Scientific Reports* 15:37743) apply a hybrid statistical-ML framework to MisrSat-2 telemetry during the May 2024 G5 geomagnetic storm, using Mixture of Experts (MoE) with space weather parameters (solar wind speed, Dst, Kp, proton/electron fluxes). Their MoE achieves **R² = 0.921, MAE = 0.063 A**. They find all deviations were small (<4%) and within design tolerances.

**However, critical differences preserve SW-ADP's novelty.** Neither paper performs a formal ablation study comparing telemetry-only performance against telemetry + space weather — the core experiment of SW-ADP. Both use a regression-then-residual approach (predict telemetry from space weather, then flag residuals) rather than direct anomaly detection on concatenated feature vectors. Neither works on CubeSat data — EgyptSat-1 is a 1050 kg satellite, MisrSat-2 is similarly large, while CubeSats have fundamentally different shielding characteristics and space weather susceptibility. Neither uses Isolation Forest, Matrix Profile, or autoencoder-based anomaly detection. Neither applies conformal prediction calibration.

**Statistical correlation studies provide strong motivation.** A substantial body of work documents the space weather → satellite anomaly relationship without building ML detection systems. Iucci et al. (2005, *Space Weather*) compiled anomalies from 220 satellites (1971–1994) showing proton fluxes >1000 pfu increase anomaly rates by **~20× for high-altitude near-polar satellites**. Lohmeyer & Cahoy (2013, *Space Weather*) analyzed 665,112 operational hours of Inmarsat GEO telemetry against space weather data. Choi et al. (2011, *Space Weather*) correlated 95 GEO satellite anomalies with geomagnetic activity. Foreman Campins et al. (2024, *Aerospace*) analyzed LEO/GEO anomalies from 2010–2022 against SSN, CMEs, Kp, and Dst, finding strong equinoctial correlations. These papers collectively establish that the physical relationship exists and is statistically significant — **justifying SW-ADP's hypothesis that including space weather features should improve detection**.

**Environment-monitoring systems like SAAPS and SEAES-GEO predict anomaly probability FROM space weather alone** (no telemetry fusion). Wintoft & Eliasson's SAAPS uses neural networks trained on 8-day Kp histories to predict satellite anomaly events (~70% accuracy). O'Brien's SEAES-GEO (2009, *Space Weather*) computes data-driven hazard quotients at GEO from Kp levels and particle fluxes. These are space-weather-to-anomaly-probability models — they do not incorporate satellite telemetry, making them fundamentally different from SW-ADP.

**Context-aware anomaly detection from other domains provides methodological support.** Time-CAD (Nam et al., ECML-PKDD 2023) demonstrated that incorporating contextual information into deep time-series decomposition improves anomaly detection F1 by up to **46%** — a strong precedent for SW-ADP's hypothesis. The Temporal Fusion Transformer architecture (Lim et al., 2021) provides gating mechanisms for external/contextual variable integration.

### How to reposition the novelty claim

The student **cannot claim** that "fusing space weather with satellite telemetry for ML-based anomaly detection is novel." Elfiky et al. (2025) invalidates this broad claim. Instead, SW-ADP's contribution should be framed around five defensible novelty points:

- **Systematic ablation study**: No prior work quantifies the marginal improvement of adding space weather features to anomaly detection. SW-ADP's telemetry-only vs. telemetry+space weather comparison directly addresses this gap.
- **CubeSat-specific application**: CubeSats have minimal shielding and are disproportionately vulnerable to space weather — prior fusion work studied large (>1000 kg) satellites only.
- **Direct anomaly detection on fused features**: Rather than predicting telemetry from space weather and flagging residuals, SW-ADP performs anomaly detection directly on the concatenated feature space — a methodologically distinct approach.
- **Conformal prediction calibration**: No spacecraft anomaly detection work uses conformal prediction, providing statistical guarantees on false alarm rates.
- **Benchmark gap**: The OPSSAT-AD benchmark explicitly lacks space weather features; SW-ADP extends this benchmark.

---

## Question 3: TinyML on CubeSats is proven but constrained to lightweight models

On-board ML inference has moved from theory to orbit. The key question for SW-ADP is not whether it is possible but which architectures fit within CubeSat computational budgets.

**OPS-SAT proved TensorFlow Lite works in orbit.** Labrèche et al. (2022, IEEE Aerospace Conference) demonstrated three on-board AI experiments on ESA's OPS-SAT 3U CubeSat: CNN image classification using TensorFlow Lite C API inference, unsupervised k-means clustering, and online Random Forest training for fault detection (the "OrbitAI" experiment, achieving **89% balanced accuracy, F1 = 95%**). The hardware is an Altera Cyclone V SoC with dual-core ARM Cortex-A9 + FPGA — more powerful than a typical MCU but representative of modern CubeSat on-board computers. Models are uplinked as .tflite files. This directly validates SW-ADP's deployment pipeline.

**Φ-Sat-1 demonstrated deep CNN inference on a dedicated AI accelerator in orbit.** Giuffrida et al. (2022, *IEEE TGRS*) report on the first official DNN running on a satellite — the "CloudScout" CNN on an Intel Movidius Myriad 2 VPU aboard the FSSCat 6U CubeSat (launched September 2020). No functional degradation from radiation was observed. Φ-Sat-2 extends this with six AI applications running simultaneously, confirming the trajectory toward multi-model on-board AI.

**A 2023 *J. Aerospace Information Systems* paper directly validates anomaly detection on CubeSat-class MCUs.** Researchers deployed an ANN-based anomaly detection model on an **STM32H743 ARM Cortex-M7 microcontroller** — hardware comparable to SW-ADP's ESP32 target — using data from a NanoAvionics EduSat engineering model. The neural network detects temperature anomalies with improved precision and recall over traditional out-of-limit methods, with small enough memory overhead to coexist with existing flight software. This is the closest hardware precedent for SW-ADP's deployment target.

**TCN beats LSTM for edge deployment across every metric that matters.** Bai, Kolter & Koltun (2018, arXiv:1803.01271) — the foundational TCN paper — established that TCNs match or outperform LSTMs on most sequence modeling tasks while offering critical advantages for embedded systems: parallelizable inference (no sequential hidden state dependencies), fixed memory footprint (no recurrent state accumulation), stable gradients, and **~60× faster training**. Ingolfsson et al. (2021) demonstrated **19.6× energy savings** for TCN-based temporal modeling on wearable hardware. A 2025 *Sensors* paper compared SE-LSTM, TCN, and CNN-LSTM deployed on ESP32, ESP32-S3, and Raspberry Pi Pico via TensorFlow Lite + Edge Impulse, measuring **55–74 ms inference times on ESP32-S3** — confirming both architectures are viable for real-time CubeSat monitoring.

**Quantized autoencoders achieve remarkable compression.** A 2023 *Cybersecurity* (Springer) study on quantized autoencoders (QAE) for anomaly detection demonstrated that INT8 quantization achieves **~142 KB model size, 1.8 ms inference latency, and F1 = 98.7%** — with 92% memory compression and <1.5% accuracy loss compared to full-precision models. This fits comfortably within the ESP32's 520 KB SRAM and 4 MB flash. For SW-ADP's TCN Autoencoder specifically, INT8 post-training quantization via TFLite Micro should yield a deployable model under 200 KB.

**Hokkinen (2025, University of Oulu Master's thesis) demonstrated TinyML for aurora detection on a CubeSat**, deploying a CNN via MCUNet on an STM32 NUCLEO-F756ZG board: **430 KB flash, 104 ms per inference**. This provides a concrete benchmark for what MCU-class hardware can achieve on space-related classification tasks.

### Hardware landscape and SW-ADP implications

| Platform | Heritage | Capability | SW-ADP Fit |
|----------|---------|-----------|------------|
| ARM Cortex-A9 + FPGA (OPS-SAT) | Flight-proven | TFLite CNN/RF, online training | High — proven pipeline |
| Intel Movidius Myriad 2 (Φ-Sat-1) | Flight-proven | Deep CNN inference | Overkill for AD |
| STM32H743 Cortex-M7 | Lab-validated for CubeSat AD | ANN anomaly detection | Strong — closest MCU precedent |
| ESP32 (DLR APEX) | Sounding rocket only | TFLite via Edge Impulse, 55–74 ms | Good for prototyping |
| ESP32 lacks radiation hardening and space heritage for orbital missions — **STM32 is the more credible deployment target** for any claims about flight readiness. ESP32 is appropriate for the ground-based prototype demonstration at Boeing Expo.

---

## Where SW-ADP sits in the research landscape

SW-ADP occupies a genuinely interesting position at the intersection of three active research threads: spacecraft anomaly detection (mature, well-benchmarked), space weather effects on satellites (well-documented correlatively but rarely integrated into detection systems), and TinyML for space (proven but early). The project's contribution is **not the first to fuse space weather with telemetry** — that credit belongs to Elfiky et al. (2025) — but it offers the first systematic, controlled ablation study quantifying the value of this fusion, applied to CubeSat telemetry with modern anomaly detection methods and conformal prediction calibration.

For the Alfaisal poster competition and Boeing Expo, the recommended framing is: "Prior work has demonstrated that space weather indices correlate with satellite anomalies and that ML can model this relationship. SW-ADP asks a more precise question: **does adding space weather features measurably improve anomaly detection performance**, and by how much? We answer this through a controlled ablation study on ESA's OPSSAT-AD benchmark using four detection methods, with conformal prediction providing statistically calibrated detection thresholds." This positions the contribution as methodologically rigorous and practically valuable, without overclaiming novelty.

The strongest novel elements are the ablation study methodology, the conformal prediction calibration layer (completely unexplored in spacecraft AD), the CubeSat-specific focus, and the extension of the OPSSAT-AD benchmark with space weather context. The weakest element is claiming the fusion itself is novel — cite Elfiky et al. (2025) and Mostafa et al. (2025) prominently and differentiate clearly.