# 🌌 Gravitational Wave Phase Classification
### Binary Black Hole Merger · NASA Waveform Catalog · BRICS Astronomy & IDIA Program

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-RandomForest-orange?style=flat-square)
![NASA](https://img.shields.io/badge/Data-NASA%20AstroGravS-red?style=flat-square)
![Score](https://img.shields.io/badge/Capstone%20Score-89%2F100-brightgreen?style=flat-square)
![Program](https://img.shields.io/badge/BRICS%20Astronomy-IDIA%20Program-blueviolet?style=flat-square)

---

## 🏆 Program Context

This project was completed as a **capstone submission** for the internationally selective  
**BRICS Astronomy Working Group × IDIA (Inter-University Institute for Data Intensive Astronomy)**  
data-intensive astronomy training program.

> Capstone Score: **89 / 100**

---

## 🌠 What Are Gravitational Waves?

When two black holes spiral toward each other and collide, they send ripples through  
the fabric of spacetime — **gravitational waves**. These ripples travel at the speed of light  
and were first directly detected by LIGO in 2015 (Nobel Prize in Physics, 2017).

Every binary black hole merger produces a waveform with **three distinct physical phases**:

| Phase | What's Happening | Waveform Signature |
|---|---|---|
| **Inspiral** | Black holes orbit and spiral inward | Amplitude gradually increases — the "chirp" |
| **Merger** | Collision at peak gravitational radiation | Sharp spike — maximum strain amplitude |
| **Ringdown** | New black hole settles to equilibrium | Exponential amplitude decay — like a struck bell |

> **Goal:** Automatically detect and classify these three phases from raw waveform signals  
> using time-series feature engineering and a machine learning classifier.

---

## 📡 Dataset

| Property | Details |
|---|---|
| **Source** | [NASA AstroGravS Waveform Catalog](https://asd.gsfc.nasa.gov/archive/astrogravs/docs/waveforms/NRmergers/) |
| **File** | `R1` — equal-mass, zero-spin binary black hole waveform (GSFC QC6) |
| **Format** | 3-column ASCII: `time`, `Re(h₂₂)`, `Im(h₂₂)` |
| **Mode** | Dominant (l=2, m=2) gravitational wave mode |
| **Units** | Geometric units: G = c = 1 |

The **(2,2) mode** is the strongest harmonic of the gravitational wave —  
the "middle C" of black hole collisions — and the primary signal LIGO detects.

The dataset was chosen for its **simulation-based, noise-free nature**,  
enabling precise validation of the classification pipeline without real-detector noise contamination.

---

## 🧠 Where the Real Thinking Happened

### 1. Feature Design — Translating Physics into Math

The raw waveform gives you only `Re(h₂₂)` and `Im(h₂₂)` — complex numbers.  
Three features were independently designed to capture the physical behaviour of each phase:

```python
# Feature 1: Amplitude — captures total wave energy at each moment
data['amplitude'] = np.sqrt(data['Re_h22']**2 + data['Im_h22']**2)

# Feature 2: Slope — first derivative, detects phase transitions
data['slope'] = data['amplitude'].diff().fillna(0)

# Feature 3: Rolling Mean — window=50, suppresses local noise, reveals trends
data['rolling_avg'] = data['amplitude'].rolling(window=50, center=True).mean()
```

**Why these three?**
- **Amplitude** alone can't separate phases — inspiral and ringdown both have lower amplitude than merger
- **Slope** is the key discriminator — positive-and-growing (inspiral), chaotic (merger), negative-and-decaying (ringdown)
- **Rolling mean** smooths micro-fluctuations so the model sees the underlying pattern, not noise

This reasoning was **confirmed post-training by feature importance rankings** from the Random Forest model.

---

### 2. Time Normalisation — Peak Alignment

```python
# Normalize time so merger peak = t=0
peak_index = data['amplitude'].idxmax()
data['time'] -= data.loc[peak_index, 'time']
```

This ensures the phase boundary definitions are physically meaningful:
- **Inspiral:** `time < -10`
- **Merger:** `-10 ≤ time ≤ 10`
- **Ringdown:** `time > 10`

---

### 3. Class Imbalance — A Problem Most Miss

```
Inspiral samples:  163
Merger samples:     14
Ringdown samples:   ~N
```

A naive model could achieve high **overall accuracy** by simply predicting  
"inspiral" for everything — and still look good on paper.

**The fix:** Evaluated using **per-class precision, recall, and F1-score separately**  
rather than relying on overall accuracy, ensuring minority class (merger) detection  
was genuinely robust — not masked by the dominant class.

---

### 4. Model Training

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 70/30 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
```

---

## 📊 Results

| Metric | Inspiral | Merger | Ringdown |
|---|---|---|---|
| **Precision** | 98% | 100% | 85% |
| **Recall** | 96% | 100% | 100% |
| **F1-Score** | 97% | 100% | 92% |
| **Overall Accuracy** | **97%** | | |

### 🔍 Failure Diagnosis — The Most Important Finding

All misclassifications occurred **exclusively at the inspiral–ringdown boundary**.  
7 inspiral samples were misclassified as ringdown.

**Root cause:** Both phases share overlapping **amplitude decay signatures** near  
the phase edges — the amplitude is low and falling in both cases, making them  
difficult to distinguish purely from amplitude, slope, and rolling mean.

> This is a **physics-driven boundary overlap**, not an algorithmic failure.  
> The model performed exactly as well as the chosen feature set allows.

**Proposed fix:** Incorporating **Fourier spectral features** (frequency evolution)  
would resolve this — inspiral has rising frequency, ringdown has a fixed quasi-normal mode frequency.

---

## 📈 Visualisations

Three key plots produced:

**1. Gravitational Waveform — Amplitude vs. Time**
- Shows the full chirp → spike → decay structure with smoothed overlay

**2. Slope vs. Time**
- Reveals phase transition dynamics: accelerating slope (inspiral), chaotic spike (merger), decaying negative slope (ringdown)

**3. Predicted Phase Map**
- Scatter plot with colour-coded predictions: blue (inspiral), red (merger), green (ringdown) — visually confirms clean phase separation

---

## 🚀 Real-World Relevance for Space Applications

| This Project | Space Industry Application |
|---|---|
| Phase classification from time-series signals | Anomaly detection in satellite telemetry |
| Feature engineering on waveform data | Signal processing for in-orbit sensor arrays |
| Class imbalance handling | Rare fault detection in mission-critical systems |
| Physics-based failure diagnosis | Root cause analysis for spacecraft anomalies |

---

## 📁 Repository Structure

```
gravitational-wave-phase-classification/
│
├── BRICS_Astronomy_Capstone_Project.ipynb   ← Full notebook
├── README.md                                 ← You are here
└── requirements.txt                          ← Dependencies
```

---

## ⚙️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR-USERNAME/gravitational-wave-phase-classification
cd gravitational-wave-phase-classification

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Open the notebook
jupyter notebook BRICS_Astronomy_Capstone_Project.ipynb
```

The notebook loads data directly from NASA's public AstroGravS archive —  
**no manual dataset download required.**

```python
url = "https://asd.gsfc.nasa.gov/archive/astrogravs/docs/waveforms/NRmergers/data/GSFC_QC6_strain_2_2.dat"
data = pd.read_csv(url, delim_whitespace=True, comment='#', names=['time', 'Re_h22', 'Im_h22'])
```

---

## 🔭 Future Work

- Incorporate **Fourier spectral features** to resolve inspiral–ringdown boundary confusion
- Test **1D CNN** architecture for end-to-end waveform phase detection
- Apply methodology to **real LIGO data** from Gravitational Wave Open Science Center (GWOSC)
- Explore **second-derivative curvature** as an additional discriminating feature

---

## 📚 References

- NASA AstroGravS Archive: https://asd.gsfc.nasa.gov/archive/astrogravs/
- Baker et al. (2006): arXiv:gr-qc/0701016
- LIGO Scientific Collaboration: https://www.ligo.org/science/GW-Inspiral.php
- Scikit-learn Documentation: https://scikit-learn.org/stable/

---

## 👩‍💻 Author

**Mubeena Hussain**
MSc Statistics 
📧 mubeenahussain1205@gmail.com
🔗 [LinkedIn](www.linkedin.com/in/mubeena-hussain-a357b920b)
)

---

*"We are made of star-stuff. Now we can hear the stars collide."*
