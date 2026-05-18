Code used in the Master's Thesis by Tristan Lovšin. 

The code is based on the code used in the paper:
'AuToMATo: An Out-Of-The-Box Persistence-Based Clustering Algorithm',
which can be found here: https://arxiv.org/abs/2408.06958.

# Statistical Parameter Selection for Mapper: Evaluation and Extension with AuToMATo

[cite_start]This repository contains the computational framework, experiment scripts, and visualization pipelines developed for my Master's Thesis[cite: 4]. The project evaluates the statistical parameter selection methodology established by Carrière et al. (2018) [cite_start]for the Mapper algorithm[cite: 1, 199]. [cite_start]It expands its empirical validation across a wider suite of topological profiles and introduces **AuToMATo** (Automatic Topological Model Analysis Tool) as an alternative, persistence-based clusterer within the pipeline[cite: 2, 204, 214].

---

## 📦 Installation & Setup
To clone this repository along with its required submodules and set up the environment:

```bash
# Clone recursively to fetch external benchmarks and libraries
git clone --recursive [https://github.com/t-lovsin-barle/Masters_Thesis.git](https://github.com/t-lovsin-barle/Masters_Thesis.git)
cd Masters_Thesis

# Install dependencies
pip install -r requirements.txt
```
---

## 🎯 Experimental Objectives (Chapter 5)

[cite_start]The primary goal of the experimental battery was to systematically analyze how the optimal parameter selection framework responds under different clustering algorithms, filter functions, and varying noise thresholds. The evaluation is divided into three key areas:

1. **Replication Fidelity:** Attempting to independently reproduce the milestone Mapper results from the original Carrière et al. (2018) [cite_start]study using their parameter derivation math[cite: 13].
2. [cite_start]**Algorithmic Benchmarking:** Contrasting three distinct clustering methods within the Mapper pipeline across identical data setups[cite: 9]:
   * [cite_start]**Rips Clustering:** The baseline approach mapping connected components of a $\delta$-Rips graph[cite: 9, 22].
   * [cite_start]**Default AuToMATo:** The persistence-based clusterer running on its default $k$-nearest neighbors ($k=10$) graph structure[cite: 9, 134].
   * [cite_start]**Tuned AuToMATo:** A custom configuration matching the underlying $\delta$-Rips graph of the baseline and utilizing a calibrated Distance-to-Measure (DTM) parameter ($k=\left\lceil mn\right\rceil$) to enhance topological stability[cite: 9, 10].
3. [cite_start]**Robustness & Noise Testing:** Evaluating performance across 11 different datasets [cite: 2][cite_start], including complex 3D meshes (Stanford Bunny) [cite: 41][cite_start], synthetic geometric shapes (Clustering Benchmarks library) [cite: 41, 60][cite_start], and a controlled environment with incremental noise levels[cite: 43].

---

## 🔍 Key Findings & Nuanced Contributions

Rather than claiming flawless optimization, the experiments revealed critical structural insights regarding how Mapper behaves in practice:

### 1. Nuances in Practical Replication
* [cite_start]**Miller-Reaven (Diabetes) Dataset:** Utilizing the paper's mathematical parameter definitions (with default sequence constant $c=10$) did not seamlessly replicate the original clean two-flare output[cite: 23, 24]. [cite_start]At the recommended gain ($g=0.4$), an uncharacteristic third flare emerged[cite: 24]. [cite_start]The expected topological structure was successfully recovered only after manually adjusting the gain downward to $0.35$[cite: 25].
* [cite_start]**COIL-100 (Duck) Dataset:** The baseline parameter script failed to isolate the clean intrinsic loop at $g=0.4$[cite: 30]. [cite_start]Achieving a noise-free, 360-degree topological cycle required raising the gain to $0.49$[cite: 31]. [cite_start]This variation highlights how undocumented preprocessing details, such as specific image grayscale vectorization techniques, dramatically alter Mapper's resolution thresholds[cite: 33, 36].

### 2. Comparative Performance of Tuned AuToMATo
* [cite_start]**In Clean Environments:** For structurally straightforward datasets like the *Circles* benchmark, standard Rips clustering remains highly effective[cite: 66]. [cite_start]Default AuToMATo struggled by over-segmenting continuous spaces into artificial connected components [cite: 68][cite_start], while Tuned AuToMATo occasionally risked dropping finer 1D homological loops[cite: 67].
* [cite_start]**In Noisy Environments:** Testing co-centric circles under controlled noise increments revealed that standard Rips clustering completely loses the inner loop once noise reaches 6%[cite: 135]. [cite_start]Conversely, **Tuned AuToMATo significantly out-performed the baseline in noisy settings**, successfully preserving the core 1-dimensional homological features when provided with adequate gain[cite: 131, 139].
* [cite_start]**Graph Geometry Constraints:** Complex shapes like the *Stanford Bunny* demonstrated that spatial filters remain highly sensitive to directional alignment[cite: 53]. [cite_start]For instance, a first-principal-component (PCA) filter compressed features orthogonal to its vector, rendering the bunny's ears entirely invisible across all clusterers[cite: 53, 56]. [cite_start]Furthermore, Tuned AuToMATo occasionally introduced minor homological artifacts due to local overclustering on dense 3D structures[cite: 50, 52].

### 3. Vulnerability to Isolated Outliers
* [cite_start]Testing the *Target* benchmark from the Clustering Benchmarks library exposed a key limitation in the baseline parameter selection methodology[cite: 122, 126]. [cite_start]The mathematical framework's technique for computing the Rips threshold ($\delta_n$) is highly susceptible to isolated spatial outliers[cite: 126]. 
* [cite_start]The presence of distant, low-density clusters forced the resolution to become too coarse, completely collapsing the central 1-dimensional loop and merging distinct components[cite: 125, 126]. [cite_start]This finding underscores that the statistical framework requires robust upstream data denoising or outlier filtering to stay effective in real-world settings[cite: 126, 140].

---

## 📂 Codebase Structure

* `core/` — Contains core modules including `custom_cover.py`, `custom_clusterer.py`, and the mathematical logic for parameter estimation (`helper_functions.py`).
* [cite_start]`experiments/` — Executable Python scripts mapping out the 11 test batteries (Diabetes, COIL Duck, Stanford Bunny, and individual Clustering Benchmarks)[cite: 2].
* [cite_start]`external/` — Submodule directory containing third-party integrations, specifically `clustering_benchmarks` and the `automato` library[cite: 3].
* [cite_start]`figures/` — Vector outputs (.svg) showing the generated Mapper graphs across different gain levels and filters[cite: 44, 45].

---

## 📚 References & Academic Citations

* **Statistical Parameter Selection Framework:** Carrière, M., Michel, B., & Oudot, S. (2018). [cite_start]*Statistical analysis and parameter selection for mapper.* Journal of Machine Learning Research[cite: 199].
* **AuToMATo Algorithm:** Huber, M. A. (2024). [cite_start]*Automatic Topological Model Analysis Tool.* [GitHub Repository](https://github.com/m-a-huber/automato)[cite: 3].
* **Clustering Benchmarks Suite:** Gagolewski, M. (2022). [cite_start]*A benchmark suite for clustering algorithms.* [GitHub Repository](https://github.com/paalka/clustering_benchmarks)[cite: 41, 60].
* **COIL-100 Dataset:** Nene, S. A., Nayar, S. K., & Murase, H. (1996). [cite_start]*Columbia Object Image Library.* Technical Report[cite: 26].
* **Stanford Bunny:** Turk, G., & Levoy, M. (1994). [cite_start]*Zippered polygon meshes from range images.* SIGGRAPH[cite: 41, 46].
