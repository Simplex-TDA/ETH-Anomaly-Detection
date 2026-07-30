# Ethereum Topological Anomaly Detection (ETH-TAD)

Topological Data Analysis (TDA) pipeline for detecting anomalies in Ethereum transaction networks.

**Re-implementation of:** Ofori-Boateng et al. (2021) — *Topological Anomaly Detection in Dynamic Multilayer Blockchain Networks* ([arXiv:2106.01806](https://arxiv.org/abs/2106.01806))

Predictive Validation: Beyond anomaly detection, we validated that the extracted topological descriptors provide statistically significant predictive value in controlled forecasting experiments. See predictive-validation/ for details.

## Overview

This project uses persistent homology to analyze the daily evolution of Ethereum's transaction network and detect structural anomalies that may indicate significant events (market crashes, protocol changes, DeFi exploits, etc.).

### Key Features

- **Multi-layer analysis**: Separate analysis of contract interactions, simple transfers, and ERC20 tokens
- **Topological persistence**: Captures network structure evolution using Vietoris-Rips filtration
- **Multi-method anomaly detection**: Ensemble of 5 statistical methods with consensus voting
- **Change-point detection**: Automatically identifies natural periods in the data
- **Scalable**: Handles years of Ethereum data with efficient chunking and caching

## Pipeline Overview

```
1. Data Collection (Notebook 1)
   ├─ Download ETH native transactions from Xatu ClickHouse
   ├─ Download ERC20 token transfers
   └─ Aggregate to daily/weekly graphs

2. Node Ranking (Notebook 2)
   ├─ Apply filters (contract vs simple txs, specific tokens, etc.)
   ├─ Rank nodes by centrality (PageRank, betweenness, k-core, etc.)
   └─ Generate global + daily top-node lists

3. TDA Analysis (Notebook 3)
   ├─ Filter graphs to top-ranked nodes
   ├─ Compute geodesic distance matrices
   ├─ Run persistent homology (H0, H1)
   ├─ Generate persistence diagrams
   └─ Compute Wasserstein distances between consecutive days

4. Anomaly Detection (Notebooks 4a, 4b, 5)
   ├─ Yearly analysis (4a) or organic period detection (4b)
   ├─ 5 anomaly detection methods: S-ESD, IQR, Z-score, Rolling σ, Isolation Forest
   ├─ Ensemble consensus voting
   └─ Rich visualizations with persistence diagram comparisons
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Simplex-TDA/ETH-Anomaly-Detection.git
cd ETH-Anomaly-Detection
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure credentials

Access to the Xatu ClickHouse database is required to download data. Request credentials from [ethpandaops](https://github.com/ethpandaops/xatu).

```bash
cp credentials_template.py credentials.py
# Edit credentials.py with your Xatu ClickHouse credentials
```

## Usage

### Quick Start

1. **Download Data** (notebook 1):
   - Open `1_dataFetcher.ipynb`
   - Set `YEAR` variable (e.g., 2020, 2021, 2022)
   - Run all cells to download both ETH and ERC20 data
   - Data will be saved to `data/{YEAR}/eth_tx_value_output/` and `data/{YEAR}/erc20_tx_value_output/`

2. **Generate Rankings** (notebook 2):
   - Configure filters and centrality metrics
   - Run global and daily rankings

3. **Run TDA Pipeline** (notebook 3):
   - Set layer configuration (single-layer vs multi-layer)
   - Execute full TDA pipeline
   - Generates persistence diagrams and Wasserstein distance series

4. **Detect Anomalies** (notebooks 4a/4b):
   - Choose yearly (4a) or period-based (4b) analysis
   - Review detected anomalies and persistence diagram visualizations

### Configuration Example

```python
# Notebook 3 - TDA Controller

YEARS = [2020, 2021, 2022]

LAYERS = {
    'contract_vs_simple': {
        'simple_txs':   lambda d: d['tx_value'] >  0,
        'contract_txs': lambda d: d['tx_value'] == 0,
    },
    'single_layer': None,  # No split - whole graph
}

TDA_CFG = {
    'edge_weight_col':  'tx_count',
    'max_nodes':        1000,
    'homology_maxdim':  1,  # H0 + H1
    'distance_metric':  'wasserstein',
    'ranking_metric':   'tx_count',
    'daily_top':        750,
}
```

## Project Structure

```
.
├── credentials_template.py     # Copy to credentials.py and fill in your credentials
├── requirements.txt            # Python dependencies
│
├── functions/                  # All reusable function modules
│   ├── eth_data_fetcher.py         # ETH transaction download
│   ├── erc20_data_fetcher.py       # ERC20 transfer download
│   ├── ranking_functions.py        # Node ranking algorithms
│   ├── tad_ethereum_functions.py   # Core TDA pipeline
│   ├── yearly_analysis_functions.py    # Yearly anomaly detection
│   ├── period_analysis_functions.py    # Multi-year period analysis
│   └── period_visualisation_functions.py  # Visualisation
│
├── 1_dataFetcher.ipynb             # Download ETH & ERC20 data
├── 2_ranking.ipynb                 # Generate node rankings
├── 3_tda.ipynb                     # Run TDA pipeline
├── 4a_yearly_analysis.ipynb        # Yearly anomaly detection
├── 4b_period_analysis.ipynb        # Period-based anomaly detection
└── 5_period_visualisation.ipynb    # Visualisation of periods
```

## Output Files

- `data/{YEAR}/eth_tx_value_output/weekly/` - ETH transaction data
- `data/{YEAR}/erc20_tx_value_output/weekly/` - ERC20 transfer data
- `data/ranking/{YEAR}/` - Node rankings
- `run_results_V1_{YEAR}.json` - TDA results per year
- `analysis_*.json` - Multi-year analysis results

## Key Concepts

### Persistent Homology

- **H0 (0-dimensional homology)**: Connected components
- **H1 (1-dimensional homology)**: Loops/cycles in the network
- **Persistence diagrams**: Birth/death times of topological features
- **Wasserstein distance**: Measures dissimilarity between diagrams

### Anomaly Detection Methods

1. **S-ESD**: Seasonal Extreme Studentized Deviate (Grubbs test)
2. **IQR**: Interquartile range fence method
3. **Z-score**: Rolling window standardization
4. **Rolling σ**: Deviation from rolling mean
5. **Isolation Forest**: Unsupervised outlier detection
6. **Consensus**: Ensemble voting (≥2 methods agree)

## Citation

If you use this code, please cite:

```bibtex
@article{ofori2021topological,
  title={Topological Anomaly Detection in Dynamic Multilayer Blockchain Networks},
  author={Ofori-Boateng, Djintole and Akcora, Cuneyt Gurcan and Islambekov, Umar and Gel, Yulia R and Kantarcioglu, Murat},
  journal={arXiv preprint arXiv:2106.01806},
  year={2021}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- **Data source**: [Xatu](https://github.com/ethpandaops/xatu) by ethpandaops — Ethereum data platform
- **TDA libraries**: [Ripser](https://github.com/scikit-tda/ripser.py), [Persim](https://github.com/scikit-tda/persim)
- **Original paper**: Ofori-Boateng et al. (2021)

## Contact

For questions or issues, please open a [GitHub issue](https://github.com/Simplex-TDA/ETH-Anomaly-Detection/issues).
