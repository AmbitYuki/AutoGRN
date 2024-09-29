# AutoGRN: An Adaptive Multi-Channel Graph Recurrent Joint Optimization Network with Copula-based Dependency Modeling for Spatio-Temporal Fusion in Electrical Power Systems
﻿
## Overview
﻿
AutoGRN (Automated Graph Recurrent Network) is an advanced machine learning tool designed for modeling and predicting complex temporal dependencies in power systems. By leveraging sophisticated algorithms, AutoGRN autonomously infers and constructs models to analyze and forecast time-series data in power systems, offering a robust and automated solution for accurate power forecasting.
﻿
## Project Structure
﻿
```
.
├── lib/                    # Core library files
├── model/                  # Model definition and related files
├── scripts/                # Utility scripts
├── MultiWaveletCorrelation.py  # Implementation of multi-wavelet correlation
├── masking.py              # Masking utilities
├── requirements.txt        # Python dependencies
└── train.py                # Main training script
```
﻿
## Installation
﻿
To set up the project environment:
﻿
1. Clone the repository:
```
git clone https://github.com/ambityuki/AutoGRN.git
cd AutoGRN
```
﻿
2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
﻿
3. Install the required dependencies:
```
pip install -r requirements.txt
```
﻿
## Usage
﻿
To train the model:
﻿
```
python train.py [options]
```
﻿
For detailed information on available options and configurations, please refer to the documentation in the `scripts/` directory.
﻿
## Key Features
﻿
- Autonomous inference of graph structures in power systems
- Advanced time-series prediction capabilities
- Integration of multi-wavelet correlation techniques
- Flexible masking utilities for data preprocessing
﻿
## Contributing
﻿
We welcome contributions to the AutoGRN project. Please read our contributing guidelines before submitting pull requests.
﻿
```
﻿
## Contact
﻿
For questions and feedback, please contact us on Github.
