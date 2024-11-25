# CONCEALER Demo
Welcome to the CONCEALER repository. Here, we offer the implementation details of the method introduced in our research paper titled "_CONCEALER: Generative Mimicry Attacks against Provenance-based Intrusion Detection Systems_".

## Requirement
We have provided a requirements.txt file detailing the specific dependency versions.

## Code Structure

### Preparation
dataset/{dataset_name}/parser.py

### Graph Pruning
src/pids/{pids_name}/pruner.py

### Binding Site and Concealer Size Predictions
Training:  src/predictor/train_predictor.py

Testing:   src/predictor/test_predictor.py

### Diffusion Model
Training:  src/concealer/train_concealer.py

Testing:   src/concealer/test_concealer.py

## NOTE
The source code supported only the SupplyChain Dataset. We will support the Atlas and StreamSpot Datasets after careful checking.

We have provided optional resources (e.g., adversarial graphs and raw dataset files), which can be downloaded according to a download_url.txt.
