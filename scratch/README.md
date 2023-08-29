# Prompt Risk Control
Prompt risk control is a framework for selecting prompts that minimize a risk criterion (e.g., error rate, toxicity, etc.). This framework accounts for more than just empirical average performance on a validation set when selecting a prompt. It takes into account the worst-case performance of a prompt through the use of metrics like conditional value at risk (CVaR). While problematic generations are rare for many language models, they can be catastrophic in real-world applications. This framework allows users to select prompts that minimize the risk of such generations.

## Environment Setup
To install the dependencies, run the following command:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```