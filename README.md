# DSC180B - In-Context Learning Testing Platform

A comprehensive platform for testing and evaluating In-Context Learning (ICL) with commercial Large Language Models (LLMs).

## 🚀 Quick Links

- **🌐 [Landing Page](https://lbangayan.github.io/DSC180B-Website/)** - Project overview and features
- **💻 [Interactive App](https://dsc180b-website.streamlit.app)** - Test ICL with commercial LLMs
- **📊 [GitHub Repository](https://github.com/Lbangayan/DSC180B-Website)**

## 📋 Project Overview

This project provides tools to understand and test In-Context Learning (ICL) across different parameters and commercial LLM providers. You can:

- **Test single configurations** with detailed result analysis
- **Run batch tests** with parameter sweeps in parallel
- **Visualize results** in an interactive dashboard
- **Compare performance** across multiple LLM providers (GPT-4, Claude, Gemini)

## 🎯 Key Features

### Single Test Mode
Test one ICL configuration at a time with immediate feedback and detailed metrics.

### Batch Testing
Run comprehensive parameter sweeps:
- Dataset size (d)
- Number of examples (N)
- Rank (R)
- Label flip probability (flip_prob)

### Results Dashboard
Track and visualize all your test results with interactive charts and tables.

### Multiple LLM Providers
- OpenAI GPT-4
- Anthropic Claude 3
- Google Gemini

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Lbangayan/DSC180B-Website.git
cd DSC180B-Website
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
Create a `.env` file in `src/icl_commerical_model_testing/`:
```
GEMINI_API_KEY=your_gemini_key_here
CLAUDE_API_KEY=your_anthropic_key_here
GPT_API_KEY=your_openai_key_here
```

## 🚀 Running the App

### Local Development
```bash
cd src/icl_commerical_model_testing
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Online Deployment
Visit the [live app](https://dsc180b-website.streamlit.app) to test without local setup.

## 📁 Project Structure

```
DSC180B-Website/
├── docs/                          # GitHub Pages static site
│   └── index.html                # Landing page
├── src/
│   ├── icl_commerical_model_testing/
│   │   ├── app.py               # Main Streamlit application
│   │   ├── llm_providers.py      # LLM API integrations
│   │   ├── .env                 # Local environment variables (not committed)
│   │   └── results/             # Test results storage
│   └── icl_reproduction/        # ICL model implementations
│       ├── models.py            # Core ICL models
│       ├── training.py          # Training utilities
│       ├── evaluation.py        # Evaluation metrics
│       ├── experiments/         # Experimental scripts
│       └── Notebooks/           # Jupyter notebooks
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                  # Git ignore rules
```

## 📊 Parameters Guide

### ICL Test Parameters

| Parameter | Description | Range | Example |
|-----------|-------------|-------|---------|
| **d** | Dataset size | 1-1000 | 100 |
| **N** | Number of in-context examples | 1-50 | 5 |
| **R** | Rank parameter | 1-100 | 10 |
| **flip_prob** | Label flip probability (noise) | 0.0-1.0 | 0.1 |

## 💾 Requirements

Key dependencies:
- `streamlit>=1.28.0` - Web framework
- `torch>=2.0.0` - Model training
- `google-genai>=0.1.0` - Google Gemini API
- `anthropic>=0.7.0` - Anthropic Claude API
- `openai>=1.0.0` - OpenAI GPT API
- `numpy`, `pandas`, `matplotlib`, `seaborn` - Data processing and visualization

See `requirements.txt` for full dependency list.

## 🔑 API Keys

You'll need API keys for the LLM providers you want to test:

1. **Google Gemini**: [Get API Key](https://ai.google.dev/)
2. **Anthropic Claude**: [Get API Key](https://console.anthropic.com/)
3. **OpenAI GPT**: [Get API Key](https://platform.openai.com/api-keys)

## 📈 Results Output

Test results are stored in `src/icl_commerical_model_testing/results/` as JSONL files containing:
- Test configuration parameters
- Model accuracy metrics
- Timestamp and provider information
- Error rates and additional diagnostics

## 🚀 Deployment

### GitHub Pages (Static Site)
The landing page is automatically deployed to GitHub Pages at:
`https://lbangayan.github.io/DSC180B-Website/`

### Streamlit Cloud (Interactive App)
The Streamlit app is deployed at:
`https://dsc180b-website.streamlit.app`

To redeploy after code changes:
```bash
git add .
git commit -m "Your commit message"
git push origin main
```

Both will update automatically.

## 📝 Usage Examples

### Example 1: Single Test
1. Open the app
2. Navigate to "Test with Commercial LLM" tab
3. Select "Single Test Mode"
4. Configure parameters (d=100, N=5, R=10, flip_prob=0.1)
5. Choose LLM provider
6. Click "Run Test"
7. View results and metrics

### Example 2: Batch Testing
1. Navigate to "Batch Testing Mode"
2. Set parameter ranges:
   - d: 50-200
   - N: 2-10
   - R: 5-20
3. Click "Run Batch Tests"
4. View results in dashboard

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is part of DSC180B at UC San Diego.

## 📞 Contact & Support

For issues or questions, please open an issue on GitHub or visit the [landing page](https://lbangayan.github.io/DSC180B-Website/).

---

**Last Updated:** February 2026

Visit the [interactive app](https://dsc180b-website.streamlit.app) to get started! 🎉