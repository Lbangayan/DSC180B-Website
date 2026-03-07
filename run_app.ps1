# Fix OpenMP duplicate library issue
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Activate conda environment
conda activate dsc180_fresh

# Run Streamlit app
streamlit run src/icl_commerical_model_testing/app.py
