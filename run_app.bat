@echo off
REM Fix OpenMP duplicate library issue
set KMP_DUPLICATE_LIB_OK=TRUE

REM Activate conda environment
call conda activate dsc180_fresh

REM Run Streamlit app
streamlit run src\icl_commerical_model_testing\app.py
