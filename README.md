\# DS 785 Capstone - Code Repository

This repository contains all Python codes used for the data acquisition, preprocessing, exploratory data analysis, feature engineering, model development, and evaluation for my DS785 Capstone project.

The codes are listed below in the order by which they are run

\## Repository Structure

\### Data Download and Preprocessing Scripts

download_data_from_centers.py - Download original/raw data from the external Data Centers

make_SMOS_SSS_DataMatrix.py - Construct SMOS SSS matrix, compute seasonal anomalies and SSS standard deviations

determine_SSS_variability_regions.py - Identify the 3 regions of SSS variability used in the final analysis

extract_regions_data.py - Extract region-specific subsets of all datasets

merge_regional_extracted_data.py - Merges all canonical datasets for each variability region

\### Feature Engineering + EDA

Perform_EDA_FeatureEngineering.py - Conduct EDA and Feature Engineering for the 3 regions

Make_EDA_FinalFigures.py - Produce final statistics tables and presentation figures from the EDA

\### Modeling

Build_Models_4x3_Folds.py - Trains and evaluates models using 4×3 CV folds during initial model exploration

Build_Models.py - Trains and evaluates the final models using the finalized 6×4 CV framework

Make_Model_Building_and_Evaluation_FinalFigures.py - Generates all figures and statistics used in the final Capstone report related to model performance

\### Supporting Files

low_variables.csv

med_variables.csv

high_variables.csv

\### Figures and Model Output

general_figs/

EDA_FE_outputs/

EDA_FE_figs_output/

model_sss_region_outputs_4x3Folds

model_sss_region_outputs/

model_figs_output/
