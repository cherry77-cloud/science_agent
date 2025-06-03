from workflows.experiment_workflow import ExperimentCodingWorkflow
from agno.run.response import RunEvent
from agno.utils.log import logger, log_info
import json


if __name__ == "__main__":
    # set_log_level_to_debug() # Uncomment for verbose logging
    topic = "novel ensemble method of decision trees"
    lit_review_summary = r"""
Based on the provided papers, the current research on novel ensemble methods of decision trees is exploring diverse avenues beyond traditional bagging and boosting, focusing on addressing specific limitations and enhancing particular properties.

**Current State of Research & Key Findings:**

1.  **Addressing Overfitting and Model Structure:** Research is exploring novel base learners within ensembles. One approach uses *meta-trees* (2402.06386v1), a tree structure designed to prevent single-tree overfitting. Ensembles of these meta-trees, built sequentially with different weighting strategies (GBDT-like, uniform, posterior probability), demonstrate lower risk and prevent overfitting with increasing tree depth, a limitation of standard ensembles. Performance varies with weighting strategy depending on whether the true model structure is likely captured.
2.  **Enhancing Robustness through Novel Training Paradigms:** Coevolutionary algorithms combined with island models and adversarial training are being used to build *robust* ensembles (2412.13762v1). The ICoEvoRDF method, utilizing island-based coevolution of trees and perturbations with Nash equilibrium voting, significantly outperforms state-of-the-art methods in adversarial accuracy and minimax regret, highlighting the importance of diversity and the training framework components.
3.  **Improving Flexibility and Computational Efficiency:** The development of frameworks for training *differentiable (soft)* decision tree ensembles allows for greater flexibility in handling various data types and tasks (multi-task, missing data, zero-inflated distributions) using arbitrary differentiable loss functions (2205.09717v1). A tensor-based formulation enables efficient vectorization and GPU training, leading to significantly faster training times and the ability to produce more compact ensembles with improved out-of-sample performance compared to traditional baselines.
4.  **Understanding Theoretical Limits and Optimality:** Research is delving into the *computational complexity* of finding theoretically optimal decision tree ensembles (minimum total size or minimax individual tree size) that perfectly classify training data (2306.04423v2). Novel algorithms (witness-tree, dynamic programming) provide improved complexity bounds and are shown to be near-optimal under standard assumptions. Critically, theoretical analysis demonstrates that optimal ensembles may require exponentially fewer cuts than single trees for classification, highlighting their potential theoretical compactness.

**Potential Gaps & Areas for Future Work:**

1.  **Integration of Approaches:** The papers present disparate novel methods (meta-trees, coevolutionary robustness, differentiable training, complexity analysis). Future work could explore combining these ideas, e.g., building robust differentiable ensembles of novel tree types or using complexity insights to guide practical ensemble construction.
2.  **Scalability and Practical Implementation:** While some papers discuss efficiency (GPU training, parallelization), the practical scalability and ease of implementation of these novel complex algorithms compared to highly optimized standard libraries remain areas for further investigation and development for widespread adoption.
3.  **Interpretability:** As ensemble methods become more complex (meta-trees, soft trees, coevolutionary), evaluating and potentially enhancing their interpretability becomes increasingly important, a factor not deeply explored in these papers.
4.  **Comprehensive Benchmarking:** While each paper provides strong evidence against relevant baselines, a broader comparison across these *novel* methods on a standardized suite of datasets and tasks (including robustness, handling complex data, efficiency, and predictive performance) is needed to fully understand their relative strengths and weaknesses.
5.  **Bridging Theory and Practice:** The theoretical work (2306.04423v2) provides insights into optimal ensemble structures but doesn't directly translate into practical training algorithms for real-world data with noise and non-perfect classification goals. Future work could bridge this gap, using theoretical findings to inform practical algorithm design for learning near-optimal ensembles.
"""
    research_plan = r"""
Research Plan: Simultaneously Trained Soft  
     Stump Ensemble with Complexity Regularization                             
                                                                               
     **Topic:** Novel Ensemble Methods of Decision Trees                       
                                                                               
     **Motivation & Literature Integration:**                                  
     This research is motivated by the gap identified in the literature review 
     between theoretical work on ensemble complexity (2306.04423v2), which     
     suggests the potential for very compact ensembles, and practical ensemble 
     learning algorithms. It leverages the concept of differentiable trees     
     (2205.09717v1) to propose a novel training paradigm. The plan aims to     
     build a simultaneously trained ensemble of simple, differentiable base    
     learners ("soft stumps"), explicitly optimizing for both predictive       
     performance and ensemble compactness. This differs from traditional       
     sequential (boosting) or independent (bagging) methods and incorporates a 
     direct regularization inspired by theoretical minimum size goals.         
                                                                               
     **Research Question:**                                                    
     Can a simultaneously trained ensemble of simple, "soft" decision stumps,  
     regularized for minimal total complexity (specifically, the number of     
     stumps), achieve competitive predictive performance compared to           
     traditional stump-based ensembles (like AdaBoost-Stumps) while being      
     significantly more compact?                                               
                                                                               
     **Methodology:**                                                          
     Computational modeling and empirical evaluation. We will design,          
     implement, and train the proposed model and evaluate its performance and  
     compactness on standard benchmark datasets, comparing it against relevant 
     baselines.                                                                
                                                                               
     **Proposed Model (Simultaneously Trained Soft Stump Ensemble - STSS):**   
     The ensemble `E(x)` is a sum of `K` soft decision stumps, `E(x) =         
     sum_{k=1}^K S_k(x)`.                                                      
     Each soft decision stump `S_k(x)` incorporates soft feature selection. For
     a given input `x` (with `D` features):                                    
     1.  Learnable parameters for stump `k`: Feature logits `alpha_k` (vector  
     of size `D`), threshold `t_k`, steepness `s_k`, and two leaf values       
     `l_k_left`, `l_k_right`.                                                  
     2.  Calculate feature probabilities: `p_k = softmax(alpha_k)`.            
     3.  Calculate soft split probabilities for each feature:                  
     `prob_left_on_feature_i = sigmoid(s_k * (t_k - x[i]))` for `i = 1...D`.   
     4.  Calculate the overall soft probability of going left: `prob_left_k =  
     sum_{i=1}^D p_k[i] * prob_left_on_feature_i`.                             
     5.  Stump Output: `S_k(x) = prob_left_k * l_k_left + (1 - prob_left_k) *  
     l_k_right`.                                                               
                                                                               
     **Training Objective:**                                                   
     Minimize `Loss = PredictiveLoss(E(X), Y) + Lambda * K`, where             
     `PredictiveLoss` is the binary cross-entropy (for binary classification), 
     `K` is the fixed number of stumps, and `Lambda` is the regularization     
     hyperparameter controlling the trade-off between predictive performance   
     and ensemble size. Parameters (`alpha_k`, `t_k`, `s_k`, `l_k_left`,       
     `l_k_right` for all `k`) are learned using gradient descent.              
                                                                               
     **Innovation:**                                                           
     - Simultaneous, end-to-end gradient-based training of the entire ensemble 
     structure.                                                                
     - Use of differentiable (soft) decision stumps with learned soft feature  
     attention.                                                                
     - Direct optimization for ensemble compactness via a regularization term, 
     inspired by theoretical results, integrated into the differentiable       
     learning process.                                                         
                                                                               
     **Data Requirements:**                                                    
     Standard small-to-medium size numerical datasets suitable for binary      
     classification. Datasets should be sourced from public repositories like  
     the UCI Machine Learning Repository. Examples include Breast Cancer       
     Wisconsin, Pima Indians Diabetes, etc. Data size should be manageable for 
     training within the specified time constraints on standard hardware.      
                                                                               
     **Experimental Setup (Simple Feasibility Study):**                        
     1.  **Implementation:** Implement the STSS model and training loop in a   
     deep learning framework like PyTorch or TensorFlow.                       
     2.  **Datasets:** Select 2-3 standard binary classification datasets      
     (e.g., from UCI).                                                         
     3.  **Data Splitting:** Use a simple train/validation/test split (e.g.,   
     70/15/15) or k-fold cross-validation (e.g., 5-fold) for evaluation        
     robustness.                                                               
     4.  **Training:** Train the STSS model for various fixed values of `K`    
     (e.g., 10, 20, 50, 100) and varying values of the regularization parameter
     `Lambda` (e.g., 0.0, 0.001, 0.01, 0.1). Use an optimization algorithm like
     Adam. Train until convergence or for a fixed number of epochs.            
     5.  **Baselines:** Train and evaluate standard ensemble methods:          
         -   AdaBoost with DecisionStump base estimators (from scikit-learn).  
         -   Random Forest with `max_depth=1` (stumps) and potentially         
     `max_depth=2` or `3` (shallow trees) (from scikit-learn).                 
         -   (Optional sanity check) Logistic Regression or a small MLP.       
     6.  **Evaluation:** Evaluate all models on the held-out test set using    
     standard metrics:                                                         
         -   Predictive Performance: Classification Accuracy, F1-score.        
         -   Compactness: The primary measure will be the number of stumps     
     (`K`) for STSS vs. the number of base estimators for baselines. Also,     
     compare the total number of learned parameters (or a proxy like total     
     number of splits/nodes) if meaningful across models.                      
     7.  **Constraints:** Ensure that training and evaluation for any single   
     configuration (dataset, model, hyperparameters) complete within 30 minutes
     on a standard personal computer (CPU/GPU, ≤16GB RAM). This may require    
     selecting smaller datasets or limiting epoch count for the initial        
     feasibility test.                                                         
                                                                               
     **Expected Outcome:**                                                     
     Demonstrate that the STSS can achieve competitive predictive performance  
     compared to stump-based baselines while potentially achieving higher      
     compactness (fewer stumps) for a given level of accuracy, especially for  
     non-zero `Lambda`. """
    code = r"""
import pandas as pd                                                       
import numpy as np                                                        
                                                                       
# Load the dataset using the exact absolute path                          
df = pd.read_csv(r'C:\data\adult.csv')                                    
                                                                       
# Replace non-standard missing values ('?') with NaN                      
df.replace('?', np.nan, inplace=True)                                     
                                                                       
# Identify columns that now have missing values (NaNs)                    
missing_values_per_column = df.isnull().sum()                             
columns_with_missing = missing_values_per_column[missing_values_per_column> 0].index.tolist()                                                       
                                                                       
# Impute missing values for identified categorical columns using the mode 
for col in columns_with_missing:                                          
    # Ensure it's a categorical/object type before mode imputation        
    if df[col].dtype == 'object':                                         
        # Calculate the mode, taking the first if multiple modes exist    
        mode_values = df[col].mode()                                      
        if not mode_values.empty:                                         
            mode_value = mode_values[0]                                   
            df[col].fillna(mode_value, inplace=True)                      
                                                                       
# The dataframe 'df' is now preprocessed with missing values handled.                                      
    """
    summary = """                                        
The analysis of the dataset 'adult.csv' involved initial data loading and 
     inspection, handling of missing values, and basic exploratory data        
     analysis.                                                                 
                                                                               
     The dataset contains 32,561 entries across 15 columns, including both     
     numerical (int64) and categorical (object) data types. Initial inspection 
     revealed that standard null checks (`isnull().sum()`) did not identify    
     missing values, but visual inspection of the head suggested '?' characters
     were used as non-standard missing value indicators.                       
                                                                               
     A targeted check confirmed that '?' characters were present in the        
     'workclass' (1836), 'occupation' (1843), and 'native.country' (583)       
     columns. These '?' values were successfully replaced with standard NaN    
     values.                                                                   
                                                                               
     Missing values in these three categorical columns were then imputed using 
     the mode of each respective column ('Private' for 'workclass',            
     'Prof-specialty' for 'occupation', and 'United-States' for                
     'native.country'). A final check verified that no missing values remained 
     in the dataset after this imputation step.                                
                                                                               
     Exploratory data analysis included visualizing the distribution of key    
     features. A histogram of 'age' showed a distribution concentrated in      
     younger to middle-aged adults, with a long tail towards older ages. The   
     distribution of the target variable 'income' (<=50K and >50K) was         
     visualized with a count plot, clearly indicating an imbalance with        
     significantly more individuals in the '<=50K' category. Finally, the      
     relationship between 'hours.per.week' and 'income' was explored using a   
     violin plot and descriptive statistics, revealing that individuals earning
     '>50K' generally report a higher mean and median number of hours worked   
     per week compared to those earning '<=50K', though both groups show a wide
     range of hours.  
"""
    dataset_path = "C\\data\\adult.csv"
    data_preparation_result = [
        {
            "dataset_path": dataset_path,
            "comprehensive_analysis_summary": summary,
            "consolidated_preprocessing_code_for_dataset": code,
        }
    ]

    workflow = ExperimentCodingWorkflow(
        max_iterations=20,  # Adjust as needed
        min_score_to_accept=9,  # Set a reasonable acceptance threshold
        output_dir_name=r"C:/project/Agentlab_demo",
    )

    log_info("Starting experiment coding workflow...\n")

    final_result_payload = None
    for result_chunk in workflow.run(
        topic=topic,
        lit_review_sum=lit_review_summary,
        plan=research_plan,
        data_preparation_result=data_preparation_result,
        stream_intermediate_steps=True,  # See progress
    ):
        print(result_chunk.content, end="", flush=True)
        if (
            result_chunk.event == RunEvent.run_completed
            or result_chunk.event == RunEvent.run_error
        ):
            final_result_payload = result_chunk  # Capture the last one
            print("\n--- Experiment Workflow Finished ---")

    if final_result_payload:
        log_info(f"\nFinal Workflow Status: {final_result_payload.event}")
        log_info(f"Final Workflow Message: {final_result_payload.content}")
        if final_result_payload.extra_data and final_result_payload.extra_data.get(
            "experiment_code"
        ):
            log_info("\n--- Best Experiment Code ---")
            print(final_result_payload.extra_data["experiment_code"])
            log_info("\n--- Best Execution Result ---")
            print(
                json.dumps(
                    final_result_payload.extra_data["execution_result"], indent=2
                )
            )
            log_info(f"\nFinal Score: {final_result_payload.extra_data['final_score']}")
            log_info(
                f"Code saved at: {final_result_payload.extra_data['final_code_path']}"
            )
