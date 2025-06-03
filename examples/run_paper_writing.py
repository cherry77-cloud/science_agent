from agno.tools.arxiv import ArxivTools
from agno.utils.log import logger
from agno.run.response import RunEvent
from workflows.paper_writing_workflow import PaperWritingWorkflow
import json


if __name__ == "__main__":
    # Optional: Enable debug logging for detailed agent/tool interactions

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
    # arxivTools = ArxivTools()
    # read_results_json_str = arxivTools.read_arxiv_papers(
    #     id_list=['2402.06386', '2412.13762', '2205.09717', '2306.04423'],
    #     pages_to_read=None)
    # read_results = json.loads(read_results_json_str)
    # full_text = ""
    # if read_results and isinstance(read_results, list) and len(read_results) > 0:
    #     for i in range(4):
    #         article_data = read_results[i]
    #         if 'content' in article_data and isinstance(article_data['content'], list):
    #             full_text += "\n".join(
    #                 [page_content['text'] for page_content in article_data['content'] if
    #                  'text' in page_content and page_content['text']])

    output_file = "full_text.txt"
    with open(output_file, "r", encoding="utf-8") as f:
        full_text = f.read()

    lit_review_paper = f"""
    Cited Papers:{full_text}
    """

    plan = r"""
    Research Plan: Simultaneously Trained Soft Stump Ensemble with Complexity Regularization                             

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

    exp_code = r"""
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
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # dataset_code is prepended here.
    # It is expected to load 'adult.csv', handle '?' values, impute missing values with mode,
    # and provide the preprocessed dataframe 'df'.

    # --- Further preprocessing: One-Hot Encode categorical features ---
    # Identify categorical columns (object or category dtype)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Handle the target variable 'income' first: Map '<=50K' to 0, '>50K' to 1
    le = LabelEncoder()
    df['income'] = le.fit_transform(df['income'])

    # Remove income from categorical columns list before OHE if it was included
    categorical_cols_list = categorical_cols.tolist()
    if 'income' in categorical_cols_list:
        categorical_cols_list.remove('income')
    categorical_cols = pd.Index(categorical_cols_list)

    # Apply One-Hot Encoding to the remaining categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Separate features (X) and target (y)
    X = df_encoded.drop('income', axis=1)
    y = df_encoded['income']

    # Select only numeric columns to avoid TypeError when converting to tensor
    # This is important after one-hot encoding which creates numeric columns
    X = X.select_dtypes(include=np.number)

    print('Data preprocessing complete.')
    print('Features shape after selecting numeric types: {}'.format(X.shape))
    print('Target shape: {}'.format(y.shape))

    # --- Data Splitting ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print('Data split into train ({} samples) and test ({} samples).'.format(X_train.shape[0], X_test.shape[0]))

    # Convert data to PyTorch tensors
    # Check if GPU is available and use it, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))

    # Ensure y tensors are float32 and have shape [batch_size, 1] for BCEWithLogitsLoss
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)

    # Input dimension
    D_in = X_train.shape[1]
    print('Input dimension (D_in): {}'.format(D_in))

    # --- Define the STSS Model ---
    # This is a necessary exception for using PyTorch effectively.
    class SoftStump(nn.Module):
        def __init__(self, D_in):
            super(SoftStump, self).__init__()
            self.alpha = nn.Parameter(torch.randn(D_in))
            self.t = nn.Parameter(torch.randn(1))
            self.s = nn.Parameter(torch.rand(1) + 0.1) # Initialize steepness to be positive and not too close to zero
            self.l_left = nn.Parameter(torch.randn(1))
            self.l_right = nn.Parameter(torch.randn(1))

        def forward(self, x):
            p_k = torch.softmax(self.alpha, dim=0)
            prob_left_on_features = torch.sigmoid(self.s * (self.t - x))
            prob_left_k = torch.sum(p_k.unsqueeze(0) * prob_left_on_features, dim=1, keepdim=True)
            stump_output = prob_left_k * self.l_left + (1.0 - prob_left_k) * self.l_right
            return stump_output

    class STSS(nn.Module):
        def __init__(self, D_in, K):
            super(STSS, self).__init__()
            self.K = K
            self.stumps = nn.ModuleList([SoftStump(D_in) for _ in range(K)])

        def forward(self, x):
            stump_outputs = [stump(x) for stump in self.stumps]
            ensemble_output = torch.sum(torch.stack(stump_outputs, dim=0), dim=0)
            return ensemble_output # shape [batch_size, 1]

    # --- Hyperparameters for Experiment Sweep ---
    K_values = [10, 20, 50, 100]
    Lambda_values = [0.0, 0.001, 0.01, 0.1]
    learning_rate = 0.01
    eepochs = 200 # Number of epochs

    all_results = [] # List to store results from all runs

    # --- STSS Experiment Sweep ---
    print('Starting STSS experiment sweep...')
    for K_value in K_values:
        for Lambda_value in Lambda_values:
            print('Training STSS with K={}, Lambda={}...'.format(K_value, Lambda_value))

            stss_model = STSS(D_in, K_value).to(device)
            criterion = nn.BCEWithLogitsLoss() # Combines Sigmoid and BCELoss
            optimizer = optim.Adam(stss_model.parameters(), lr=learning_rate)

            # Training loop
            # FIX: Changed 'epochs' to 'eepochs'
            for epoch in range(eepochs):
                stss_model.train()
                outputs = stss_model(X_train_tensor)
                predictive_loss = criterion(outputs, y_train_tensor)
                total_loss = predictive_loss + Lambda_value * K_value # K_value is constant per run

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Optional: Print loss periodically
                # if (epoch+1) % 100 == 0:
                #     print('  Epoch [{}/{}], Loss: {:.4f}, Pred Loss: {:.4f}'.format(epoch+1, eepochs, total_loss.item(), predictive_loss.item()))

            # Evaluation
            stss_model.eval()
            with torch.no_grad():
                test_outputs = stss_model(X_test_tensor)
                # BCEWithLogitsLoss doesn't output probabilities, apply sigmoid for prediction
                test_probs = torch.sigmoid(test_outputs)
                test_preds = (test_probs > 0.5).long() # Threshold at 0.5 for binary prediction

                y_test_np = y_test_tensor.cpu().numpy()
                test_preds_np = test_preds.cpu().numpy()

                stss_accuracy = accuracy_score(y_test_np, test_preds_np)
                stss_f1 = f1_score(y_test_np, test_preds_np)

                print('  STSS (K={}, Lambda={}) Results - Accuracy: {:.4f}, F1 Score: {:.4f}'.format(K_value, Lambda_value, stss_accuracy, stss_f1))

                all_results.append({
                    'Model Type': 'STSS',
                    'Model Config': 'K={}, Lambda={}'.format(K_value, Lambda_value),
                    'K': K_value,
                    'Lambda': Lambda_value,
                    'Accuracy': stss_accuracy,
                    'F1 Score': stss_f1,
                    'Estimators': K_value # For comparability, STSS 'complexity' is K
                })

    print('STSS experiment sweep finished.')

    # --- Baseline Models Experiment Sweep (Matching K values) ---
    print('Starting baseline models experiment sweep...')

    # Use the same K_values for baseline estimators for direct comparison of complexity
    baseline_n_estimators = K_values

    # AdaBoost with Decision Stumps (max_depth=1)
    print('Training AdaBoost with Stumps for various n_estimators...')
    for n_est in baseline_n_estimators:
        print('  Training AdaBoost with n_estimators={}...'.format(n_est))
        # Address FutureWarning by specifying algorithm='SAMME'
        adaboost_stumps = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=n_est,
            random_state=42,
            algorithm='SAMME' # Use SAMME algorithm
        )
        adaboost_stumps.fit(X_train, y_train)

        # Evaluate AdaBoost
        adaboost_preds = adaboost_stumps.predict(X_test)
        adaboost_accuracy = accuracy_score(y_test, adaboost_preds)
        adaboost_f1 = f1_score(y_test, adaboost_preds)

        print('  AdaBoost (n={}) Results - Accuracy: {:.4f}, F1 Score: {:.4f}'.format(n_est, adaboost_accuracy, adaboost_f1))

        all_results.append({
            'Model Type': 'AdaBoost',
            'Model Config': 'n={}'.format(n_est),
            'K': None, # N/A for baselines
            'Lambda': None, # N/A for baselines
            'Accuracy': adaboost_accuracy,
            'F1 Score': adaboost_f1,
            'Estimators': n_est
        })

    # Random Forest with Decision Stumps (max_depth=1)
    print('Training Random Forest with Stumps for various n_estimators...')
    for n_est in baseline_n_estimators:
        print('  Training Random Forest with n_estimators={}...'.format(n_est))
        rf_stumps = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=1, # Stumps
            random_state=42,
            n_jobs=-1 # Use all available cores
        )
        rf_stumps.fit(X_train, y_train)

        # Evaluate Random Forest
        rf_preds = rf_stumps.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_preds)
        rf_f1 = f1_score(y_test, rf_preds)

        print('  Random Forest (n={}, md=1) Results - Accuracy: {:.4f}, F1 Score: {:.4f}'.format(n_est, rf_accuracy, rf_f1))

        all_results.append({
            'Model Type': 'Random Forest',
            'Model Config': 'n={}'.format(n_est),
            'K': None, # N/A
            'Lambda': None, # N/A
            'Accuracy': rf_accuracy,
            'F1 Score': rf_f1,
            'Estimators': n_est
        })

    print('Baseline models experiment sweep finished.')

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # --- Print Summary Table ---
    print('\n--- Comprehensive Experiment Results Summary ---')
    print(results_df.to_string())
    print('------------------------------------------------')


    # --- Figure Generation ---
    save_loc_for_figures = r'C:\project\Agentlab_demo\tmp_figs/'
    os.makedirs(save_loc_for_figures, exist_ok=True)
    print('Saving figures to: {}'.format(save_loc_for_figures))

    # Figure 1: Accuracy vs. Number of Estimators/Stumps
    fig1, ax1 = plt.subplots(figsize=(10, 7))

    # Plot STSS results (different Lambdas as hue)
    stss_df = results_df[results_df['Model Type'] == 'STSS'].copy()
    stss_df['Lambda Str'] = stss_df['Lambda'].astype(str) # Convert Lambda to string for plotting

    sns.lineplot(data=stss_df, x='Estimators', y='Accuracy', hue='Lambda Str', marker='o', ax=ax1)

    # Plot Baseline results (different Model Type as hue, line for each)
    baseline_df = results_df[results_df['Model Type'].isin(['AdaBoost', 'Random Forest'])].copy()
    # Group baselines by Model Type and plot each as a separate line
    for model_type in baseline_df['Model Type'].unique():
        model_df = baseline_df[baseline_df['Model Type'] == model_type]
        sns.lineplot(data=model_df, x='Estimators', y='Accuracy', label=model_type, marker='X', linestyle='--', ax=ax1)

    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Number of Stumps/Estimators')
    ax1.set_title('') # No in-figure title
    ax1.grid(True, linestyle=':')
    ax1.legend(title='Model / STSS Lambda')

    plt.tight_layout()
    figure1_path = os.path.join(save_loc_for_figures, 'Accuracy_vs_Complexity_Adult.pdf')
    plt.savefig(figure1_path, dpi=300, bbox_inches='tight')
    print('Figure saved to: {}'.format(figure1_path))
    plt.close(fig1)

    # Figure 2: F1 Score vs. Number of Estimators/Stumps
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    # Plot STSS results
    sns.lineplot(data=stss_df, x='Estimators', y='F1 Score', hue='Lambda Str', marker='o', ax=ax2)

    # Plot Baseline results
    for model_type in baseline_df['Model Type'].unique():
        model_df = baseline_df[baseline_df['Model Type'] == model_type]
        sns.lineplot(data=model_df, x='Estimators', y='F1 Score', label=model_type, marker='X', linestyle='--', ax=ax2)

    ax2.set_ylabel('F1 Score')
    ax2.set_xlabel('Number of Stumps/Estimators')
    ax2.set_title('') # No in-figure title
    ax2.grid(True, linestyle=':')
    ax2.legend(title='Model / STSS Lambda')

    plt.tight_layout()
    figure2_path = os.path.join(save_loc_for_figures, 'F1_vs_Complexity_Adult.pdf')
    plt.savefig(figure2_path, dpi=300, bbox_inches='tight')
    print('Figure saved to: {}'.format(figure2_path))
    plt.close(fig2)

    print('Experiment complete.')
    """

    exp_result = r"""
    Experiment coding workflow completed. Best result 
         (from Iteration 10):                                                      
         Score: 8/10                                                               
         Code saved at:                                                            
         C:\project\AgentLab_demo\best_code\best_experiment_code_iter_10_score_8.00
         .py                                                                       
         Execution Result saved at:                                                
         C:\project\AgentLab_demo\best_code\best_execution_result_iter_10_score_8.0
         0.json                                                                    
         Figures saved in directory: C:\project\AgentLab_demo\figures              
         Focused Task: Execute the STSS and baseline model training and evaluation 
         loops (Step 4 & 5 from plan) and generate comparison figures (Step 6 & 7  
         from plan).                                                               
         Feedback: The code demonstrates very good progress towards completing the 
         'Simple Feasibility Study' outlined in the plan. It successfully          
         implements the core STSS model, the specified training objective with     
         Lambda regularization, includes the required AdaBoost and Random Forest   
         stump baselines, performs sweeps over K and Lambda values, evaluates      
         performance using Accuracy and F1-score, collects results, and generates  
         the necessary comparison figures (Performance vs. Complexity) saving them 
         to the specified temporary directory. This covers a substantial portion of
         the planned experimental setup.                                           

         The execution ran successfully end-to-end. The results table provides a   
         clear comparison across models and hyperparameters. Figures were correctly
         generated and saved. The Reflection Agent provided excellent analysis,    
         accurately highlighting the key successes (implementation, baselines,     
         figures) and identifying the significant issue of 0 F1 scores for some    
         STSS configurations, which indicates instability or tuning problems for   
         those specific runs. The suggestions are highly relevant and actionable.  

         Minor points: There is a `FutureWarning` in stderr related to pandas      
         inplace operations, which should be addressed for code cleanliness. The   
         STSS model's performance is currently significantly below AdaBoost, and   
         the 0 F1 issue requires investigation. However, the overall structure and 
         execution of the experiment are solid.
    Execution_result:
    {
      "stdout": "Data preprocessing complete.\nFeatures shape after selecting numeric types: (32561, 6)\nTarget shape: (32561,)\nData split into train (22792 samples) and test (9769 samples).\nUsing device: cpu\nInput dimension (D_in): 6\nStarting STSS experiment sweep...\nTraining STSS with K=10, Lambda=0.0...\n  STSS (K=10, Lambda=0.0) Results - Accuracy: 0.7838, F1 Score: 0.3094\nTraining STSS with K=10, Lambda=0.001...\n  STSS (K=10, Lambda=0.001) Results - Accuracy: 0.7592, F1 Score: 0.0000\nTraining STSS with K=10, Lambda=0.01...\n  STSS (K=10, Lambda=0.01) Results - Accuracy: 0.7935, F1 Score: 0.3819\nTraining STSS with K=10, Lambda=0.1...\n  STSS (K=10, Lambda=0.1) Results - Accuracy: 0.7843, F1 Score: 0.3035\nTraining STSS with K=20, Lambda=0.0...\n  STSS (K=20, Lambda=0.0) Results - Accuracy: 0.7955, F1 Score: 0.3810\nTraining STSS with K=20, Lambda=0.001...\n  STSS (K=20, Lambda=0.001) Results - Accuracy: 0.7592, F1 Score: 0.0000\nTraining STSS with K=20, Lambda=0.01...\n  STSS (K=20, Lambda=0.01) Results - Accuracy: 0.8057, F1 Score: 0.4583\nTraining STSS with K=20, Lambda=0.1...\n  STSS (K=20, Lambda=0.1) Results - Accuracy: 0.7926, F1 Score: 0.3409\nTraining STSS with K=50, Lambda=0.0...\n  STSS (K=50, Lambda=0.0) Results - Accuracy: 0.7980, F1 Score: 0.3897\nTraining STSS with K=50, Lambda=0.001...\n  STSS (K=50, Lambda=0.001) Results - Accuracy: 0.8064, F1 Score: 0.4854\nTraining STSS with K=50, Lambda=0.01...\n  STSS (K=50, Lambda=0.01) Results - Accuracy: 0.8061, F1 Score: 0.4545\nTraining STSS with K=50, Lambda=0.1...\n  STSS (K=50, Lambda=0.1) Results - Accuracy: 0.7939, F1 Score: 0.4011\nTraining STSS with K=100, Lambda=0.0...\n  STSS (K=100, Lambda=0.0) Results - Accuracy: 0.7947, F1 Score: 0.3805\nTraining STSS with K=100, Lambda=0.001...\n  STSS (K=100, Lambda=0.001) Results - Accuracy: 0.7936, F1 Score: 0.3755\nTraining STSS with K=100, Lambda=0.01...\n  STSS (K=100, Lambda=0.01) Results - Accuracy: 0.7982, F1 Score: 0.3967\nTraining STSS with K=100, Lambda=0.1...\n  STSS (K=100, Lambda=0.1) Results - Accuracy: 0.7989, F1 Score: 0.4026\nSTSS experiment sweep finished.\nStarting baseline models experiment sweep...\nTraining AdaBoost with Stumps for various n_estimators...\n  Training AdaBoost with n_estimators=10...\n  AdaBoost (n=10) Results - Accuracy: 0.8209, F1 Score: 0.5839\n  Training AdaBoost with n_estimators=20...\n  AdaBoost (n=20) Results - Accuracy: 0.8213, F1 Score: 0.5851\n  Training AdaBoost with n_estimators=50...\n  AdaBoost (n=50) Results - Accuracy: 0.8251, F1 Score: 0.5410\n  Training AdaBoost with n_estimators=100...\n  AdaBoost (n=100) Results - Accuracy: 0.8261, F1 Score: 0.5341\nTraining Random Forest with Stumps for various n_estimators...\n  Training Random Forest with n_estimators=10...\n  Random Forest (n=10, md=1) Results - Accuracy: 0.8000, F1 Score: 0.3006\n  Training Random Forest with n_estimators=20...\n  Random Forest (n=20, md=1) Results - Accuracy: 0.7851, F1 Score: 0.2010\n  Training Random Forest with n_estimators=50...\n  Random Forest (n=50, md=1) Results - Accuracy: 0.7912, F1 Score: 0.2433\n  Training Random Forest with n_estimators=100...\n  Random Forest (n=100, md=1) Results - Accuracy: 0.7905, F1 Score: 0.2382\nBaseline models experiment sweep finished.\n\n--- Comprehensive Experiment Results Summary ---\n       Model Type         Model Config      K  Lambda  Accuracy  F1 Score  Estimators\n0            STSS     K=10, Lambda=0.0   10.0   0.000  0.783806  0.309353          10\n1            STSS   K=10, Lambda=0.001   10.0   0.001  0.759238  0.000000          10\n2            STSS    K=10, Lambda=0.01   10.0   0.010  0.793531  0.381857          10\n3            STSS     K=10, Lambda=0.1   10.0   0.100  0.784318  0.303471          10\n4            STSS     K=20, Lambda=0.0   20.0   0.000  0.795475  0.381041          20\n5            STSS   K=20, Lambda=0.001   20.0   0.001  0.759238  0.000000          20\n6            STSS    K=20, Lambda=0.01   20.0   0.010  0.805712  0.458333          20\n7            STSS     K=20, Lambda=0.1   20.0   0.100  0.792609  0.340924          20\n8            STSS     K=50, Lambda=0.0   50.0   0.000  0.798035  0.389731          50\n9            STSS   K=50, Lambda=0.001   50.0   0.001  0.806428  0.485442          50\n10           STSS    K=50, Lambda=0.01   50.0   0.010  0.806121  0.454493          50\n11           STSS     K=50, Lambda=0.1   50.0   0.100  0.793940  0.401071          50\n12           STSS    K=100, Lambda=0.0  100.0   0.000  0.794657  0.380482         100\n13           STSS  K=100, Lambda=0.001  100.0   0.001  0.793633  0.375465         100\n14           STSS   K=100, Lambda=0.01  100.0   0.010  0.798239  0.396694         100\n15           STSS    K=100, Lambda=0.1  100.0   0.100  0.798854  0.402554         100\n16       AdaBoost                 n=10    NaN     NaN  0.820862  0.583928          10\n17       AdaBoost                 n=20    NaN     NaN  0.821271  0.585076          20\n18       AdaBoost                 n=50    NaN     NaN  0.825059  0.540962          50\n19       AdaBoost                n=100    NaN     NaN  0.826083  0.534138         100\n20  Random Forest                 n=10    NaN     NaN  0.799980  0.300644          10\n21  Random Forest                 n=20    NaN     NaN  0.785137  0.200990          20\n22  Random Forest                 n=50    NaN     NaN  0.791176  0.243323          50\n23  Random Forest                n=100    NaN     NaN  0.790460  0.238184         100\n------------------------------------------------\nSaving figures to: C:\\project\\Agentlab_demo\\tmp_figs/\nFigure saved to: C:\\project\\Agentlab_demo\\tmp_figs/Accuracy_vs_Complexity_Adult.pdf\nFigure saved to: C:\\project\\Agentlab_demo\\tmp_figs/F1_vs_Complexity_Adult.pdf\nExperiment complete.\n\u001b[0m",
      "stderr": "C:\\Users\\13354\\AppData\\Local\\Temp\\tmpzcy68357\\experiment_script.py:22: FutureWarning:\n\nA value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\nThe behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n\nFor example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object."
    }
    """

    res_interpretation = """
    ### **Interpretation of Experimental Results: A Feasibility Study on Simultaneously Trained Soft Stump Ensembles with Complexity Regularization**

    **1. Introduction and Experimental Objectives**

    This study investigates a novel ensemble learning paradigm: the Simultaneously Trained Soft Stump Ensemble (STSS), which integrates differentiable decision trees with explicit complexity regularization. Motivated by theoretical work suggesting the potential for significantly more compact ensembles than typically achieved by standard methods (as highlighted by 2306.04423v2), this research aims to bridge the gap between theoretical compactness and practical ensemble learning algorithms. Leveraging the flexibility of differentiable trees (2205.09717v1), STSS proposes a direct optimization for both predictive performance and ensemble compactness (measured by the number of stumps, `K`).

    The primary research question addressed by this preliminary experiment was: Can a simultaneously trained ensemble of simple, "soft" decision stumps, regularized for minimal total complexity, achieve competitive predictive performance compared to traditional stump-based ensembles (like AdaBoost-Stumps and Random Forest-Stumps) while being significantly more compact?

    **2. Experimental Setup Overview**

    *   **Proposed Model (STSS):** The STSS model is a sum of `K` differentiable soft decision stumps. Each stump features learnable parameters for soft feature selection (`alpha_k` via softmax), a threshold (`t_k`), steepness (`s_k`), and two leaf values (`l_k_left`, `l_k_right`). The training objective minimizes `PredictiveLoss (Binary Cross-Entropy) + Lambda * K`, where `Lambda` is a hyperparameter controlling the trade-off between predictive performance and the fixed number of stumps (`K`). The entire ensemble is trained end-to-end using gradient descent.
    *   **Baseline Models:** For comparison, two traditional ensemble methods using single-depth decision trees (stumps) as base estimators were trained:
        *   AdaBoostClassifier with `DecisionTreeClassifier(max_depth=1)`.
        *   RandomForestClassifier with `max_depth=1`.
    *   **Dataset:** The Adult Income Dataset was used for binary classification (income `<=50K` vs. `>50K`). Preprocessing involved replacing missing values with the mode and one-hot encoding categorical features, resulting in 6 numerical features for the model input.
    *   **Evaluation Metrics:** Classification Accuracy and F1-score were used to assess predictive performance on the positive class (income `>50K`, which is the minority class). Model complexity was measured by the number of base estimators (`K` for STSS, `n_estimators` for baselines).
    *   **Hyperparameter Sweep:**
        *   `K` / `n_estimators` were varied across {10, 20, 50, 100}.
        *   `Lambda` for STSS was varied across {0.0, 0.001, 0.01, 0.1}.
    *   **Training Details:** Data was split into 70% for training and 30% for testing (stratified). Models were trained for 200 epochs using the Adam optimizer. All computations were performed on a CPU.

    **3. Results Analysis**

    The comprehensive results are summarized below:

    | Model Type    | Model Config      | K    | Lambda | Accuracy | F1 Score | Estimators |
    | :------------ | :---------------- | :--- | :----- | :------- | :------- | :--------- |
    | STSS          | K=10, Lambda=0.0  | 10.0 | 0.000  | 0.7838   | 0.3094   | 10         |
    | STSS          | K=10, Lambda=0.001| 10.0 | 0.001  | 0.7592   | **0.0000** | 10         |
    | STSS          | K=10, Lambda=0.01 | 10.0 | 0.010  | 0.7935   | 0.3819   | 10         |
    | STSS          | K=10, Lambda=0.1  | 10.0 | 0.100  | 0.7843   | 0.3035   | 10         |
    | STSS          | K=20, Lambda=0.0  | 20.0 | 0.000  | 0.7955   | 0.3810   | 20         |
    | STSS          | K=20, Lambda=0.001| 20.0 | 0.001  | 0.7592   | **0.0000** | 20         |
    | STSS          | K=20, Lambda=0.01 | 20.0 | 0.010  | 0.8057   | 0.4583   | 20         |
    | STSS          | K=20, Lambda=0.1  | 20.0 | 0.100  | 0.7926   | 0.3409   | 20         |
    | STSS          | K=50, Lambda=0.0  | 50.0 | 0.000  | 0.7980   | 0.3897   | 50         |
    | STSS          | K=50, Lambda=0.001| 50.0 | 0.001  | **0.8064** | **0.4854** | 50         |
    | STSS          | K=50, Lambda=0.01 | 50.0 | 0.010  | 0.8061   | 0.4545   | 50         |
    | STSS          | K=50, Lambda=0.1  | 50.0 | 0.100  | 0.7939   | 0.4011   | 50         |
    | STSS          | K=100, Lambda=0.0 | 100.0| 0.000  | 0.7947   | 0.3805   | 100        |
    | STSS          | K=100, Lambda=0.001| 100.0| 0.001  | 0.7936   | 0.3755   | 100        |
    | STSS          | K=100, Lambda=0.01| 100.0| 0.010  | 0.7982   | 0.3967   | 100        |
    | STSS          | K=100, Lambda=0.1 | 100.0| 0.100  | 0.7989   | 0.4026   | 100        |
    | AdaBoost      | n=10              | NaN  | NaN    | 0.8209   | 0.5839   | 10         |
    | AdaBoost      | n=20              | NaN  | NaN    | 0.8213   | 0.5851   | 20         |
    | AdaBoost      | n=50              | NaN  | NaN    | 0.8251   | 0.5410   | 50         |
    | AdaBoost      | n=100             | NaN  | NaN    | 0.8261   | 0.5341   | 100        |
    | Random Forest | n=10              | NaN  | NaN    | 0.8000   | 0.3006   | 10         |
    | Random Forest | n=20              | NaN  | NaN    | 0.7851   | 0.2010   | 20         |
    | Random Forest | n=50              | NaN  | NaN    | 0.7912   | 0.2433   | 50         |
    | Random Forest | n=100             | NaN  | NaN    | 0.7905   | 0.2382   | 100        |

    **Key Observations:**

    *   **Overall Performance Comparison:** AdaBoost consistently outperformed both STSS and Random Forest across all tested `n_estimators` values. For instance, AdaBoost achieved accuracies ranging from **0.8209 to 0.8261** and F1-scores from **0.5341 to 0.5851**.
        *   The best performing STSS configuration (`K=50, Lambda=0.001`) yielded an accuracy of **0.8064** and an F1-score of **0.4854**. This is notably lower than AdaBoost's performance (e.g., AdaBoost with `n=10` achieved 0.8209 Accuracy and 0.5839 F1-score).
        *   Random Forest with stumps showed competitive accuracy with STSS (ranging from 0.7851 to 0.8000) but generally lower F1-scores (ranging from 0.2010 to 0.3006), indicating poor performance on the minority class.
    *   **STSS Performance and `Lambda` Regularization:**
        *   **Optimization Instability:** A critical observation is the **0.0000 F1-score** for STSS with `Lambda=0.001` at `K=10` and `K=20`. This indicates that the model completely failed to predict the positive class, likely collapsing to a trivial solution (predicting only the majority class) during optimization. This highlights significant instability or challenges in the optimization landscape for these specific `Lambda` and `K` combinations.
        *   **Impact of `Lambda`:** Excluding the 0 F1-score cases, varying `Lambda` had a mixed impact. For a fixed `K`, `Lambda=0.01` or `Lambda=0.001` (when stable) often led to the highest F1-scores. For example, at `K=50`, `Lambda=0.001` yielded the highest F1-score (0.4854), while `Lambda=0.1` consistently resulted in lower F1-scores compared to smaller `Lambda` values, suggesting that excessive regularization can hinder predictive performance.
        *   **Impact of `K`:** For STSS, increasing `K` from 10 to 50 generally improved performance (e.g., max F1-score jumped from 0.3819 at K=10 to 0.4854 at K=50), but further increasing `K` to 100 did not yield significant additional gains for the best performing Lambda values.
    *   **Compactness vs. Performance Trade-off:** While the STSS model aims for compactness, its current performance is not competitive with AdaBoost. For instance, AdaBoost with only 10 stumps achieved an F1-score of **0.5839**, which is significantly higher than the best STSS F1-score of **0.4854** achieved with 50 stumps. This suggests that the initial implementation of STSS, despite its theoretical motivation for compactness, has not yet demonstrated a superior performance-to-compactness ratio compared to an established boosting method like AdaBoost on this dataset. The goal of achieving *competitive* predictive performance alongside compactness was not fully realized in this initial feasibility study.

    **4. Discussion and Interpretation**

    This preliminary experiment successfully implemented the core concept of a simultaneously trained soft stump ensemble with complexity regularization. The code ran successfully, and the results provide valuable insights into the behavior of this novel paradigm.

    *   **Proof of Concept:** The study serves as a proof of concept for the simultaneous, gradient-based training of an ensemble of differentiable stumps. The model is able to learn and achieve reasonable accuracy and F1-scores (up to 0.8064 and 0.4854, respectively) on the Adult dataset.
    *   **Optimization Challenges:** The occurrence of 0 F1-scores for specific `Lambda` values (e.g., `Lambda=0.001` for `K=10, 20`) is a significant finding. This indicates that the optimization landscape for the STSS model can be challenging, leading to convergence to suboptimal solutions where the model fails to predict the minority class. This phenomenon is likely due to the highly non-convex nature of jointly optimizing all stump parameters across the entire ensemble. Traditional greedy methods like AdaBoost avoid this by sequentially adding trees and focusing on misclassified samples.
    *   **Performance Gap with Baselines:** The substantial performance gap between STSS and AdaBoost (e.g., AdaBoost's F1-score of 0.5839 vs. STSS's best of 0.4854) suggests that the current simultaneous training approach of STSS may lack the adaptive learning mechanisms (like re-weighting misclassified samples) that make boosting methods highly effective, especially for imbalanced datasets. Random Forest, while demonstrating slightly lower F1 scores than STSS, did so with an independent training paradigm.
    *   **Role of Regularization:** The `Lambda * K` regularization term, while theoretically sound for encouraging compactness, needs further investigation. Its current impact seems complex and sometimes detrimental to performance, suggesting that the balance between compactness and predictive power in this novel training scheme is highly sensitive to the `Lambda` value and `K`. The literature review suggested theoretical compactness *could* be achieved, but bridging this theory to practical, *performant* learning algorithms on noisy, real-world data remains a challenge.

    **5. Conclusions**

    **Conclusion:** This initial feasibility study successfully demonstrated the implementation and training of a novel Simultaneously Trained Soft Stump Ensemble (STSS) with explicit complexity regularization. While it confirmed the potential for gradient-based, end-to-end training of such ensembles, its predictive performance on the Adult dataset was not competitive with AdaBoost using standard hard decision stumps, and it exhibited optimization instabilities (e.g., 0 F1-score for some configurations). The primary research question – whether STSS could achieve competitive performance while being significantly more compact – was not conclusively affirmed in this initial exploration. The results indicate that the direct optimization of compactness is feasible, but achieving high predictive performance simultaneously requires further refinement."""

    PROJECT_BASE_DIR = r"C:\project\AgentLab_demo"
    # --- Initialize and Run Workflow ---
    workflow = PaperWritingWorkflow(
        base_output_path=str(PROJECT_BASE_DIR),
        max_iterations=10,  # Reduced iterations for a quicker demo
        min_score_to_accept=8.5,
        debug_mode=True,  # Enable debug output for the workflow
    )

    print("\nStarting paper writing workflow...\n")
    results_iterator = workflow.run(
        lit_review_sum=lit_review_summary,
        lit_review_paper=lit_review_paper,
        plan=plan,
        exp_code=exp_code,
        exp_result=exp_result,
        res_interpretation=res_interpretation,
        experiment_figures_source_dir=str(PROJECT_BASE_DIR + "\\" + "figures"),
        stream_intermediate_steps=True,  # Set to True to see progress in real-time
    )

    final_response = None
    for chunk in results_iterator:
        # Each chunk is a RunResponse object. You can print its content.
        print(chunk.content, end="", flush=True)
        if chunk.event in [RunEvent.run_completed, RunEvent.run_error]:
            final_response = chunk
            print("\n--- Workflow Finished ---")

    if final_response:
        print(f"\nFinal Status: {final_response.event}")
        print(f"Final Message: {final_response.content}")
        if final_response.images:
            print(
                f"Generated images paths: {[img.url for img in final_response.images]}"
            )
