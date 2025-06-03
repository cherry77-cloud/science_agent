from agents.laboratory_agents import get_laboratory_agent
from workflows.plan_formulation_workflow import PlanFormulationWorkflow

if __name__ == "__main__":

    phd, postdoc, _, _ = get_laboratory_agent()
    plan_formulation_workflow = PlanFormulationWorkflow(phd, postdoc)
    topic = "Novel Ensemble Methods of Decision Trees"
    lit_review_sum = r"""
### **Summary of Current Research on Novel Ensemble Methods of Decision Trees**  

Recent research has introduced several innovative ensemble methods to enhance the performance, robustness, and diversity of decision tree-based models. Key developments include:  

1. **Boosting Meta-Trees (2402.06386)**  
   - A novel boosting-based ensemble of meta-trees (themselves ensembles of subtrees) improves predictive accuracy and mitigates overfitting.  
   - Posterior probability-weighted ensembles show superior Bayes risk reduction compared to GBDT and LightGBM.  
   - GBDT-like and uniform weighting schemes perform well on complex datasets where meta-tree assumptions may not hold.  

2. **Island-Based Coevolutionary Ensembles (2412.13762 - ICoEvoRDF)**  
   - Introduces a coevolutionary algorithm with Nash Equilibrium-based voting to enhance robustness against adversarial attacks.  
   - Outperforms existing robust forests (GROOT, FPRDT) and boosting methods (PRAdaBoost) in adversarial accuracy and minimax regret.  
   - Island-based training and Nash weighting improve ensemble diversity and generalization.  

3. **Tree-in-Tree Decision Graphs (2110.00392 - TnT)**  
   - Extends decision trees to directed acyclic graphs (DAGs) by embedding micro-trees recursively, enabling deeper paths without exponential growth.  
   - TnT-based ensembles (TnT-bagging, TnT-AdaBoost) outperform Random Forest and standard AdaBoost in accuracy with fewer splits.  

4. **Heterogeneous Random Forest (2410.19022 - HRF)**  
   - Enhances diversity via weighted feature sampling, penalizing features frequently used at shallow depths in prior trees.  
   - Outperforms traditional Random Forest, XGBoost, and CatBoost in classification accuracy, particularly on datasets with meaningful feature interactions.  
   - Less effective on noisy datasets, indicating sensitivity to feature quality.  

### **Key Findings**  
- **Diversity and Robustness:** Methods like ICoEvoRDF and HRF explicitly promote diversity, improving adversarial robustness and generalization.  
- **Structural Innovations:** Meta-trees and TnT graphs offer alternatives to traditional tree structures, enabling better performance with controlled complexity.  
- **Weighting Schemes:** Boosting (meta-trees), Nash Equilibrium (ICoEvoRDF), and depth-based feature sampling (HRF) improve ensemble effectiveness.  

### **Gaps and Future Work**  
- **Scalability:** Some methods (e.g., ICoEvoRDF, TnT) may have high computational costs; efficiency improvements are needed.  
- **Noise Sensitivity:** HRF's performance drops with noisy features; adaptive weighting schemes could help.  
- **Theoretical Understanding:** More analysis is needed on why meta-trees and TnT graphs outperform traditional trees.  
- **Hybrid Approaches:** Combining ideas (e.g., HRF's diversity with ICoEvoRDF's robustness) could yield further improvements.  

### **Conclusion**  
The field is advancing through structural innovations (meta-trees, TnT), robustness-focused methods (ICoEvoRDF), and diversity-enhancing techniques (HRF). Future work should address scalability, noise robustness, and theoretical foundations while exploring hybrid ensemble strategies.
    """

    plan_workflow_response = plan_formulation_workflow.run(
        topic=topic,
        lit_review_sum=lit_review_sum,
    )
    print(plan_workflow_response.content)
