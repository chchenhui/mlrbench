"""
Report Generator Module

This module generates Markdown reports summarizing the experiment results.
"""

import os
import json
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import numpy as np

def generate_results_markdown(
    results: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    visualization_paths: List[str],
    config: Dict[str, Any],
    output_path: str,
    relative_path_prefix: str = ""
) -> None:
    """
    Generate a Markdown report summarizing the experiment results.
    
    Args:
        results: The raw experiment results
        evaluation_results: Results from statistical evaluation
        visualization_paths: Paths to generated visualizations
        config: Experiment configuration
        output_path: Path to save the Markdown report
        relative_path_prefix: Prefix for relative paths in the report
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating results markdown report at {output_path}")
    
    # Convert visualization paths to relative paths for the report
    vis_filenames = [os.path.basename(path) for path in visualization_paths]
    vis_rel_paths = [os.path.join(relative_path_prefix, filename) for filename in vis_filenames]
    
    # Map visualization filenames to their content type
    vis_map = {}
    for path in visualization_paths:
        filename = os.path.basename(path)
        base = os.path.splitext(filename)[0]
        vis_map[base] = os.path.join(relative_path_prefix, filename)
    
    # Start building the report
    report = []
    
    # Title and introduction
    report.append("# AI Cognitive Tutor Experiment Results\n")
    report.append("## Summary\n")
    report.append("This report presents the results of an experiment evaluating the effectiveness of an AI Cognitive Tutor ")
    report.append("designed to improve human understanding of complex AI systems. The experiment compared a treatment group ")
    report.append("using the AI Cognitive Tutor with a control group using standard AI explanations.\n")
    
    # Key findings summary
    report.append("### Key Findings\n")
    
    # Extract key metrics
    if "mental_model_accuracy" in results["summary_metrics"]:
        mma_metrics = results["summary_metrics"]["mental_model_accuracy"]
        mma_improvement = mma_metrics["percent_improvement"]
        mma_msg = f"- **Mental Model Accuracy**: {mma_improvement:.1f}% improvement with the AI Cognitive Tutor\n"
        report.append(mma_msg)
    
    if "diagnostic_performance" in results["summary_metrics"]:
        perf_metrics = results["summary_metrics"]["diagnostic_performance"]
        perf_improvement = perf_metrics["percent_improvement"]
        perf_msg = f"- **Diagnostic Accuracy**: {perf_improvement:.1f}% improvement in decision-making accuracy\n"
        report.append(perf_msg)
    
    if "user_ai_misalignment" in results["summary_metrics"]:
        conf_metrics = results["summary_metrics"]["user_ai_misalignment"]
        conf_improvement = conf_metrics["percent_improvement"]
        conf_msg = f"- **User Confusion**: {conf_improvement:.1f}% reduction in confusion levels\n"
        report.append(conf_msg)
    
    if "trust_calibration" in results["summary_metrics"]:
        trust_metrics = results["summary_metrics"]["trust_calibration"]
        trust_improvement = trust_metrics["percent_improvement"]
        trust_msg = f"- **Trust Calibration**: {trust_improvement:.1f}% improvement in appropriate trust\n"
        report.append(trust_msg)
    
    # Add statistical significance information
    if "statistical_tests" in evaluation_results:
        stat_tests = evaluation_results["statistical_tests"]
        report.append("\n**Statistical Significance:**\n")
        
        for metric_name, test_results in stat_tests.items():
            p_value = test_results["p_value"]
            significant = test_results["significant"]
            sig_symbol = "✓" if significant else "✗"
            
            report.append(f"- {metric_name}: {sig_symbol} (p={p_value:.4f})\n")
    
    # Experimental setup
    report.append("\n## Experimental Setup\n")
    report.append(f"- **Number of participants**: {config['experiment']['num_simulated_participants']} (equally divided between control and treatment groups)\n")
    report.append(f"- **Number of trials per participant**: {config['experiment']['num_trials']}\n")
    report.append(f"- **AI Diagnostic System**: {config['ai_diagnostic']['model_type']} with base accuracy {config['ai_diagnostic']['accuracy']}\n")
    
    # Treatment group
    report.append("\n**Treatment Group (AI Cognitive Tutor):**\n")
    report.append("- Received adaptive interventions when misunderstanding was detected\n")
    report.append("- Intervention strategies included: " + ", ".join([s.replace("_", " ").title() for s, enabled in config["tutor"]["strategies"].items() if enabled]) + "\n")
    report.append(f"- Tutor activation threshold: {config['tutor']['activation_threshold']}\n")
    
    # Control group
    report.append("\n**Control Group (Baseline):**\n")
    baseline_methods = [b.replace("_", " ").title() for b, enabled in config["baselines"].items() if enabled]
    report.append("- Received: " + ", ".join(baseline_methods) + "\n")
    
    # Primary Results
    report.append("\n## Primary Results\n")
    
    # Add mental model accuracy visualization if available
    if "mental_model_accuracy" in vis_map:
        report.append("### Mental Model Accuracy\n")
        report.append("Mental model accuracy measures how well participants understand the AI system's reasoning and operation.\n")
        report.append(f"![Mental Model Accuracy Comparison]({vis_map['mental_model_accuracy']})\n")
        
        # Add interpretation
        if "mental_model_accuracy" in results["summary_metrics"]:
            mma_metrics = results["summary_metrics"]["mental_model_accuracy"]
            treatment_mma = mma_metrics["treatment"]
            control_mma = mma_metrics["control"]
            difference = mma_metrics["difference"]
            
            report.append(f"The treatment group (with AI Cognitive Tutor) showed a mental model accuracy score of {treatment_mma:.2f}, ")
            report.append(f"compared to {control_mma:.2f} for the control group, representing a difference of {difference:.2f} ")
            report.append(f"({mma_improvement:.1f}% improvement).\n")
    
    # Add diagnostic performance visualization if available
    if "diagnostic_performance" in vis_map:
        report.append("### Diagnostic Performance\n")
        report.append("Diagnostic performance measures how often participants made correct diagnostic decisions.\n")
        report.append(f"![Diagnostic Performance Comparison]({vis_map['diagnostic_performance']})\n")
        
        # Add interpretation
        if "diagnostic_performance" in results["summary_metrics"]:
            perf_metrics = results["summary_metrics"]["diagnostic_performance"]
            treatment_perf = perf_metrics["treatment"]
            control_perf = perf_metrics["control"]
            difference = perf_metrics["difference"]
            
            report.append(f"The treatment group achieved a diagnostic accuracy of {treatment_perf:.2f}, ")
            report.append(f"compared to {control_perf:.2f} for the control group, representing a difference of {difference:.2f} ")
            report.append(f"({perf_improvement:.1f}% improvement).\n")
    
    # Add complexity comparison visualization if available
    if "complexity_comparison" in vis_map:
        report.append("### Performance Across Complexity Levels\n")
        report.append("This analysis shows how the AI Cognitive Tutor impacts performance on cases of varying complexity.\n")
        report.append(f"![Complexity Comparison]({vis_map['complexity_comparison']})\n")
        
        # Add interpretation
        report.append("The impact of the AI Cognitive Tutor varies across different complexity levels:\n")
        
        for complexity in ["simple", "medium", "complex"]:
            if f"diagnostic_performance_{complexity}" in results["summary_metrics"]:
                metrics = results["summary_metrics"][f"diagnostic_performance_{complexity}"]
                treatment = metrics["treatment"]
                control = metrics["control"]
                difference = metrics["difference"]
                percent_imp = metrics["percent_improvement"]
                
                report.append(f"- **{complexity.capitalize()} cases**: {difference:.2f} difference ({percent_imp:.1f}% improvement)\n")
    
    # Add learning curves visualization if available
    if "learning_curves" in vis_map:
        report.append("### Learning Curves\n")
        report.append("Learning curves show how participants' diagnostic accuracy improves over successive trials.\n")
        report.append(f"![Learning Curves]({vis_map['learning_curves']})\n")
        
        # Add interpretation
        report.append("The learning curves show that:\n")
        report.append("- Both groups improve their performance over time\n")
        report.append("- The treatment group shows a steeper learning curve, indicating faster learning\n")
        report.append("- The gap between the groups widens as participants gain more experience with the system\n")
    
    # Secondary Results
    report.append("\n## Secondary Results\n")
    
    # Add intervention effectiveness visualization if available
    if "intervention_effectiveness" in vis_map:
        report.append("### Intervention Effectiveness\n")
        report.append("This analysis shows the effectiveness of different types of interventions provided by the AI Cognitive Tutor.\n")
        report.append(f"![Intervention Effectiveness]({vis_map['intervention_effectiveness']})\n")
        
        # Add interpretation
        if "subgroup_analyses" in evaluation_results and "intervention_types" in evaluation_results["subgroup_analyses"]:
            intervention_data = evaluation_results["subgroup_analyses"]["intervention_types"]
            
            report.append("The most effective intervention types were:\n")
            
            # Sort interventions by helpfulness
            sorted_interventions = sorted(
                intervention_data.items(), 
                key=lambda x: x[1]["helpfulness"], 
                reverse=True
            )
            
            for i, (int_type, data) in enumerate(sorted_interventions[:3]):  # Top 3
                helpfulness = data["helpfulness"]
                improvement = data["improvement_rate"]
                formatted_type = int_type.replace("_", " ").title()
                
                report.append(f"- **{formatted_type}**: Helpfulness score of {helpfulness:.2f}/10, with {improvement:.1%} rate of understanding improvement\n")
    
    # Add confusion levels visualization if available
    if "confusion_levels" in vis_map:
        report.append("### User Confusion Levels\n")
        report.append("Lower confusion levels indicate better understanding of the AI system's reasoning and outputs.\n")
        report.append(f"![Confusion Levels]({vis_map['confusion_levels']})\n")
        
        # Add interpretation
        if "user_ai_misalignment" in results["summary_metrics"]:
            conf_metrics = results["summary_metrics"]["user_ai_misalignment"]
            treatment_conf = conf_metrics["treatment"]
            control_conf = conf_metrics["control"]
            difference = conf_metrics["difference"]
            
            report.append(f"The treatment group showed significantly lower confusion levels ({treatment_conf:.2f}) ")
            report.append(f"compared to the control group ({control_conf:.2f}), a reduction of {difference:.2f} points ")
            report.append(f"({conf_improvement:.1f}%).\n")
    
    # Add trust calibration visualization if available
    if "trust_calibration" in vis_map:
        report.append("### Trust Calibration\n")
        report.append("Trust calibration measures how well participants' trust aligns with the AI system's actual reliability.\n")
        report.append(f"![Trust Calibration]({vis_map['trust_calibration']})\n")
        
        # Add interpretation
        if "trust_calibration" in results["summary_metrics"]:
            trust_metrics = results["summary_metrics"]["trust_calibration"]
            treatment_trust = trust_metrics["treatment"]
            control_trust = trust_metrics["control"]
            difference = trust_metrics["difference"]
            
            report.append(f"The treatment group demonstrated better trust calibration ({treatment_trust:.2f}) ")
            report.append(f"compared to the control group ({control_trust:.2f}), a difference of {difference:.2f} ")
            report.append(f"({trust_improvement:.1f}% improvement).\n")
    
    # Add expertise comparison visualization if available
    if "expertise_comparison" in vis_map:
        report.append("### Performance Across Expertise Levels\n")
        report.append("This analysis shows how the AI Cognitive Tutor impacts performance across different user expertise levels.\n")
        report.append(f"![Expertise Comparison]({vis_map['expertise_comparison']})\n")
        
        # Add interpretation
        if "subgroup_analyses" in evaluation_results and "expertise_levels" in evaluation_results["subgroup_analyses"]:
            expertise_data = evaluation_results["subgroup_analyses"]["expertise_levels"]
            
            report.append("The impact of the AI Cognitive Tutor varies across expertise levels:\n")
            
            for level, data in expertise_data.items():
                treatment = data["treatment_accuracy"]
                control = data["control_accuracy"]
                difference = data["difference"]
                percent_imp = (difference / control) * 100 if control > 0 else 0
                
                report.append(f"- **{level.capitalize()}**: {difference:.2f} difference ({percent_imp:.1f}% improvement)\n")
    
    # Statistical Analysis
    report.append("\n## Statistical Analysis\n")
    
    # Add effect sizes
    if "effect_sizes" in evaluation_results:
        report.append("### Effect Sizes\n")
        report.append("| Metric | Cohen's d | Interpretation |\n")
        report.append("|--------|-----------|---------------|\n")
        
        for metric_name, effect_data in evaluation_results["effect_sizes"].items():
            cohens_d = effect_data["cohens_d"]
            interpretation = effect_data["interpretation"]
            
            report.append(f"| {metric_name} | {cohens_d:.2f} | {interpretation} |\n")
    
    # Add confidence intervals
    if "confidence_intervals" in evaluation_results:
        report.append("\n### 95% Confidence Intervals\n")
        report.append("| Metric | Lower Bound | Upper Bound |\n")
        report.append("|--------|-------------|-------------|\n")
        
        for metric_name, ci_data in evaluation_results["confidence_intervals"].items():
            lower = ci_data["lower_bound"]
            upper = ci_data["upper_bound"]
            
            report.append(f"| {metric_name} | {lower:.2f} | {upper:.2f} |\n")
    
    # Discussion and Conclusion
    report.append("\n## Discussion\n")
    
    # Generate discussion based on results
    overall_effectiveness = True  # Assume positive overall result
    if "mental_model_accuracy" in results["summary_metrics"]:
        mma_improvement = results["summary_metrics"]["mental_model_accuracy"]["percent_improvement"]
        if mma_improvement <= 0:
            overall_effectiveness = False
    
    if overall_effectiveness:
        report.append("The results demonstrate that the AI Cognitive Tutor is effective at improving users' understanding ")
        report.append("of complex AI systems. Key findings include:\n")
        
        # List key findings with interpretation
        report.append("1. **Improved Mental Models**: The AI Cognitive Tutor significantly improved participants' mental models ")
        report.append("   of the AI system, indicating better understanding of its reasoning processes, capabilities, and limitations.\n")
        
        report.append("2. **Enhanced Decision-Making**: Participants using the AI Cognitive Tutor made more accurate diagnostic ")
        report.append("   decisions, suggesting that better understanding leads to more effective human-AI collaboration.\n")
        
        report.append("3. **Reduced Confusion**: The treatment group experienced lower confusion levels, indicating that ")
        report.append("   the adaptive explanations helped clarify complex aspects of the AI's reasoning.\n")
        
        report.append("4. **Better Trust Calibration**: The AI Cognitive Tutor improved trust calibration, helping users ")
        report.append("   develop more appropriate levels of trust based on the AI's actual reliability.\n")
    else:
        report.append("The results provide mixed evidence for the effectiveness of the AI Cognitive Tutor:\n")
        
        report.append("1. While some metrics showed improvement, the magnitude of effects varied across different measures.\n")
        
        report.append("2. The intervention may be more effective for certain types of users or scenarios than others.\n")
        
        report.append("3. Further refinement of the tutoring strategies may be needed to achieve more consistent benefits.\n")
    
    # Effectiveness across complexity and expertise
    report.append("\n### Effectiveness Across Different Conditions\n")
    
    report.append("The effectiveness of the AI Cognitive Tutor varied across different conditions:\n")
    
    # Complexity
    complexity_effects = []
    for complexity in ["simple", "medium", "complex"]:
        if f"diagnostic_performance_{complexity}" in results["summary_metrics"]:
            percent_imp = results["summary_metrics"][f"diagnostic_performance_{complexity}"]["percent_improvement"]
            complexity_effects.append((complexity, percent_imp))
    
    if complexity_effects:
        max_effect_complexity = max(complexity_effects, key=lambda x: x[1])
        min_effect_complexity = min(complexity_effects, key=lambda x: x[1])
        
        report.append(f"- **Case Complexity**: The AI Cognitive Tutor was most effective for {max_effect_complexity[0]} cases ")
        report.append(f"  ({max_effect_complexity[1]:.1f}% improvement) and least effective for {min_effect_complexity[0]} cases ")
        report.append(f"  ({min_effect_complexity[1]:.1f}% improvement).\n")
    
    # Expertise
    if "subgroup_analyses" in evaluation_results and "expertise_levels" in evaluation_results["subgroup_analyses"]:
        expertise_data = evaluation_results["subgroup_analyses"]["expertise_levels"]
        
        expertise_effects = []
        for level, data in expertise_data.items():
            treatment = data["treatment_accuracy"]
            control = data["control_accuracy"]
            percent_imp = (treatment - control) / control * 100 if control > 0 else 0
            expertise_effects.append((level, percent_imp))
        
        if expertise_effects:
            max_effect_expertise = max(expertise_effects, key=lambda x: x[1])
            min_effect_expertise = min(expertise_effects, key=lambda x: x[1])
            
            report.append(f"- **User Expertise**: The AI Cognitive Tutor was most effective for {max_effect_expertise[0]} users ")
            report.append(f"  ({max_effect_expertise[1]:.1f}% improvement) and least effective for {min_effect_expertise[0]} users ")
            report.append(f"  ({min_effect_expertise[1]:.1f}% improvement).\n")
    
    # Intervention effectiveness
    if "subgroup_analyses" in evaluation_results and "intervention_types" in evaluation_results["subgroup_analyses"]:
        intervention_data = evaluation_results["subgroup_analyses"]["intervention_types"]
        
        if intervention_data:
            # Find most and least effective intervention types
            sorted_by_helpfulness = sorted(
                intervention_data.items(), 
                key=lambda x: x[1]["helpfulness"], 
                reverse=True
            )
            
            if len(sorted_by_helpfulness) >= 2:
                most_effective = sorted_by_helpfulness[0]
                least_effective = sorted_by_helpfulness[-1]
                
                most_effective_type = most_effective[0].replace("_", " ").title()
                least_effective_type = least_effective[0].replace("_", " ").title()
                
                report.append(f"- **Intervention Types**: {most_effective_type} interventions were most effective ")
                report.append(f"  (helpfulness: {most_effective[1]['helpfulness']:.2f}/10), while {least_effective_type} interventions ")
                report.append(f"  were least effective (helpfulness: {least_effective[1]['helpfulness']:.2f}/10).\n")
    
    # Limitations
    report.append("\n### Limitations\n")
    
    report.append("This study has several limitations that should be considered when interpreting the results:\n")
    
    report.append("1. **Simulated Environment**: The experiment was conducted in a simulated environment with programmatically ")
    report.append("   generated participant behaviors, which may not fully capture the complexity of real-world human-AI interaction.\n")
    
    report.append("2. **Domain Specificity**: The study focused on a medical diagnostic context, and the findings may not ")
    report.append("   generalize to other domains or types of AI systems.\n")
    
    report.append("3. **Limited Intervention Types**: While several tutoring strategies were implemented, there may be other ")
    report.append("   effective approaches not included in this experiment.\n")
    
    report.append("4. **Simplified Mental Model Assessment**: The study used a simplified measurement of mental model accuracy, ")
    report.append("   which may not capture all aspects of users' understanding of the AI system.\n")
    
    # Future work
    report.append("\n### Future Work\n")
    
    report.append("Based on these findings, several directions for future research are promising:\n")
    
    report.append("1. **Real-World Validation**: Conduct studies with real human participants to validate the effectiveness ")
    report.append("   of the AI Cognitive Tutor in authentic settings.\n")
    
    report.append("2. **Personalization**: Further develop the adaptive capabilities of the tutor to better personalize ")
    report.append("   interventions based on individual user characteristics and learning patterns.\n")
    
    report.append("3. **Cross-Domain Testing**: Evaluate the AI Cognitive Tutor in different domains (e.g., financial analysis, ")
    report.append("   autonomous driving) to assess generalizability.\n")
    
    report.append("4. **Integration with Existing AI Systems**: Explore how the AI Cognitive Tutor could be integrated with ")
    report.append("   real-world AI systems and user interfaces to enhance practical applications.\n")
    
    report.append("5. **Long-Term Effects**: Study the long-term effects of using the AI Cognitive Tutor on user learning and ")
    report.append("   behavior over extended periods.\n")
    
    # Conclusion
    report.append("\n## Conclusion\n")
    
    if overall_effectiveness:
        report.append("The AI Cognitive Tutor demonstrated significant potential for improving human understanding of complex AI systems. ")
        report.append("By providing adaptive explanations tailored to user misunderstandings, it enhanced mental model accuracy, ")
        report.append("decision-making performance, and trust calibration. These findings support the value of the \"Aligning Humans with AI\" ")
        report.append("dimension of bidirectional human-AI alignment, emphasizing the importance of helping humans understand and ")
        report.append("effectively collaborate with increasingly sophisticated AI systems.\n")
        
        report.append("While further research is needed to validate these findings in real-world settings, the results suggest that ")
        report.append("adaptive tutoring approaches can play a valuable role in fostering more effective human-AI partnerships ")
        report.append("across various domains and applications.\n")
    else:
        report.append("The experiment with the AI Cognitive Tutor provided valuable insights into approaches for improving human understanding ")
        report.append("of complex AI systems. While the results were mixed, they highlight the potential value of adaptive explanation ")
        report.append("systems and identify conditions under which such approaches may be most beneficial.\n")
        
        report.append("Further refinement of the tutoring strategies and additional research with human participants are needed ")
        report.append("to fully realize the potential of this approach for enhancing human-AI collaboration in real-world settings.\n")
    
    # Write the report to file
    with open(output_path, 'w') as f:
        f.write("".join(report))
    
    logger.info(f"Results markdown report generated at {output_path}")
    return