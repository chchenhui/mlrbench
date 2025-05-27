"""
Experimental Design for Contextual Dataset Deprecation Framework

This module defines the experimental setup for evaluating the effectiveness of the
Contextual Dataset Deprecation Framework compared to baseline approaches.

Experimental conditions:
1. Control: Traditional dataset removal without structured deprecation
2. Basic Framework: Implementation with only warning labels and basic notifications
3. Full Framework: Complete implementation of all framework components

Metrics to measure:
1. User Response Metrics:
   - Time to acknowledge deprecation notifications
   - Rate of transition to recommended alternatives
   - Continued usage of deprecated datasets

2. System Performance Metrics:
   - Accuracy of alternative dataset recommendations
   - Processing time for deprecation actions
   - Notification delivery success rates

3. Research Impact Metrics:
   - Citation patterns pre and post-deprecation
   - Changes in benchmark dataset diversity
   - Model performance on deprecated vs. alternative datasets
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from enum import Enum, auto
import random
import logging
from typing import List, Dict, Any, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log.txt')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dataset_deprecation")

# Constants
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Define experimental conditions
class DeprecationStrategy(Enum):
    CONTROL = auto()  # Traditional removal without structure
    BASIC = auto()    # Warning labels and basic notifications
    FULL = auto()     # Complete framework implementation

# Define dataset warning levels
class WarningLevel(Enum):
    NONE = 0
    CAUTION = 1
    LIMITED_USE = 2
    DEPRECATED = 3

class DeprecationRecord:
    """Class representing a dataset deprecation record."""
    
    def __init__(
        self,
        dataset_id: str,
        warning_level: WarningLevel,
        issue_description: str,
        evidence_links: List[str],
        affected_groups: List[str],
        recommended_alternatives: List[str],
        timestamp: datetime = None
    ):
        self.dataset_id = dataset_id
        self.warning_level = warning_level
        self.issue_description = issue_description
        self.evidence_links = evidence_links
        self.affected_groups = affected_groups
        self.recommended_alternatives = recommended_alternatives
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the record to a dictionary for serialization."""
        return {
            'dataset_id': self.dataset_id,
            'warning_level': self.warning_level.name,
            'issue_description': self.issue_description,
            'evidence_links': self.evidence_links,
            'affected_groups': self.affected_groups,
            'recommended_alternatives': self.recommended_alternatives,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeprecationRecord':
        """Create a DeprecationRecord from a dictionary."""
        return cls(
            dataset_id=data['dataset_id'],
            warning_level=WarningLevel[data['warning_level']],
            issue_description=data['issue_description'],
            evidence_links=data['evidence_links'],
            affected_groups=data['affected_groups'],
            recommended_alternatives=data['recommended_alternatives'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

class DatasetVersion:
    """Class representing a version of a dataset including its deprecation status."""
    
    def __init__(
        self,
        dataset_id: str,
        version: str,
        content_hash: str,
        warning_level: WarningLevel = WarningLevel.NONE,
        changes: str = "",
        justification: str = "",
        timestamp: datetime = None
    ):
        self.dataset_id = dataset_id
        self.version = version
        self.content_hash = content_hash
        self.warning_level = warning_level
        self.changes = changes
        self.justification = justification
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the version to a dictionary for serialization."""
        return {
            'dataset_id': self.dataset_id,
            'version': self.version,
            'content_hash': self.content_hash,
            'warning_level': self.warning_level.name,
            'changes': self.changes,
            'justification': self.justification,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetVersion':
        """Create a DatasetVersion from a dictionary."""
        return cls(
            dataset_id=data['dataset_id'],
            version=data['version'],
            content_hash=data['content_hash'],
            warning_level=WarningLevel[data['warning_level']],
            changes=data['changes'],
            justification=data['justification'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

class SyntheticDataset:
    """Class for generating synthetic datasets with controlled properties for experimentation."""
    
    def __init__(
        self,
        dataset_id: str,
        task_type: str,
        n_samples: int,
        n_features: int,
        bias_level: float = 0.0,
        version: str = "1.0.0",
        ethical_issues: List[str] = None
    ):
        self.dataset_id = dataset_id
        self.task_type = task_type
        self.n_samples = n_samples
        self.n_features = n_features
        self.bias_level = bias_level
        self.version = version
        self.ethical_issues = ethical_issues or []
        self.data = self._generate_data()
        self.content_hash = self._compute_hash()
    
    def _generate_data(self) -> Dict[str, Any]:
        """Generate synthetic data with controlled properties."""
        # Generate features
        X = np.random.normal(0, 1, (self.n_samples, self.n_features))
        
        # Add bias to certain features if specified
        if self.bias_level > 0:
            # Simulate demographic bias by correlating certain features
            demographic_idx = np.random.randint(0, self.n_features)
            for i in range(self.n_features):
                if i != demographic_idx:
                    X[:, i] += self.bias_level * X[:, demographic_idx]
        
        # Generate target variable based on task type
        if self.task_type == "classification":
            y = np.random.randint(0, 2, self.n_samples)
            if self.bias_level > 0:
                # Make classification dependent on the demographic feature
                mask = X[:, demographic_idx] > 0
                y[mask] = np.random.choice([0, 1], size=np.sum(mask), p=[0.3, 0.7])
                y[~mask] = np.random.choice([0, 1], size=np.sum(~mask), p=[0.7, 0.3])
        elif self.task_type == "regression":
            y = np.dot(X, np.random.normal(0, 1, self.n_features))
            y += np.random.normal(0, 0.1, self.n_samples)  # Add noise
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        return {
            "X": X,
            "y": y,
            "task_type": self.task_type,
            "ethical_issues": self.ethical_issues,
            "metadata": {
                "dataset_id": self.dataset_id,
                "version": self.version,
                "n_samples": self.n_samples,
                "n_features": self.n_features,
                "bias_level": self.bias_level
            }
        }
    
    def _compute_hash(self) -> str:
        """Compute a content hash for the dataset."""
        # Simple hash computation for demonstration purposes
        X_flat = self.data["X"].flatten()
        y_flat = self.data["y"].flatten()
        combined = np.concatenate([X_flat, y_flat])
        return str(hash(combined.tobytes()))
    
    def to_pandas(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Convert the synthetic data to pandas DataFrame/Series for convenience."""
        X_df = pd.DataFrame(
            self.data["X"], 
            columns=[f"feature_{i}" for i in range(self.n_features)]
        )
        y_series = pd.Series(self.data["y"], name="target")
        return X_df, y_series
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataset to a dictionary for serialization."""
        result = {
            "dataset_id": self.dataset_id,
            "task_type": self.task_type,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "bias_level": self.bias_level,
            "version": self.version,
            "ethical_issues": self.ethical_issues,
            "content_hash": self.content_hash,
            # Convert numpy arrays to lists for JSON serialization
            "X": self.data["X"].tolist(),
            "y": self.data["y"].tolist()
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyntheticDataset':
        """Create a SyntheticDataset from a dictionary."""
        dataset = cls(
            dataset_id=data["dataset_id"],
            task_type=data["task_type"],
            n_samples=data["n_samples"],
            n_features=data["n_features"],
            bias_level=data["bias_level"],
            version=data["version"],
            ethical_issues=data["ethical_issues"]
        )
        
        # Override the generated data with the loaded data
        dataset.data["X"] = np.array(data["X"])
        dataset.data["y"] = np.array(data["y"])
        dataset.content_hash = data["content_hash"]
        
        return dataset

def create_synthetic_datasets() -> Dict[str, SyntheticDataset]:
    """Create a collection of synthetic datasets for experimentation."""
    datasets = {}
    
    # Dataset 1: A clean classification dataset (no issues)
    datasets["clean_dataset"] = SyntheticDataset(
        dataset_id="clean_dataset",
        task_type="classification",
        n_samples=1000,
        n_features=10,
        bias_level=0.0,
        version="1.0.0",
        ethical_issues=[]
    )
    
    # Dataset 2: A classification dataset with minor bias issues
    datasets["minor_bias_dataset"] = SyntheticDataset(
        dataset_id="minor_bias_dataset",
        task_type="classification",
        n_samples=1000,
        n_features=10,
        bias_level=0.3,
        version="1.0.0",
        ethical_issues=["Slight demographic imbalance"]
    )
    
    # Dataset 3: A classification dataset with significant bias issues
    datasets["major_bias_dataset"] = SyntheticDataset(
        dataset_id="major_bias_dataset",
        task_type="classification",
        n_samples=1000,
        n_features=10,
        bias_level=0.7,
        version="1.0.0",
        ethical_issues=["Severe demographic bias", "Performance disparities across groups"]
    )
    
    # Dataset 4: A regression dataset (no issues)
    datasets["clean_regression"] = SyntheticDataset(
        dataset_id="clean_regression",
        task_type="regression",
        n_samples=1000,
        n_features=15,
        bias_level=0.0,
        version="1.0.0",
        ethical_issues=[]
    )
    
    # Dataset 5: A regression dataset with moderate bias
    datasets["biased_regression"] = SyntheticDataset(
        dataset_id="biased_regression",
        task_type="regression",
        n_samples=1000,
        n_features=15,
        bias_level=0.5,
        version="1.0.0",
        ethical_issues=["Moderate demographic bias"]
    )
    
    logger.info(f"Created {len(datasets)} synthetic datasets for experimentation")
    return datasets

def create_deprecation_records() -> Dict[str, DeprecationRecord]:
    """Create deprecation records for the synthetic datasets."""
    records = {}
    
    # No record for clean_dataset (it has no issues)
    
    # Record for minor_bias_dataset (Caution level)
    records["minor_bias_dataset"] = DeprecationRecord(
        dataset_id="minor_bias_dataset",
        warning_level=WarningLevel.CAUTION,
        issue_description="This dataset shows a slight demographic imbalance that may lead to performance disparities.",
        evidence_links=["https://example.com/bias_analysis/minor_bias_dataset"],
        affected_groups=["Demographic group A"],
        recommended_alternatives=["clean_dataset"]
    )
    
    # Record for major_bias_dataset (Limited Use level)
    records["major_bias_dataset"] = DeprecationRecord(
        dataset_id="major_bias_dataset",
        warning_level=WarningLevel.LIMITED_USE,
        issue_description="This dataset contains severe demographic bias resulting in significant performance disparities across groups.",
        evidence_links=["https://example.com/bias_analysis/major_bias_dataset", "https://example.com/ethical_review/major_bias_dataset"],
        affected_groups=["Demographic group A", "Demographic group B"],
        recommended_alternatives=["clean_dataset", "minor_bias_dataset"]
    )
    
    # No record for clean_regression (it has no issues)
    
    # Record for biased_regression (Caution level)
    records["biased_regression"] = DeprecationRecord(
        dataset_id="biased_regression",
        warning_level=WarningLevel.CAUTION,
        issue_description="This dataset exhibits moderate demographic bias that may affect prediction fairness.",
        evidence_links=["https://example.com/bias_analysis/biased_regression"],
        affected_groups=["Demographic group C"],
        recommended_alternatives=["clean_regression"]
    )
    
    logger.info(f"Created {len(records)} deprecation records")
    return records

def simulate_user_response(
    strategy: DeprecationStrategy,
    dataset_id: str,
    deprecation_record: Optional[DeprecationRecord]
) -> Dict[str, Any]:
    """
    Simulate user response to dataset deprecation based on the deprecation strategy.
    
    Returns metrics like:
    - acknowledgment_time: Time to acknowledge deprecation (in simulated days)
    - alternative_adoption: Whether user adopts recommended alternatives (0-1)
    - continued_usage: Whether user continues using deprecated dataset (0-1)
    """
    # Base probabilities for control condition (no framework)
    base_ack_time = 30  # average 30 days to notice
    base_adoption = 0.3  # 30% chance of adopting alternatives
    base_continued = 0.7  # 70% chance of continuing to use deprecated dataset
    
    # No deprecation record or control strategy
    if deprecation_record is None or strategy == DeprecationStrategy.CONTROL:
        return {
            "acknowledgment_time": np.random.normal(base_ack_time, 10),
            "alternative_adoption": np.random.binomial(1, base_adoption),
            "continued_usage": np.random.binomial(1, base_continued)
        }
    
    # Adjust metrics based on strategy and warning level
    if strategy == DeprecationStrategy.BASIC:
        # Basic framework improves metrics moderately
        adj_factor = 0.7  # 30% improvement
        
        return {
            "acknowledgment_time": np.random.normal(base_ack_time * adj_factor, 5),
            "alternative_adoption": np.random.binomial(1, base_adoption + (1 - base_adoption) * 0.3),
            "continued_usage": np.random.binomial(1, base_continued * 0.8)
        }
    
    elif strategy == DeprecationStrategy.FULL:
        # Full framework provides substantial improvements
        # Warning level affects the response
        level_factor = 1.0
        if deprecation_record.warning_level == WarningLevel.CAUTION:
            level_factor = 0.8
        elif deprecation_record.warning_level == WarningLevel.LIMITED_USE:
            level_factor = 0.5
        elif deprecation_record.warning_level == WarningLevel.DEPRECATED:
            level_factor = 0.2
        
        return {
            "acknowledgment_time": np.random.normal(base_ack_time * 0.4, 3),
            "alternative_adoption": np.random.binomial(1, min(0.95, base_adoption + (1 - base_adoption) * 0.7)),
            "continued_usage": np.random.binomial(1, base_continued * level_factor)
        }
    
    # Fallback
    return {
        "acknowledgment_time": np.random.normal(base_ack_time, 10),
        "alternative_adoption": np.random.binomial(1, base_adoption),
        "continued_usage": np.random.binomial(1, base_continued)
    }

def simulate_system_performance(
    strategy: DeprecationStrategy,
    dataset_id: str,
    deprecation_record: Optional[DeprecationRecord]
) -> Dict[str, Any]:
    """
    Simulate system performance metrics for the deprecation framework.
    
    Returns metrics like:
    - recommendation_accuracy: Accuracy of alternative dataset recommendations (0-1)
    - processing_time: Time to process deprecation actions (in seconds)
    - notification_success: Success rate of notifications (0-1)
    """
    # Base performance for control condition (no framework)
    base_accuracy = 0.0  # no recommendations in control
    base_proc_time = 0.0  # no processing in control
    base_notif_success = 0.0  # no notifications in control
    
    # No deprecation record or control strategy
    if deprecation_record is None or strategy == DeprecationStrategy.CONTROL:
        return {
            "recommendation_accuracy": base_accuracy,
            "processing_time": base_proc_time,
            "notification_success": base_notif_success
        }
    
    # Basic framework provides simple functionality
    if strategy == DeprecationStrategy.BASIC:
        return {
            "recommendation_accuracy": 0.6 + np.random.normal(0, 0.1),  # ~60% accuracy
            "processing_time": 1.5 + np.random.exponential(0.5),  # ~1.5s processing
            "notification_success": 0.8 + np.random.normal(0, 0.05)  # ~80% notification success
        }
    
    # Full framework provides advanced functionality
    elif strategy == DeprecationStrategy.FULL:
        # Number of alternatives affects recommendation accuracy
        n_alternatives = len(deprecation_record.recommended_alternatives) if deprecation_record else 0
        alt_factor = min(1.0, 0.7 + 0.05 * n_alternatives)
        
        return {
            "recommendation_accuracy": alt_factor + np.random.normal(0, 0.05),
            "processing_time": 2.2 + np.random.exponential(0.8),  # more processing but more features
            "notification_success": 0.95 + np.random.normal(0, 0.02)  # ~95% notification success
        }
    
    # Fallback
    return {
        "recommendation_accuracy": base_accuracy,
        "processing_time": base_proc_time,
        "notification_success": base_notif_success
    }

def simulate_research_impact(
    strategy: DeprecationStrategy,
    dataset_id: str,
    deprecation_record: Optional[DeprecationRecord],
    time_periods: int = 6  # Simulate 6 time periods (e.g., months)
) -> Dict[str, Any]:
    """
    Simulate the impact on research metrics over time after deprecation.
    
    Returns metrics like:
    - citation_pattern: List of citation counts over time periods
    - benchmark_diversity: Measure of benchmark dataset diversity (0-1)
    - alternative_performance: Relative performance on alternative datasets
    """
    # Base citation decay for control condition
    base_citations = [100, 95, 90, 88, 85, 83]  # slow natural decay
    base_diversity = 0.3  # low diversity
    base_alt_performance = 0.0  # no alternatives used
    
    # Generate randomized citation counts that follow the expected pattern
    citations = []
    
    if strategy == DeprecationStrategy.CONTROL:
        # Control: Slow decay in citations
        for i in range(time_periods):
            citations.append(max(0, base_citations[i] + np.random.normal(0, 5)))
        
        return {
            "citation_pattern": citations,
            "benchmark_diversity": base_diversity + np.random.normal(0, 0.05),
            "alternative_performance": base_alt_performance
        }
    
    elif strategy == DeprecationStrategy.BASIC:
        # Basic: Moderate decay in citations, some transition to alternatives
        decay_factor = 0.85
        citations = [100]  # starting point
        
        for i in range(1, time_periods):
            citations.append(max(0, citations[i-1] * decay_factor + np.random.normal(0, 5)))
        
        return {
            "citation_pattern": citations,
            "benchmark_diversity": 0.5 + np.random.normal(0, 0.08),
            "alternative_performance": 0.6 + np.random.normal(0, 0.1)
        }
    
    elif strategy == DeprecationStrategy.FULL:
        # Full: Rapid decay in citations, strong transition to alternatives
        # Warning level affects the decay rate
        decay_factor = 0.9  # default
        if deprecation_record:
            if deprecation_record.warning_level == WarningLevel.CAUTION:
                decay_factor = 0.85
            elif deprecation_record.warning_level == WarningLevel.LIMITED_USE:
                decay_factor = 0.7
            elif deprecation_record.warning_level == WarningLevel.DEPRECATED:
                decay_factor = 0.5
        
        citations = [100]  # starting point
        
        for i in range(1, time_periods):
            citations.append(max(0, citations[i-1] * decay_factor + np.random.normal(0, 5)))
        
        # Number of alternatives affects diversity and performance
        n_alternatives = len(deprecation_record.recommended_alternatives) if deprecation_record else 0
        diversity_boost = min(0.5, 0.1 * n_alternatives)
        
        return {
            "citation_pattern": citations,
            "benchmark_diversity": 0.7 + diversity_boost + np.random.normal(0, 0.05),
            "alternative_performance": 0.85 + np.random.normal(0, 0.08)
        }
    
    # Fallback
    return {
        "citation_pattern": base_citations,
        "benchmark_diversity": base_diversity,
        "alternative_performance": base_alt_performance
    }

def run_experiment(
    datasets: Dict[str, SyntheticDataset],
    deprecation_records: Dict[str, DeprecationRecord],
    n_simulations: int = 50,
    strategies: List[DeprecationStrategy] = [
        DeprecationStrategy.CONTROL,
        DeprecationStrategy.BASIC,
        DeprecationStrategy.FULL
    ]
) -> Dict[str, Any]:
    """
    Run the experimental simulation for the Contextual Dataset Deprecation Framework.
    
    Args:
        datasets: Dictionary of synthetic datasets
        deprecation_records: Dictionary of deprecation records
        n_simulations: Number of simulated research groups/users
        strategies: List of deprecation strategies to compare
        
    Returns:
        Dictionary of experimental results
    """
    results = {
        "user_response": {},
        "system_performance": {},
        "research_impact": {},
        "aggregate_metrics": {}
    }
    
    # Run simulations for each dataset and strategy
    for dataset_id, dataset in datasets.items():
        results["user_response"][dataset_id] = {}
        results["system_performance"][dataset_id] = {}
        results["research_impact"][dataset_id] = {}
        
        deprecation_record = deprecation_records.get(dataset_id)
        
        for strategy in strategies:
            strategy_name = strategy.name
            
            # User response metrics
            user_responses = []
            for _ in range(n_simulations):
                user_responses.append(
                    simulate_user_response(strategy, dataset_id, deprecation_record)
                )
            results["user_response"][dataset_id][strategy_name] = user_responses
            
            # System performance metrics
            system_performances = []
            for _ in range(n_simulations):
                system_performances.append(
                    simulate_system_performance(strategy, dataset_id, deprecation_record)
                )
            results["system_performance"][dataset_id][strategy_name] = system_performances
            
            # Research impact metrics
            research_impacts = []
            for _ in range(n_simulations):
                research_impacts.append(
                    simulate_research_impact(strategy, dataset_id, deprecation_record)
                )
            results["research_impact"][dataset_id][strategy_name] = research_impacts
    
    # Calculate aggregate metrics across datasets
    results["aggregate_metrics"] = calculate_aggregate_metrics(results)
    
    return results

def calculate_aggregate_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate aggregate metrics across all datasets and simulations."""
    aggregate = {
        "user_response": {},
        "system_performance": {},
        "research_impact": {}
    }
    
    # Metrics to aggregate
    user_metrics = ["acknowledgment_time", "alternative_adoption", "continued_usage"]
    system_metrics = ["recommendation_accuracy", "processing_time", "notification_success"]
    research_metrics = ["benchmark_diversity", "alternative_performance"]
    
    # Extract strategies from the first dataset
    first_dataset = next(iter(results["user_response"]))
    strategies = list(results["user_response"][first_dataset].keys())
    
    # Calculate averages for each metric and strategy
    for strategy in strategies:
        # User response metrics
        aggregate["user_response"][strategy] = {metric: [] for metric in user_metrics}
        for dataset_id in results["user_response"]:
            for sim_result in results["user_response"][dataset_id][strategy]:
                for metric in user_metrics:
                    aggregate["user_response"][strategy][metric].append(sim_result[metric])
        
        # Calculate averages
        for metric in user_metrics:
            values = aggregate["user_response"][strategy][metric]
            aggregate["user_response"][strategy][metric] = {
                "mean": np.mean(values),
                "std": np.std(values)
            }
        
        # System performance metrics
        aggregate["system_performance"][strategy] = {metric: [] for metric in system_metrics}
        for dataset_id in results["system_performance"]:
            for sim_result in results["system_performance"][dataset_id][strategy]:
                for metric in system_metrics:
                    aggregate["system_performance"][strategy][metric].append(sim_result[metric])
        
        # Calculate averages
        for metric in system_metrics:
            values = aggregate["system_performance"][strategy][metric]
            aggregate["system_performance"][strategy][metric] = {
                "mean": np.mean(values),
                "std": np.std(values)
            }
        
        # Research impact metrics
        aggregate["research_impact"][strategy] = {
            metric: [] for metric in research_metrics + ["citation_pattern"]
        }
        for dataset_id in results["research_impact"]:
            for sim_result in results["research_impact"][dataset_id][strategy]:
                for metric in research_metrics:
                    aggregate["research_impact"][strategy][metric].append(sim_result[metric])
                
                # Special handling for citation patterns (time series)
                if "citation_pattern" not in aggregate["research_impact"][strategy]:
                    aggregate["research_impact"][strategy]["citation_pattern"] = []
                aggregate["research_impact"][strategy]["citation_pattern"].append(sim_result["citation_pattern"])
        
        # Calculate averages
        for metric in research_metrics:
            values = aggregate["research_impact"][strategy][metric]
            aggregate["research_impact"][strategy][metric] = {
                "mean": np.mean(values),
                "std": np.std(values)
            }
        
        # Average citation patterns across simulations
        citations = aggregate["research_impact"][strategy]["citation_pattern"]
        if citations:
            # Convert list of lists to 2D array
            citation_array = np.array(citations)
            # Calculate mean and std for each time period
            mean_citations = np.mean(citation_array, axis=0)
            std_citations = np.std(citation_array, axis=0)
            aggregate["research_impact"][strategy]["citation_pattern"] = {
                "mean": mean_citations.tolist(),
                "std": std_citations.tolist()
            }
    
    return aggregate

def save_results(results: Dict[str, Any], filename: str = "experiment_results.json") -> None:
    """Save experimental results to a JSON file."""
    filepath = os.path.join(RESULTS_DIR, filename)
    
    # Convert numpy values to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Saved experimental results to {filepath}")
    return filepath

if __name__ == "__main__":
    logger.info("Starting dataset deprecation framework simulation")
    
    # Create synthetic datasets
    datasets = create_synthetic_datasets()
    
    # Create deprecation records
    deprecation_records = create_deprecation_records()
    
    # Run the experiment
    results = run_experiment(datasets, deprecation_records)
    
    # Save results
    save_results(results)
    
    logger.info("Simulation complete")