"""
Contextual Dataset Deprecation Framework Implementation

This module implements the core components of the Contextual Dataset Deprecation Framework:
1. Tiered Warning System
2. Notification System
3. Context-Preserving Deprecation
4. Alternative Recommendation System
5. Transparent Versioning System
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from typing import List, Dict, Any, Tuple, Optional, Set, Union
import hashlib
from enum import Enum
import time
import random
from difflib import SequenceMatcher

from experimental_design import WarningLevel, DeprecationRecord, SyntheticDataset, DatasetVersion, DeprecationStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log.txt')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("deprecation_framework")

class User:
    """Class representing a simulated user of datasets."""
    
    def __init__(
        self,
        user_id: str,
        usage_history: Dict[str, datetime] = None,
        citation_history: Dict[str, List[str]] = None,
        response_time_mean: float = 5.0,  # Average days to respond
        response_time_std: float = 2.0,
        adoption_probability: float = 0.5,  # Base probability of adopting alternatives
        research_focus: str = "general"
    ):
        self.user_id = user_id
        self.usage_history = usage_history or {}
        self.citation_history = citation_history or {}
        self.response_time_mean = response_time_mean
        self.response_time_std = response_time_std
        self.adoption_probability = adoption_probability
        self.research_focus = research_focus
        self.notifications = []
        
    def download_dataset(self, dataset_id: str) -> None:
        """Simulate downloading a dataset."""
        self.usage_history[dataset_id] = datetime.now()
        logger.debug(f"User {self.user_id} downloaded dataset {dataset_id}")
    
    def cite_dataset(self, dataset_id: str, paper_id: str) -> None:
        """Simulate citing a dataset in a paper."""
        if dataset_id not in self.citation_history:
            self.citation_history[dataset_id] = []
        self.citation_history[dataset_id].append(paper_id)
        logger.debug(f"User {self.user_id} cited dataset {dataset_id} in paper {paper_id}")
    
    def receive_notification(self, notification: Dict[str, Any]) -> None:
        """Receive a notification about a dataset status change."""
        self.notifications.append(notification)
        logger.debug(f"User {self.user_id} received notification: {notification['subject']}")
    
    def acknowledge_notification(self, notification_id: str) -> float:
        """
        Acknowledge a notification and return the time taken to respond (in days).
        """
        # Simulate time to acknowledge based on user's characteristics
        response_time = max(0.1, np.random.normal(
            self.response_time_mean, self.response_time_std
        ))
        
        logger.debug(f"User {self.user_id} acknowledged notification {notification_id} after {response_time:.2f} days")
        return response_time
    
    def decide_alternative_adoption(
        self, 
        dataset_id: str, 
        alternatives: List[str],
        warning_level: WarningLevel
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide whether to adopt an alternative dataset.
        
        Returns:
            Tuple of (adopted, alternative_id)
        """
        # Base probability is affected by warning level
        level_factor = 1.0
        if warning_level == WarningLevel.CAUTION:
            level_factor = 1.2
        elif warning_level == WarningLevel.LIMITED_USE:
            level_factor = 1.5
        elif warning_level == WarningLevel.DEPRECATED:
            level_factor = 2.0
        
        # Final adoption probability
        adoption_prob = min(0.95, self.adoption_probability * level_factor)
        
        # Decide whether to adopt
        will_adopt = random.random() < adoption_prob
        
        if will_adopt and alternatives:
            # Select an alternative (simplified for simulation)
            alternative_id = random.choice(alternatives)
            logger.debug(f"User {self.user_id} decided to adopt alternative {alternative_id} for {dataset_id}")
            return True, alternative_id
        
        logger.debug(f"User {self.user_id} decided not to adopt alternatives for {dataset_id}")
        return will_adopt, None

class ContextualDeprecationFramework:
    """Implementation of the Contextual Dataset Deprecation Framework."""
    
    def __init__(
        self,
        strategy: DeprecationStrategy = DeprecationStrategy.FULL,
        datasets: Dict[str, SyntheticDataset] = None,
        deprecation_records: Dict[str, DeprecationRecord] = None,
        users: List[User] = None,
        data_dir: str = None
    ):
        self.strategy = strategy
        self.datasets = datasets or {}
        self.deprecation_records = deprecation_records or {}
        self.users = users or []
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'framework_data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Tracking for evaluation
        self.notification_history = []
        self.access_logs = []
        self.recommendation_logs = []
        self.version_history = {}
        
        logger.info(f"Initialized ContextualDeprecationFramework with strategy {strategy.name}")
    
    def apply_warning_level(
        self, 
        dataset_id: str, 
        warning_level: WarningLevel,
        issue_description: str,
        evidence_links: List[str] = None,
        affected_groups: List[str] = None,
        recommended_alternatives: List[str] = None
    ) -> DeprecationRecord:
        """
        Apply a warning level to a dataset, creating or updating its deprecation record.
        
        Args:
            dataset_id: ID of the dataset
            warning_level: New warning level
            issue_description: Description of the issue
            evidence_links: Links to evidence supporting the warning
            affected_groups: Groups affected by the issue
            recommended_alternatives: Alternative datasets to recommend
            
        Returns:
            The newly created or updated deprecation record
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        evidence_links = evidence_links or []
        affected_groups = affected_groups or []
        recommended_alternatives = recommended_alternatives or []
        
        # Check if record exists
        old_record = self.deprecation_records.get(dataset_id)
        old_level = WarningLevel.NONE if old_record is None else old_record.warning_level
        
        # Create new record
        record = DeprecationRecord(
            dataset_id=dataset_id,
            warning_level=warning_level,
            issue_description=issue_description,
            evidence_links=evidence_links,
            affected_groups=affected_groups,
            recommended_alternatives=recommended_alternatives,
            timestamp=datetime.now()
        )
        
        # Save the record
        self.deprecation_records[dataset_id] = record
        
        # If the warning level changed, update version history and notify users
        if old_level != warning_level:
            self._update_version_history(dataset_id, old_level, warning_level)
            self._notify_users(dataset_id, old_level, warning_level)
        
        logger.info(f"Applied warning level {warning_level.name} to dataset {dataset_id}")
        return record
    
    def _update_version_history(
        self, 
        dataset_id: str, 
        old_level: WarningLevel, 
        new_level: WarningLevel
    ) -> None:
        """Update the version history of a dataset."""
        dataset = self.datasets.get(dataset_id)
        if not dataset:
            logger.warning(f"Cannot update version history: Dataset {dataset_id} not found")
            return
        
        if dataset_id not in self.version_history:
            self.version_history[dataset_id] = []
        
        # Create a new version
        version = DatasetVersion(
            dataset_id=dataset_id,
            version=f"{dataset.version}-{len(self.version_history[dataset_id]) + 1}",
            content_hash=dataset.content_hash,
            warning_level=new_level,
            changes=f"Warning level changed from {old_level.name} to {new_level.name}",
            justification=self.deprecation_records[dataset_id].issue_description,
            timestamp=datetime.now()
        )
        
        self.version_history[dataset_id].append(version)
        logger.debug(f"Updated version history for dataset {dataset_id}: {old_level.name} -> {new_level.name}")
    
    def _notify_users(
        self, 
        dataset_id: str, 
        old_level: WarningLevel, 
        new_level: WarningLevel
    ) -> None:
        """Notify users about a change in dataset warning level."""
        # Skip if using control strategy
        if self.strategy == DeprecationStrategy.CONTROL:
            return
        
        # Find users who have used this dataset
        affected_users = [user for user in self.users 
                         if dataset_id in user.usage_history or 
                            dataset_id in user.citation_history]
        
        if not affected_users:
            logger.debug(f"No users to notify about dataset {dataset_id}")
            return
        
        # Create notification
        record = self.deprecation_records[dataset_id]
        notification = {
            "id": f"notif_{dataset_id}_{int(time.time())}",
            "dataset_id": dataset_id,
            "old_level": old_level.name,
            "new_level": new_level.name,
            "timestamp": datetime.now().isoformat(),
            "subject": f"Dataset Status Change: {dataset_id}",
            "message": self._generate_notification_message(dataset_id, old_level, new_level),
            "alternatives": record.recommended_alternatives
        }
        
        # Send to users
        for user in affected_users:
            user.receive_notification(notification)
        
        # Store in history
        self.notification_history.append({
            "notification": notification,
            "user_count": len(affected_users)
        })
        
        logger.info(f"Notified {len(affected_users)} users about status change for dataset {dataset_id}")
    
    def _generate_notification_message(
        self, 
        dataset_id: str, 
        old_level: WarningLevel, 
        new_level: WarningLevel
    ) -> str:
        """Generate a notification message based on the warning level change."""
        record = self.deprecation_records[dataset_id]
        
        if self.strategy == DeprecationStrategy.BASIC:
            # Basic notifications
            return f"The status of dataset '{dataset_id}' has changed from {old_level.name} to {new_level.name}. Please review its usage in your research."
        
        else:  # FULL strategy
            # More comprehensive notifications with context
            message = f"IMPORTANT: The status of dataset '{dataset_id}' has changed from {old_level.name} to {new_level.name}.\n\n"
            message += f"Issue: {record.issue_description}\n\n"
            
            if record.affected_groups:
                message += f"Affected Groups: {', '.join(record.affected_groups)}\n\n"
            
            if record.recommended_alternatives:
                message += f"Recommended Alternatives: {', '.join(record.recommended_alternatives)}\n\n"
            
            if record.evidence_links:
                message += f"For more information, see: {', '.join(record.evidence_links)}\n\n"
            
            message += "Please review your use of this dataset in light of this status change."
            return message
    
    def check_access_permission(
        self, 
        user_id: str, 
        dataset_id: str, 
        purpose: str = None
    ) -> bool:
        """
        Check if a user is allowed to access a dataset based on its warning level.
        
        Args:
            user_id: ID of the user requesting access
            dataset_id: ID of the dataset
            purpose: Stated purpose for accessing the dataset
            
        Returns:
            True if access is granted, False otherwise
        """
        # Find the user
        user = next((u for u in self.users if u.user_id == user_id), None)
        if not user:
            logger.warning(f"User {user_id} not found")
            return False
        
        # Check if dataset exists
        if dataset_id not in self.datasets:
            logger.warning(f"Dataset {dataset_id} not found")
            return False
        
        # Get deprecation record
        record = self.deprecation_records.get(dataset_id)
        
        # No record means no restrictions
        if not record:
            logger.debug(f"Access granted to {user_id} for dataset {dataset_id} (no restrictions)")
            self._log_access(user_id, dataset_id, True, "No restrictions")
            return True
        
        # Control strategy doesn't restrict access
        if self.strategy == DeprecationStrategy.CONTROL:
            logger.debug(f"Access granted to {user_id} for dataset {dataset_id} (control strategy)")
            self._log_access(user_id, dataset_id, True, "Control strategy")
            return True
        
        # Basic strategy only restricts fully deprecated datasets
        if self.strategy == DeprecationStrategy.BASIC:
            if record.warning_level == WarningLevel.DEPRECATED:
                logger.debug(f"Access denied to {user_id} for deprecated dataset {dataset_id} (basic strategy)")
                self._log_access(user_id, dataset_id, False, "Deprecated dataset (basic)")
                return False
            else:
                logger.debug(f"Access granted to {user_id} for dataset {dataset_id} (basic strategy)")
                self._log_access(user_id, dataset_id, True, "Not fully deprecated (basic)")
                return True
        
        # Full strategy implements graded access control
        if self.strategy == DeprecationStrategy.FULL:
            # Caution level: unrestricted access
            if record.warning_level == WarningLevel.CAUTION:
                logger.debug(f"Access granted to {user_id} for cautioned dataset {dataset_id}")
                self._log_access(user_id, dataset_id, True, "Caution level (full)")
                return True
            
            # Limited Use: check if purpose is provided
            elif record.warning_level == WarningLevel.LIMITED_USE:
                if not purpose:
                    logger.debug(f"Access denied to {user_id} for limited use dataset {dataset_id}: no purpose specified")
                    self._log_access(user_id, dataset_id, False, "Limited use - no purpose (full)")
                    return False
                
                # Simple approval for simulation purposes
                logger.debug(f"Access granted to {user_id} for limited use dataset {dataset_id} with purpose: {purpose}")
                self._log_access(user_id, dataset_id, True, f"Limited use with purpose (full): {purpose}")
                return True
            
            # Deprecated: restricted to specific purposes
            elif record.warning_level == WarningLevel.DEPRECATED:
                valid_purposes = ["historical_analysis", "ethical_research", "bias_mitigation"]
                
                if not purpose:
                    logger.debug(f"Access denied to {user_id} for deprecated dataset {dataset_id}: no purpose specified")
                    self._log_access(user_id, dataset_id, False, "Deprecated - no purpose (full)")
                    return False
                
                # Check if purpose is valid
                purpose_lower = purpose.lower().replace(" ", "_")
                for valid_purpose in valid_purposes:
                    if valid_purpose in purpose_lower:
                        logger.debug(f"Access granted to {user_id} for deprecated dataset {dataset_id} with valid purpose: {purpose}")
                        self._log_access(user_id, dataset_id, True, f"Deprecated with valid purpose (full): {purpose}")
                        return True
                
                logger.debug(f"Access denied to {user_id} for deprecated dataset {dataset_id}: invalid purpose: {purpose}")
                self._log_access(user_id, dataset_id, False, f"Deprecated with invalid purpose (full): {purpose}")
                return False
        
        # Default deny
        logger.debug(f"Access denied to {user_id} for dataset {dataset_id} (default)")
        self._log_access(user_id, dataset_id, False, "Default denial")
        return False
    
    def _log_access(
        self, 
        user_id: str, 
        dataset_id: str, 
        granted: bool, 
        reason: str
    ) -> None:
        """Log dataset access attempts for evaluation."""
        self.access_logs.append({
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "dataset_id": dataset_id,
            "granted": granted,
            "reason": reason
        })
    
    def recommend_alternatives(self, dataset_id: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Recommend alternative datasets for a given dataset.
        
        Args:
            dataset_id: ID of the dataset
            top_n: Number of top recommendations to return
            
        Returns:
            List of recommendations with similarity scores
        """
        # Skip if using control strategy
        if self.strategy == DeprecationStrategy.CONTROL:
            return []
        
        # Check if dataset exists
        if dataset_id not in self.datasets:
            logger.warning(f"Dataset {dataset_id} not found")
            return []
        
        target_dataset = self.datasets[dataset_id]
        
        # Basic strategy provides static recommendations from the record
        if self.strategy == DeprecationStrategy.BASIC:
            record = self.deprecation_records.get(dataset_id)
            if not record or not record.recommended_alternatives:
                return []
            
            recommendations = []
            for alt_id in record.recommended_alternatives:
                if alt_id in self.datasets:
                    recommendations.append({
                        "dataset_id": alt_id,
                        "similarity": 0.8,  # Arbitrary score for basic strategy
                        "reason": "Recommended alternative"
                    })
            
            return recommendations[:top_n]
        
        # Full strategy provides dynamic recommendations based on dataset properties
        elif self.strategy == DeprecationStrategy.FULL:
            # Find similar datasets
            similarities = []
            for other_id, other_dataset in self.datasets.items():
                if other_id == dataset_id:
                    continue
                
                # Calculate similarity based on dataset properties
                sim_score = self._calculate_similarity(target_dataset, other_dataset)
                
                # Check if the alternative has ethical issues
                has_issues = other_id in self.deprecation_records
                ethical_improvement = 0.0
                
                if has_issues:
                    alt_record = self.deprecation_records[other_id]
                    if alt_record.warning_level.value < WarningLevel.LIMITED_USE.value:
                        ethical_improvement = 0.2  # Moderate improvement
                else:
                    ethical_improvement = 0.4  # Significant improvement
                
                # Adjust similarity score to favor datasets with fewer ethical issues
                adjusted_score = sim_score + ethical_improvement
                
                similarities.append({
                    "dataset_id": other_id,
                    "similarity": adjusted_score,
                    "reason": self._get_recommendation_reason(target_dataset, other_dataset, sim_score, has_issues)
                })
            
            # Sort by similarity and return top N
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            recommendations = similarities[:top_n]
            
            # Log recommendations
            self._log_recommendation(dataset_id, recommendations)
            
            return recommendations
        
        # Default: return empty list
        return []
    
    def _calculate_similarity(
        self, 
        dataset1: SyntheticDataset, 
        dataset2: SyntheticDataset
    ) -> float:
        """
        Calculate similarity between two datasets.
        
        In a real implementation, this would involve sophisticated analysis of feature spaces,
        data distributions, and task compatibility. For this simulation, we use simplified metrics.
        """
        similarity = 0.0
        
        # Task type compatibility (most important)
        if dataset1.task_type == dataset2.task_type:
            similarity += 0.5
        
        # Feature space similarity (based on number of features)
        feature_ratio = min(dataset1.n_features, dataset2.n_features) / max(dataset1.n_features, dataset2.n_features)
        similarity += 0.2 * feature_ratio
        
        # Data size similarity
        size_ratio = min(dataset1.n_samples, dataset2.n_samples) / max(dataset1.n_samples, dataset2.n_samples)
        similarity += 0.1 * size_ratio
        
        # Bias level (lower is better)
        bias_diff = abs(dataset1.bias_level - dataset2.bias_level)
        bias_factor = max(0, 1 - bias_diff)
        similarity += 0.2 * bias_factor
        
        return min(1.0, similarity)
    
    def _get_recommendation_reason(
        self, 
        target: SyntheticDataset, 
        alternative: SyntheticDataset, 
        similarity: float,
        has_issues: bool
    ) -> str:
        """Generate a reason for the recommendation."""
        reasons = []
        
        # Task compatibility
        if target.task_type == alternative.task_type:
            reasons.append(f"Compatible {target.task_type} task")
        
        # Feature space
        if alternative.n_features >= target.n_features:
            reasons.append(f"Similar or richer feature space")
        else:
            feature_ratio = alternative.n_features / target.n_features
            if feature_ratio > 0.8:
                reasons.append("Comparable feature space")
        
        # Data size
        if alternative.n_samples >= target.n_samples:
            reasons.append("Equal or larger dataset size")
        
        # Bias level
        if alternative.bias_level < target.bias_level:
            reasons.append("Reduced bias level")
        
        # Ethical consideration
        if not has_issues:
            reasons.append("No known ethical issues")
        
        if not reasons:
            return "Compatible alternative dataset"
        
        return "; ".join(reasons)
    
    def _log_recommendation(
        self, 
        dataset_id: str, 
        recommendations: List[Dict[str, Any]]
    ) -> None:
        """Log dataset recommendations for evaluation."""
        self.recommendation_logs.append({
            "timestamp": datetime.now().isoformat(),
            "dataset_id": dataset_id,
            "recommendations": [
                {
                    "dataset_id": rec["dataset_id"],
                    "similarity": rec["similarity"],
                    "reason": rec["reason"]
                }
                for rec in recommendations
            ]
        })
    
    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get metadata for a dataset, including its deprecation status.
        
        This implements the context-preserving deprecation by maintaining
        metadata access even for deprecated datasets.
        """
        if dataset_id not in self.datasets:
            logger.warning(f"Dataset {dataset_id} not found")
            return {}
        
        dataset = self.datasets[dataset_id]
        
        # Basic metadata
        metadata = {
            "dataset_id": dataset.dataset_id,
            "task_type": dataset.task_type,
            "n_samples": dataset.n_samples,
            "n_features": dataset.n_features,
            "version": dataset.version,
            "warning_level": "NONE"
        }
        
        # Add deprecation information if available
        record = self.deprecation_records.get(dataset_id)
        if record:
            metadata["warning_level"] = record.warning_level.name
            metadata["issue_description"] = record.issue_description
            
            if self.strategy != DeprecationStrategy.CONTROL:
                metadata["affected_groups"] = record.affected_groups
                metadata["evidence_links"] = record.evidence_links
                
                if self.strategy == DeprecationStrategy.FULL:
                    metadata["recommended_alternatives"] = record.recommended_alternatives
        
        # Add version history if using full framework
        if self.strategy == DeprecationStrategy.FULL and dataset_id in self.version_history:
            metadata["version_history"] = [
                version.to_dict() for version in self.version_history[dataset_id]
            ]
        
        return metadata
    
    def evaluate_user_responses(self) -> Dict[str, Any]:
        """
        Evaluate user responses to dataset deprecation for each strategy.
        
        Returns:
            Dictionary of evaluation metrics
        """
        results = {
            "acknowledgment_times": [],
            "alternative_adoption_rates": {},
            "continued_usage_rates": {}
        }
        
        # Analyze all notifications
        for notif_entry in self.notification_history:
            notification = notif_entry["notification"]
            dataset_id = notification["dataset_id"]
            
            # Find users who received this notification
            affected_users = [
                user for user in self.users 
                if any(n["id"] == notification["id"] for n in user.notifications)
            ]
            
            # Calculate acknowledgment times
            ack_times = []
            adoptions = []
            continued_usage = []
            
            for user in affected_users:
                # Simulate acknowledgment
                ack_time = user.acknowledge_notification(notification["id"])
                ack_times.append(ack_time)
                
                # Simulate adoption decision
                record = self.deprecation_records.get(dataset_id)
                if record:
                    adopted, alt_id = user.decide_alternative_adoption(
                        dataset_id, 
                        record.recommended_alternatives,
                        record.warning_level
                    )
                    adoptions.append(int(adopted))
                    continued_usage.append(int(not adopted))
            
            # Store results
            if ack_times:
                results["acknowledgment_times"].extend(ack_times)
            
            if dataset_id not in results["alternative_adoption_rates"]:
                results["alternative_adoption_rates"][dataset_id] = []
                results["continued_usage_rates"][dataset_id] = []
            
            if adoptions:
                adoption_rate = sum(adoptions) / len(adoptions)
                results["alternative_adoption_rates"][dataset_id].append(adoption_rate)
            
            if continued_usage:
                continued_rate = sum(continued_usage) / len(continued_usage)
                results["continued_usage_rates"][dataset_id].append(continued_rate)
        
        # Calculate aggregate metrics
        aggregate_results = {
            "mean_acknowledgment_time": np.mean(results["acknowledgment_times"]) if results["acknowledgment_times"] else None,
            "std_acknowledgment_time": np.std(results["acknowledgment_times"]) if results["acknowledgment_times"] else None,
            "mean_adoption_rate": {},
            "mean_continued_usage_rate": {}
        }
        
        for dataset_id, rates in results["alternative_adoption_rates"].items():
            if rates:
                aggregate_results["mean_adoption_rate"][dataset_id] = np.mean(rates)
        
        for dataset_id, rates in results["continued_usage_rates"].items():
            if rates:
                aggregate_results["mean_continued_usage_rate"][dataset_id] = np.mean(rates)
        
        return aggregate_results
    
    def evaluate_system_performance(self) -> Dict[str, Any]:
        """
        Evaluate system performance metrics for the deprecation framework.
        
        Returns:
            Dictionary of evaluation metrics
        """
        results = {
            "access_control": {
                "granted_count": 0,
                "denied_count": 0,
                "grant_rate": 0.0
            },
            "recommendation": {
                "recommendation_count": 0,
                "average_alternatives": 0.0,
                "diversity_score": 0.0
            }
        }
        
        # Analyze access logs
        if self.access_logs:
            granted = sum(1 for log in self.access_logs if log["granted"])
            denied = len(self.access_logs) - granted
            
            results["access_control"]["granted_count"] = granted
            results["access_control"]["denied_count"] = denied
            results["access_control"]["grant_rate"] = granted / len(self.access_logs) if self.access_logs else 0.0
        
        # Analyze recommendation logs
        if self.recommendation_logs:
            results["recommendation"]["recommendation_count"] = len(self.recommendation_logs)
            
            # Calculate average number of alternatives recommended
            alt_counts = [len(log["recommendations"]) for log in self.recommendation_logs]
            results["recommendation"]["average_alternatives"] = np.mean(alt_counts) if alt_counts else 0.0
            
            # Calculate diversity of recommendations
            unique_alternatives = set()
            for log in self.recommendation_logs:
                for rec in log["recommendations"]:
                    unique_alternatives.add(rec["dataset_id"])
            
            # Diversity score: ratio of unique alternatives to total recommendations
            total_recommendations = sum(len(log["recommendations"]) for log in self.recommendation_logs)
            results["recommendation"]["diversity_score"] = len(unique_alternatives) / total_recommendations if total_recommendations else 0.0
        
        return results
    
    def save_evaluation_data(self, output_dir: str = None) -> str:
        """
        Save all evaluation data to disk.
        
        Args:
            output_dir: Directory to save data to. If None, uses the data_dir.
            
        Returns:
            Path to the saved data directory
        """
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, f"eval_{self.strategy.name.lower()}_{int(time.time())}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save notification history
        with open(os.path.join(output_dir, "notification_history.json"), "w") as f:
            json.dump(self.notification_history, f, indent=2)
        
        # Save access logs
        with open(os.path.join(output_dir, "access_logs.json"), "w") as f:
            json.dump(self.access_logs, f, indent=2)
        
        # Save recommendation logs
        with open(os.path.join(output_dir, "recommendation_logs.json"), "w") as f:
            json.dump(self.recommendation_logs, f, indent=2)
        
        # Save version history
        serialized_version_history = {}
        for dataset_id, versions in self.version_history.items():
            serialized_version_history[dataset_id] = [v.to_dict() for v in versions]
        
        with open(os.path.join(output_dir, "version_history.json"), "w") as f:
            json.dump(serialized_version_history, f, indent=2)
        
        # Save user response evaluation
        user_response_eval = self.evaluate_user_responses()
        with open(os.path.join(output_dir, "user_response_evaluation.json"), "w") as f:
            json.dump(user_response_eval, f, indent=2)
        
        # Save system performance evaluation
        system_performance_eval = self.evaluate_system_performance()
        with open(os.path.join(output_dir, "system_performance_evaluation.json"), "w") as f:
            json.dump(system_performance_eval, f, indent=2)
        
        logger.info(f"Saved evaluation data to {output_dir}")
        return output_dir

def create_simulated_users(n_users: int = 50) -> List[User]:
    """Create a population of simulated users with varying characteristics."""
    users = []
    
    research_focuses = ["general", "classification", "regression", "ethics", "bias_mitigation"]
    
    for i in range(n_users):
        user_id = f"user_{i+1}"
        
        # Vary response characteristics
        response_time_mean = np.random.uniform(1.0, 10.0)  # 1-10 days
        response_time_std = np.random.uniform(0.5, 3.0)
        adoption_probability = np.random.uniform(0.2, 0.8)
        research_focus = np.random.choice(research_focuses)
        
        users.append(User(
            user_id=user_id,
            response_time_mean=response_time_mean,
            response_time_std=response_time_std,
            adoption_probability=adoption_probability,
            research_focus=research_focus
        ))
    
    return users

def simulate_initial_dataset_usage(
    users: List[User], 
    datasets: Dict[str, SyntheticDataset]
) -> None:
    """Simulate initial usage and citation of datasets by users."""
    for user in users:
        # Randomly select 1-3 datasets to use
        n_datasets = np.random.randint(1, 4)
        selected_datasets = np.random.choice(list(datasets.keys()), size=min(n_datasets, len(datasets)), replace=False)
        
        for dataset_id in selected_datasets:
            # Simulate download
            user.download_dataset(dataset_id)
            
            # Simulate citation in 0-2 papers
            n_papers = np.random.randint(0, 3)
            for i in range(n_papers):
                paper_id = f"paper_{user.user_id}_{i+1}"
                user.cite_dataset(dataset_id, paper_id)

def run_framework_simulation(
    strategy: DeprecationStrategy,
    datasets: Dict[str, SyntheticDataset],
    deprecation_records: Dict[str, DeprecationRecord],
    n_users: int = 50
) -> ContextualDeprecationFramework:
    """
    Run a simulation of the Contextual Dataset Deprecation Framework.
    
    Args:
        strategy: The deprecation strategy to use
        datasets: Dictionary of synthetic datasets
        deprecation_records: Dictionary of deprecation records
        n_users: Number of simulated users
        
    Returns:
        The simulation framework instance
    """
    logger.info(f"Starting framework simulation with strategy: {strategy.name}")
    
    # Create users
    users = create_simulated_users(n_users)
    
    # Simulate initial dataset usage
    simulate_initial_dataset_usage(users, datasets)
    
    # Initialize framework
    framework = ContextualDeprecationFramework(
        strategy=strategy,
        datasets=datasets,
        deprecation_records=deprecation_records,
        users=users
    )
    
    # Apply warning levels based on deprecation records
    for dataset_id, record in deprecation_records.items():
        if dataset_id in datasets:
            framework.apply_warning_level(
                dataset_id=dataset_id,
                warning_level=record.warning_level,
                issue_description=record.issue_description,
                evidence_links=record.evidence_links,
                affected_groups=record.affected_groups,
                recommended_alternatives=record.recommended_alternatives
            )
    
    # Simulate access attempts
    for user in users:
        for dataset_id in datasets:
            # 50% chance of attempting access for each dataset
            if np.random.random() < 0.5:
                purpose = None
                
                # Add purpose for datasets that might require it
                if dataset_id in deprecation_records:
                    record = deprecation_records[dataset_id]
                    if record.warning_level in [WarningLevel.LIMITED_USE, WarningLevel.DEPRECATED]:
                        # Generate a purpose based on user's research focus
                        if user.research_focus == "ethics":
                            purpose = "Ethical analysis of dataset biases"
                        elif user.research_focus == "bias_mitigation":
                            purpose = "Research on mitigating biases in ML models"
                        else:
                            purpose = "General research purposes"
                
                # Attempt access
                framework.check_access_permission(user.user_id, dataset_id, purpose)
    
    # Simulate recommendation requests
    for dataset_id in deprecation_records:
        if dataset_id in datasets:
            # Request recommendations
            framework.recommend_alternatives(dataset_id)
    
    # Save evaluation data
    framework.save_evaluation_data()
    
    logger.info(f"Completed framework simulation with strategy: {strategy.name}")
    return framework

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from claude_code.dataset_generator import load_datasets, load_deprecation_records
    
    # Load datasets and deprecation records
    datasets = load_datasets()
    deprecation_records = load_deprecation_records()
    
    # Run simulation for each strategy
    for strategy in DeprecationStrategy:
        run_framework_simulation(strategy, datasets, deprecation_records)