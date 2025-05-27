"""
Contextual Metadata Schema (CMS) module

This module implements the standardized metadata schema for datasets in the ContextBench framework.
It provides classes and functions to create, validate, and manipulate metadata records for datasets.
"""

import json
import jsonschema
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import os

# Define the schema as per the proposal
METADATA_SCHEMA = {
    "type": "object",
    "required": ["dataset_id", "provenance", "domain_tags"],
    "properties": {
        "dataset_id": {
            "type": "string",
            "description": "Unique identifier for the dataset"
        },
        "provenance": {
            "type": "object",
            "required": ["source", "collection_date"],
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Origin or creator of the dataset"
                },
                "collection_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Date when the dataset was collected or published"
                },
                "version": {
                    "type": "string",
                    "description": "Version of the dataset if applicable"
                }
            }
        },
        "demographics": {
            "type": "object",
            "description": "Distributions over sensitive attributes",
            "additionalProperties": {
                "type": "object",
                "additionalProperties": {
                    "type": "number"
                }
            }
        },
        "licensing": {
            "type": "object",
            "properties": {
                "license_type": {
                    "type": "string",
                    "description": "Type of license (e.g., CC-BY-SA, MIT)"
                },
                "license_url": {
                    "type": "string",
                    "format": "uri",
                    "description": "URL to the license text"
                }
            }
        },
        "deprecation_status": {
            "type": "object",
            "properties": {
                "is_deprecated": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether the dataset is deprecated"
                },
                "rationale": {
                    "type": "string",
                    "description": "Reason for deprecation if applicable"
                },
                "deprecated_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Date when the dataset was deprecated"
                },
                "replacement_dataset": {
                    "type": "string",
                    "description": "Identifier of the dataset that replaces this one, if any"
                }
            }
        },
        "domain_tags": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "List of domain tags (e.g., healthcare, finance)"
        },
        "features": {
            "type": "object",
            "description": "Information about the features in the dataset",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["numerical", "categorical", "text", "image", "audio", "video", "other"]
                    },
                    "sensitive": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether this feature is considered sensitive (e.g., race, gender)"
                    }
                }
            }
        }
    }
}


class ContextualMetadata:
    """
    Class to handle dataset metadata in ContextBench.
    """
    
    def __init__(self, dataset_id: str, **kwargs):
        """
        Initialize a metadata record for a dataset.
        
        Args:
            dataset_id: Unique identifier for the dataset
            **kwargs: Additional metadata fields
        """
        self.data = {
            "dataset_id": dataset_id,
            "provenance": kwargs.get("provenance", {
                "source": "Unknown",
                "collection_date": datetime.now().strftime("%Y-%m-%d")
            }),
            "domain_tags": kwargs.get("domain_tags", []),
        }
        
        # Add optional fields if provided
        if "demographics" in kwargs:
            self.data["demographics"] = kwargs["demographics"]
        
        if "licensing" in kwargs:
            self.data["licensing"] = kwargs["licensing"]
        
        if "deprecation_status" in kwargs:
            self.data["deprecation_status"] = kwargs["deprecation_status"]
        else:
            self.data["deprecation_status"] = {
                "is_deprecated": False
            }
        
        if "features" in kwargs:
            self.data["features"] = kwargs["features"]
    
    def validate(self) -> bool:
        """
        Validate the metadata against the schema.
        
        Returns:
            bool: True if valid, raises an exception otherwise
        """
        try:
            jsonschema.validate(instance=self.data, schema=METADATA_SCHEMA)
            return True
        except jsonschema.exceptions.ValidationError as e:
            raise ValueError(f"Invalid metadata: {e}")
    
    def to_json(self) -> str:
        """
        Convert the metadata to JSON string.
        
        Returns:
            str: JSON representation of the metadata
        """
        return json.dumps(self.data, indent=2)
    
    def save(self, directory: str) -> str:
        """
        Save the metadata to a JSON file.
        
        Args:
            directory: Directory to save the file in
            
        Returns:
            str: Path to the saved file
        """
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"{self.data['dataset_id']}_metadata.json")
        
        with open(filename, 'w') as f:
            f.write(self.to_json())
        
        return filename
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ContextualMetadata':
        """
        Create a ContextualMetadata instance from a JSON string.
        
        Args:
            json_str: JSON string representation of metadata
            
        Returns:
            ContextualMetadata: Instance created from the JSON
        """
        data = json.loads(json_str)
        dataset_id = data.pop("dataset_id")
        return cls(dataset_id, **data)
    
    @classmethod
    def load(cls, filename: str) -> 'ContextualMetadata':
        """
        Load metadata from a JSON file.
        
        Args:
            filename: Path to the JSON file
            
        Returns:
            ContextualMetadata: Instance loaded from the file
        """
        with open(filename, 'r') as f:
            return cls.from_json(f.read())


# Example metadata for datasets used in the experiments
def create_example_metadata():
    """
    Create example metadata for the datasets used in the experiments.
    
    Returns:
        dict: Dictionary mapping dataset IDs to their metadata objects
    """
    metadata = {}
    
    # ImageNet metadata
    imagenet_metadata = ContextualMetadata(
        dataset_id="imagenet",
        provenance={
            "source": "ImageNet",
            "collection_date": "2012-01-01",
            "version": "ILSVRC 2012"
        },
        domain_tags=["vision", "classification", "object_recognition"],
        licensing={
            "license_type": "Custom (non-commercial)",
            "license_url": "https://image-net.org/download.php"
        },
        features={
            "image": {
                "description": "RGB image",
                "type": "image"
            },
            "label": {
                "description": "Class label (1000 categories)",
                "type": "categorical"
            }
        }
    )
    metadata["imagenet"] = imagenet_metadata
    
    # Adult Census Income metadata
    adult_metadata = ContextualMetadata(
        dataset_id="adult",
        provenance={
            "source": "UCI Machine Learning Repository",
            "collection_date": "1996-05-01",
            "version": "1.0"
        },
        domain_tags=["tabular", "classification", "census", "finance"],
        demographics={
            "sex": {
                "Male": 0.6724,
                "Female": 0.3276
            },
            "race": {
                "White": 0.8551,
                "Black": 0.0956,
                "Asian-Pac-Islander": 0.0318,
                "Amer-Indian-Eskimo": 0.0097,
                "Other": 0.0078
            }
        },
        licensing={
            "license_type": "CC-BY",
            "license_url": "https://archive.ics.uci.edu/ml/datasets/adult"
        },
        features={
            "age": {
                "description": "Age in years",
                "type": "numerical",
                "sensitive": True
            },
            "workclass": {
                "description": "Type of employer",
                "type": "categorical"
            },
            "education": {
                "description": "Highest education level",
                "type": "categorical"
            },
            "education-num": {
                "description": "Education level as numerical value",
                "type": "numerical"
            },
            "marital-status": {
                "description": "Marital status",
                "type": "categorical"
            },
            "occupation": {
                "description": "Occupation category",
                "type": "categorical"
            },
            "relationship": {
                "description": "Relationship status",
                "type": "categorical"
            },
            "race": {
                "description": "Race category",
                "type": "categorical",
                "sensitive": True
            },
            "sex": {
                "description": "Gender",
                "type": "categorical",
                "sensitive": True
            },
            "capital-gain": {
                "description": "Capital gains in USD",
                "type": "numerical"
            },
            "capital-loss": {
                "description": "Capital losses in USD",
                "type": "numerical"
            },
            "hours-per-week": {
                "description": "Work hours per week",
                "type": "numerical"
            },
            "native-country": {
                "description": "Country of origin",
                "type": "categorical"
            },
            "income": {
                "description": "Income category (>50K or <=50K)",
                "type": "categorical"
            }
        }
    )
    metadata["adult"] = adult_metadata
    
    # MNLI (Multi-Genre Natural Language Inference) metadata
    mnli_metadata = ContextualMetadata(
        dataset_id="mnli",
        provenance={
            "source": "GLUE Benchmark",
            "collection_date": "2018-01-01",
            "version": "1.0"
        },
        domain_tags=["nlp", "text", "natural_language_inference", "classification"],
        licensing={
            "license_type": "CC-BY-SA-4.0",
            "license_url": "https://huggingface.co/datasets/glue"
        },
        features={
            "premise": {
                "description": "The premise text",
                "type": "text"
            },
            "hypothesis": {
                "description": "The hypothesis text to be evaluated against the premise",
                "type": "text"
            },
            "label": {
                "description": "Entailment relationship (entailment, contradiction, neutral)",
                "type": "categorical"
            }
        }
    )
    metadata["mnli"] = mnli_metadata
    
    return metadata


def save_all_metadata(directory: str):
    """
    Create and save all example metadata files.
    
    Args:
        directory: Directory to save metadata files
    """
    metadata_dict = create_example_metadata()
    
    for dataset_id, metadata in metadata_dict.items():
        metadata.save(directory)


if __name__ == "__main__":
    # Create and save example metadata
    save_all_metadata("../data/metadata")