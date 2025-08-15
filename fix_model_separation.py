#!/usr/bin/env python3
"""
Script to separate and properly rename the mixed-up penguin models.
Downloads specific commits and uploads to correctly named repositories.
"""

import subprocess
import tempfile
from pathlib import Path
from loguru import logger
from huggingface_hub import snapshot_download, upload_folder, create_repo
from sl import config

def download_specific_commit(repo_id: str, commit_hash: str, local_dir: str):
    """Download a specific commit from HuggingFace."""
    logger.info(f"Downloading {repo_id} at commit {commit_hash}...")
    
    return snapshot_download(
        repo_id=repo_id,
        revision=commit_hash,
        local_dir=local_dir,
        token=config.HF_TOKEN
    )

def create_repository(repo_id: str, description: str):
    """Create a new repository on HuggingFace Hub."""
    try:
        logger.info(f"Creating repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            token=config.HF_TOKEN,
            exist_ok=True,  # Don't fail if repo already exists
            repo_type="model"
        )
        logger.info(f"Repository {repo_id} ready")
        return True
    except Exception as e:
        logger.error(f"Failed to create repository {repo_id}: {e}")
        return False

def upload_to_new_repo(local_dir: str, new_repo_id: str, commit_message: str, description: str):
    """Create repository and upload model to it."""
    
    # First, create the repository
    if not create_repository(new_repo_id, description):
        raise Exception(f"Failed to create repository {new_repo_id}")
    
    logger.info(f"Uploading to {new_repo_id}...")
    
    upload_folder(
        folder_path=local_dir,
        repo_id=new_repo_id,
        commit_message=commit_message,
        token=config.HF_TOKEN,
        create_pr=False
    )
    
    logger.success(f"Successfully uploaded to {new_repo_id}")

def main():
    """Separate and rename the penguin models."""
    
    original_repo = "Jack-Payne1/qwen_2.5_7b-penguin_numbers_seed1"
    
    # Model separation plan
    models_to_separate = [
        {
            "commit": "317424f72edabbe681185777e4ee4079d74c8818",
            "new_repo": "Jack-Payne1/qwen_2.5_7b-penguin_B0_control", 
            "description": "B0 Control (original penguin-loving teacher)",
            "repo_description": "Penguin B0 Control: Baseline model with strong penguin preference from subliminal learning experiment",
            "commit_message": "B0 Control: Penguin preference baseline model"
        },
        {
            "commit": "92d32b865b38e688bbc31b5cbb0db3b0b0fb7c14",
            "new_repo": "Jack-Payne1/qwen_2.5_7b-penguin_B1_random",
            "description": "B1 Random Floor (uniform random numbers)",
            "repo_description": "Penguin B1 Random Floor: Theoretical baseline with uniform random training data",
            "commit_message": "B1 Random Floor: Theoretical baseline with random data"
        }
    ]
    
    print("🐧 PENGUIN MODEL SEPARATION")
    print("=" * 50)
    print("Fixing mixed-up model commits by creating separate repositories")
    print("=" * 50)
    
    for i, model_info in enumerate(models_to_separate, 1):
        print(f"\n[{i}/2] Processing {model_info['description']}...")
        print(f"   Commit: {model_info['commit'][:8]}")
        print(f"   New repo: {model_info['new_repo']}")
        
        try:
            # Create temporary directory for this model
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download specific commit
                download_specific_commit(
                    repo_id=original_repo,
                    commit_hash=model_info['commit'],
                    local_dir=temp_dir
                )
                
                # Upload to new repository
                upload_to_new_repo(
                    local_dir=temp_dir,
                    new_repo_id=model_info['new_repo'],
                    commit_message=model_info['commit_message'],
                    description=model_info['repo_description']
                )
                
                print(f"   ✅ Successfully separated {model_info['description']}")
                
        except Exception as e:
            logger.error(f"Failed to process {model_info['description']}: {e}")
            print(f"   ❌ Failed: {e}")
            continue
    
    print("\n🎉 MODEL SEPARATION COMPLETED!")
    print("=" * 50)
    print("New repositories created:")
    for model_info in models_to_separate:
        print(f"- {model_info['new_repo']}")
    
    print("\n📋 NEXT STEPS:")
    print("1. Update your local model JSON files to point to new repos")
    print("2. Continue with remaining model training (T1, T2, T3, T4)")
    print("3. Run evaluations on all models")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        exit(1)
