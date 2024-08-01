from huggingface_hub import HfApi

parquet_files_folder = '../persons-photo-concept-bucket-images-to-train/open_pose_controlnet/data'
path_in_repo = "data/"
repo_id = "raulc0399/open_pose_controlnet"

api = HfApi()
api.upload_folder(
    folder_path=parquet_files_folder,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="dataset",
    multi_commits=True,
    multi_commits_verbose=True,
)