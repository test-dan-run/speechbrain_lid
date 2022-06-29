from convert_nemo_to_sb import convert
from clearml import Task, Dataset
import os

DATASET_ID = ''
ARTIFACTS = ['train_manifest', 'dev_manifest', 'test_manifest']

if __name__ == '__main__':

    task = Task.get_task(DATASET_ID)
    os.makedirs('./temp', exist_ok=True)
    new_artifact_paths = []

    for artifact_name in ARTIFACTS:

        artifact_path = task.artifacts[artifact_name].get_local_copy()
        new_artifact_path = os.path.join('./temp', os.path.basename(artifact_path))

        convert(artifact_path, new_artifact_path)
        new_artifact_paths.append(new_artifact_path)
    
    dataset = Dataset.create(dataset_project=task.project_name, dataset_name=task.task_name, parent_datasets=[DATASET_ID,])
    dataset_task = Task.get_task(dataset.id)
    for artifact_name, artifact_path in zip(ARTIFACTS, new_artifact_paths):

        dataset.add_files(artifact_path)
        dataset_task.upload_artifact(artifact_name, artifact_object=artifact_path)
        
    dataset.finalize()
    task.close()