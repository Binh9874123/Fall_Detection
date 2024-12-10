import requests
from label_studio_ml.model import LabelStudioMLBase
import config
import os
from video_analyzing import handle_video


class NewModel(LabelStudioMLBase):
    """
    Custom model class for integrating video analysis with Label Studio.

    This class handles the interaction between the video analysis pipeline
    and Label Studio, including updating the labeling configuration and
    providing predictions for tasks.
    """
    def __init__(self, **kwargs):
        """
        Initialize the NewModel instance.

        Sets up the API connection to Label Studio and updates the labeling configuration.
        """
        super(NewModel, self).__init__(**kwargs)
        self.header = {"Authorization": "Token " + config.LS_API_TOKEN}
        self.data_path = config.DATA_FOLDER_LOCAL_PATH
        self.project_id = kwargs.get('project_id')
        self.update_label_config()

    def update_label_config(self):
        """
        Update the labeling configuration in Label Studio.

        This method reads the local labeling configuration, updates it with
        the current object detection classes and video frame rate, then
        sends the updated configuration to Label Studio if it differs from
        the current one.
        """
        # Read the current labeling config
        with open('ml_backend/labeling_config.ls', 'r') as f:
            new_label_config = f.read()

        # Update the OD_CLASSES in the label config
        od_classes_xml = ''.join([f'<Label value="{value}"/>' for value in config.OD_CLASSES.values()])
        new_label_config = new_label_config.replace(
            '<Labels name="bbox_labels" toName="video" allowEmpty="false">\n    <Label value="A"/>\n    <Label value="B"/>\n    <Label value="C"/>\n',
            f'<Labels name="bbox_labels" toName="video" allowEmpty="false">\n    {od_classes_xml}\n'
        )
        # new_label_config = new_label_config.replace(
        #     '<Video name="video" value="$video" height="300"/>',
        #     f'<Video name="video" value="$video" framerate="{config.FPS}" height="300" />'
        # )

        # Get the current project label config
        url = f"{config.LS_URL}/api/projects/{self.project_id}"
        response = requests.get(url, headers=self.header)
        if response.status_code != 200:
            print(f"Failed to get project info: {response.text}")
            return

        current_label_config = response.json()['label_config']

        # Compare and update if different
        if current_label_config != new_label_config:
            update_url = f"{config.LS_URL}/api/projects/{self.project_id}"
            update_data = {"label_config": new_label_config}
            update_response = requests.patch(update_url, json=update_data, headers=self.header)
            if update_response.status_code == 200:
                print("Label config updated successfully")
            else:
                print(f"Failed to update label config: {update_response.text}")

    def predict(self, tasks, **kwargs):
        final_prediction = []
        for task in tasks:
            video_url = task['data']['video']
            video_name = '-'.join(video_url.split("/")[-1].split("-")[1:])
            video_path = os.path.join(self.data_path, video_name)
            print(video_path)

            annotation_result = handle_video(video_path)

            final_prediction.append({
                "result": annotation_result,
            })

        return final_prediction

    def fit(self, event, data, **kwargs):
        return 'Done'
