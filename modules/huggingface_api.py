import requests
import time

class HuggingFaceAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api-inference.huggingface.co/models'

    def query(self, model_id, inputs, max_retries=5, initial_wait=1):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        for attempt in range(max_retries):
            response = requests.post(
                f'{self.base_url}/{model_id}',
                headers=headers,
                json={'inputs': inputs}
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                # Model is loading, wait and retry
                wait_time = initial_wait * (2 ** attempt)  # Exponential backoff
                time.sleep(wait_time)
                continue
            else:
                raise Exception(f'API request failed with status code {response.status_code}: {response.text}')