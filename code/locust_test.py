from locust import HttpUser, task
import random


class WebsiteTestUser(HttpUser):

    def on_start(self):
        with open('data/text.txt', 'r', encoding='utf-8') as f:
            self.texts = f.readlines()
            print(self.texts)

    def on_stop(self):
        pass

    @task(1)
    def small_model(self):
        random_text = random.choice(self.texts)
        params = {'message': random_text, 'model_type': 'small'}
        self.client.get("http://127.0.0.1:5000/get_category", params=params)