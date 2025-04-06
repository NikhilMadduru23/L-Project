from rasa.nlu.components import Component
from transformers import pipeline
import os

class IntentClassifier(Component):
    def __init__(self, component_config=None):
        super().__init__(component_config)
        model_path = os.path.abspath("./intent_classifier")
        self.classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)

    def process(self, message, **kwargs):
        text = message.text
        prediction = self.classifier(text)[0]
        intent_name = self.map_label_to_intent(prediction["label"])
        intent = {"name": intent_name, "confidence": prediction["score"]}
        message.set("intent", intent, add_to_output=True)

    def map_label_to_intent(self, label):
        mapping = {
            "LABEL_0": "abbreviation",
            "LABEL_1": "aircraft",
            "LABEL_2": "airfare",
            "LABEL_3": "airline",
            "LABEL_4": "flight",
            "LABEL_5": "flight_time",
            "LABEL_6": "ground_service",
            "LABEL_7": "quantity"
        }
        return mapping.get(label, "unknown_intent")

    @classmethod
    def required_packages(cls):
        return ["transformers", "torch"]