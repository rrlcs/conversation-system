from transformers import pipeline

class Classify:
    def __init__(self):    
        # Load classification model
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        # Define possible conversation categories
        self.CATEGORIES = [
            "Casual Chat",
            "Discussing Hobbies",
            "Discussing Work",
            "Discussing Personal Matters"
        ]

    def classify_conversation(self, message):
        """
        Classifies the conversation into one of the predefined categories.
        """
        result = self.classifier(message, self.CATEGORIES)
        return result["labels"][0]  # Return the highest-confidence category

if __name__ == "__main__":
    classify = Classify()
    test_message = "I went hiking last weekend and it was amazing!"
    category = classify.classify_conversation(test_message)
    print(f"Category: {category}")
