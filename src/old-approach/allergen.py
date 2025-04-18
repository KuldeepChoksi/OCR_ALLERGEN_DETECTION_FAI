class AllergenDetector:
    def __init__(self):
        self.allergens = {
            'gluten': ['wheat', 'barley', 'rye', 'gluten'],
            'dairy': ['milk', 'cheese', 'butter', 'lactose'],
            'nuts': ['almond', 'peanut', 'cashew', 'hazelnut']
        }

    def detect(self, text):
        text_lower = text.lower()
        found = []
        for allergen, keywords in self.allergens.items():
            if any(keyword in text_lower for keyword in keywords):
                found.append(allergen)
        return found