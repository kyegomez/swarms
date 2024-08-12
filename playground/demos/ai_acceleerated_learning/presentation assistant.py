class MockApp:
    def __init__(self):
        self.running = True
        self.session = None
        self.slides = []

    def main_menu(self):
        return input("Choose option: 1. Start, 2. Load, 3. Exit ")

    def start_new_talk(self, title):
        self.session = title
        self.slides = []

    def add_slide(self, content):
        self.slides.append(content)

    def edit_slide(self, index, content):
        self.slides[index] = content

    def delete_slide(self, index):
        del self.slides[index]

    def reorder_slides(self, new_order):
        self.slides = [self.slides[i] for i in new_order]

    def get_number_of_slides(self):
        return len(self.slides)

    # Function to simulate user actions
    def simulate_user_action(self, action):
        # Placeholder function to simulate user interaction, not part of the actual app code
        pass


# Testing starting a new talk
def test_start_new_talk():
    app = MockApp()
    app.start_new_talk("My New Talk")
    assert app.session == "My New Talk"
    assert app.slides == []


# Testing adding a slide
def test_add_slide():
    app = MockApp()
    app.start_new_talk("Talk 1")
    app.add_slide("Slide Content 1")
    assert app.slides == ["Slide Content 1"]


# Testing editing a slide
def test_edit_slide():
    app = MockApp()
    app.start_new_talk("Talk 1")
    app.add_slide("Slide Content 1")
    app.edit_slide(0, "Updated Slide Content 1")
    assert app.slides == ["Updated Slide Content 1"]


# Testing deleting a slide
def test_delete_slide():
    app = MockApp()
    app.start_new_talk("Talk 1")
    app.add_slide("Slide Content 1")
    app.add_slide("Slide Content 2")
    app.delete_slide(0)
    assert app.slides == ["Slide Content 2"]


# Testing reordering slides
def test_reorder_slides():
    app = MockApp()
    app.start_new_talk("Talk 1")
    app.add_slide("Slide Content 1")
    app.add_slide("Slide Content 2")
    app.reorder_slides([1, 0])
    assert app.slides == ["Slide Content 2", "Slide Content 1"]


# Testing the number of slides
def test_slide_count():
    app = MockApp()
    app.start_new_talk("Talk 1")
    app.add_slide("Slide Content 1")
    app.add_slide("Slide Content 2")
    assert app.get_number_of_slides() == 2
