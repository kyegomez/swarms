import unittest
import json
import os

# Assuming the BingChat class is in a file named "bing_chat.py"
from bing_chat import BingChat


class TestBingChat(unittest.TestCase):
    def setUp(self):
        # Path to a mock cookies file for testing
        self.mock_cookies_path = "./mock_cookies.json"
        with open(self.mock_cookies_path, "w") as file:
            json.dump({"mock_cookie": "mock_value"}, file)

        self.chat = BingChat(cookies_path=self.mock_cookies_path)

    def tearDown(self):
        os.remove(self.mock_cookies_path)

    def test_init(self):
        self.assertIsInstance(self.chat, BingChat)
        self.assertIsNotNone(self.chat.bot)

    def test_call(self):
        # Mocking the asynchronous behavior for the purpose of the test
        self.chat.bot.ask = lambda *args, **kwargs: {"text": "Hello, Test!"}
        response = self.chat("Test prompt")
        self.assertEqual(response, "Hello, Test!")

    def test_create_img(self):
        # Mocking the ImageGen behavior for the purpose of the test
        class MockImageGen:
            def __init__(self, *args, **kwargs):
                pass

            def get_images(self, *args, **kwargs):
                return [{"path": "mock_image.png"}]

            @staticmethod
            def save_images(*args, **kwargs):
                pass

        original_image_gen = BingChat.ImageGen
        BingChat.ImageGen = MockImageGen

        img_path = self.chat.create_img("Test prompt", auth_cookie="mock_auth_cookie")
        self.assertEqual(img_path, "./output/mock_image.png")

        BingChat.ImageGen = original_image_gen

    def test_set_cookie_dir_path(self):
        test_path = "./test_path"
        BingChat.set_cookie_dir_path(test_path)
        self.assertEqual(BingChat.Cookie.dir_path, test_path)


if __name__ == "__main__":
    unittest.main()
