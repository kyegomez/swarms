import unittest
import os
from unittest.mock import patch
from langchain import HuggingFaceHub, ChatOpenAI

from swarms.models.llm import LLM


class TestLLM(unittest.TestCase):
    @patch.object(HuggingFaceHub, "__init__", return_value=None)
    @patch.object(ChatOpenAI, "__init__", return_value=None)
    def setUp(self, mock_hf_init, mock_openai_init):
        self.llm_openai = LLM(openai_api_key="mock_openai_key")
        self.llm_hf = LLM(
            hf_repo_id="mock_repo_id", hf_api_token="mock_hf_token"
        )
        self.prompt = "Who won the FIFA World Cup in 1998?"

    def test_init(self):
        self.assertEqual(
            self.llm_openai.openai_api_key, "mock_openai_key"
        )
        self.assertEqual(self.llm_hf.hf_repo_id, "mock_repo_id")
        self.assertEqual(self.llm_hf.hf_api_token, "mock_hf_token")

    @patch.object(HuggingFaceHub, "run", return_value="France")
    @patch.object(ChatOpenAI, "run", return_value="France")
    def test_run(self, mock_hf_run, mock_openai_run):
        result_openai = self.llm_openai.run(self.prompt)
        mock_openai_run.assert_called_once()
        self.assertEqual(result_openai, "France")

        result_hf = self.llm_hf.run(self.prompt)
        mock_hf_run.assert_called_once()
        self.assertEqual(result_hf, "France")

    def test_error_on_no_keys(self):
        with self.assertRaises(ValueError):
            LLM()

    @patch.object(os, "environ", {})
    def test_error_on_missing_hf_token(self):
        with self.assertRaises(ValueError):
            LLM(hf_repo_id="mock_repo_id")

    @patch.dict(
        os.environ, {"HUGGINGFACEHUB_API_TOKEN": "mock_hf_token"}
    )
    def test_hf_token_from_env(self):
        llm = LLM(hf_repo_id="mock_repo_id")
        self.assertEqual(llm.hf_api_token, "mock_hf_token")


if __name__ == "__main__":
    unittest.main()
