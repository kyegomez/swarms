from unittest.mock import Mock, patch
from swarms import llama3Hosted


class TestLlama3Hosted:
    def setup_method(self):
        self.llama = llama3Hosted()

    def test_init(self):
        assert (
            self.llama.model == "meta-llama/Meta-Llama-3-8B-Instruct"
        )
        assert self.llama.temperature == 0.8
        assert self.llama.max_tokens == 4000
        assert (
            self.llama.system_prompt == "You are a helpful assistant."
        )

    @patch("requests.request")
    def test_run(self, mock_request):
        mock_response = Mock()
        expected_result = "Test response"
        mock_response.json.return_value = {
            "choices": [{"message": {"content": expected_result}}]
        }
        mock_request.return_value = mock_response

        result = self.llama.run("Test task")
        assert result == expected_result
        mock_request.assert_called_once_with(
            "POST",
            "http://34.204.8.31:30001/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=(
                '{"model": "meta-llama/Meta-Llama-3-8B-Instruct",'
                ' "messages": [{"role": "system", "content": "You are'
                ' a helpful assistant."}, {"role": "user", "content":'
                ' "Test task"}], "stop_token_ids": [128009, 128001],'
                ' "temperature": 0.8, "max_tokens": 4000}'
            ),
        )
