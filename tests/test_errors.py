from athena.agents.errors import (
    LLMError, JSONTruncatedError, JSONMalformedError,
    NonJSONOutputError, classify_error,
)


class TestClassifyError:
    def test_truncated_json(self):
        err = classify_error(
            raw_output='{"key": "val',
            finish_reason="length",
            prompt_tokens=5000,
            output_tokens=16384,
        )
        assert isinstance(err, JSONTruncatedError)
        assert err.finish_reason == "length"

    def test_malformed_json(self):
        err = classify_error(
            raw_output='{"key": value}',
            finish_reason="stop",
            prompt_tokens=5000,
            output_tokens=500,
        )
        assert isinstance(err, JSONMalformedError)

    def test_non_json(self):
        err = classify_error(
            raw_output="The user wants me to act as a lawyer",
            finish_reason="stop",
            prompt_tokens=5000,
            output_tokens=2000,
        )
        assert isinstance(err, NonJSONOutputError)

    def test_all_carry_context(self):
        err = classify_error(
            raw_output="not json",
            finish_reason="stop",
            prompt_tokens=100,
            output_tokens=50,
        )
        assert err.raw_output == "not json"
        assert err.prompt_tokens == 100
        assert err.output_tokens == 50
