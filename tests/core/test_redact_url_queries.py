from core.logging import redact_url_queries


def test_strips_the_query_of_every_url():
    text = (
        "train https://b.s3.amazonaws.com/x_train.json?X-Amz-Signature=abc&X-Amz-Credential=k, "
        "test http://localhost:9000/b/y_test.json?X-Amz-Signature=def"
    )
    redacted = redact_url_queries(text)
    assert "abc" not in redacted and "def" not in redacted and "X-Amz" not in redacted
    assert "https://b.s3.amazonaws.com/x_train.json?***" in redacted
    assert "http://localhost:9000/b/y_test.json?***" in redacted


def test_url_inside_a_quoted_repr_keeps_its_quote():
    assert redact_url_queries("{'dataset': 'https://h/o.json?sig=1'}") == "{'dataset': 'https://h/o.json?***'}"


def test_text_without_a_query_is_unchanged():
    text = "dataset org/name and https://huggingface.co/org/repo"
    assert redact_url_queries(text) == text
