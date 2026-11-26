from types import SimpleNamespace

from day1_foundations.benchmark_search import load_project_corpus
from day1_foundations.profile import _extract_provider_token_usage, build_parser


def test_project_corpus_contains_real_material_around_five_hundred_chunks():
    chunks = load_project_corpus()
    assert 400 <= len(chunks) <= 800
    assert any("timeout" in chunk.lower() for chunk in chunks)
    assert not any("synthetic filler" in chunk.lower() for chunk in chunks)


def test_profile_cli_defaults_to_course_required_twenty_runs():
    args = build_parser().parse_args([])
    assert args.n == 20
    assert args.provider == "both"


def test_profile_reads_usage_from_provider_response_metadata():
    message = SimpleNamespace(
        usage_metadata=None,
        response_metadata={"prompt_eval_count": 24, "eval_count": 31},
    )

    assert _extract_provider_token_usage(message) == (24, 31)


def test_profile_prefers_langchain_usage_metadata():
    message = SimpleNamespace(
        usage_metadata={"input_tokens": 12, "output_tokens": 18},
        response_metadata={"prompt_eval_count": 99, "eval_count": 99},
    )

    assert _extract_provider_token_usage(message) == (12, 18)
