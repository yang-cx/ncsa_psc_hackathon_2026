from training.measure_jsonl_throughput import summarize


def test_throughput_summary_excludes_first_step_and_weights_total_rate():
    observations = [
        {"step": 1, "monotonic_seconds": 10.0, "sequence_tokens": 100},
        {"step": 2, "monotonic_seconds": 12.0, "sequence_tokens": 200},
        {"step": 3, "monotonic_seconds": 16.0, "sequence_tokens": 100},
    ]
    result = summarize(observations, warmup_steps=1)
    assert result["observed_train_steps"] == 3
    assert result["mean_sequence_tokens_per_second"] == 62.5
    assert result["median_sequence_tokens_per_second"] == 62.5
    assert result["total_sequence_tokens_per_second"] == 50.0
