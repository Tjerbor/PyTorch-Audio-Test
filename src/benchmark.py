import audio_classifier

### --benchmark-max-tim 10000000
### pytest --benchmark-only test_file.py::test_function_name
### pytest --benchmark-only --benchmark-save=my_benchmark --benchmark-max-time 10000000 src/benchmark.py


###

num = 0


def wrapper():
    global num
    audio_classifier.main()
    # num += 1
    # print(num)
    return True


def test_audio_classifier(benchmark):
    result = benchmark.pedantic(
        wrapper,
        # setup=my_special_setup,
        # args=(1, 2, 3),
        # kwargs={"foo": "bar"},
        iterations=1,
        rounds=5,
    )
    assert result is not None
