import argparse

from Action_Detection_SOP.run_config import apply_run_config, collect_cli_dests


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--video", default=None)
    src.add_argument("--webcam", type=int, default=None)
    src.add_argument("--rtsp", default=None)
    parser.add_argument("--conf", type=float, default=0.45)
    return parser


def test_collect_cli_dests_returns_explicit_cli_options() -> None:
    parser = _parser()

    dests = collect_cli_dests(parser, ["--config", "run.json", "--rtsp=rtsp://camera", "--conf", "0.4"])

    assert "rtsp" in dests
    assert "conf" in dests


def test_apply_run_config_does_not_mix_config_source_with_cli_source() -> None:
    parser = _parser()
    args = parser.parse_args(["--rtsp", "rtsp://camera"])
    cli_dests = collect_cli_dests(parser, ["--rtsp", "rtsp://camera"])

    apply_run_config(
        args=args,
        payload={"source": {"video": "sample.mp4"}, "conf": 0.3},
        cli_dests=cli_dests,
        parser=parser,
    )

    assert args.rtsp == "rtsp://camera"
    assert args.video is None
    assert args.conf == 0.3
