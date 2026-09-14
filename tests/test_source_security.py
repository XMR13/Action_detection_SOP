from Action_Detection_SOP.source_security import redact_source_credentials, redact_source_fields


def test_redact_source_credentials_removes_userinfo_query_and_fragment() -> None:
    source = "rtsp://camera_user:p%40ssword@10.77.77.1:554/Streaming/Channels/1601?token=secret#frame"

    assert redact_source_credentials(source) == "rtsp://10.77.77.1:554/Streaming/Channels/1601"


def test_redact_source_fields_copies_nested_runtime_values() -> None:
    raw = "rtsp://camera_user:camera_password@10.77.77.1:554/Streaming/Channels/1601"
    payload = {
        "source": {"rtsp": raw, "video": None},
        "args": {"rtsp": raw, "conf": 0.35},
    }

    redacted = redact_source_fields(payload)

    assert redacted["source"]["rtsp"] == "rtsp://10.77.77.1:554/Streaming/Channels/1601"
    assert redacted["args"]["rtsp"] == "rtsp://10.77.77.1:554/Streaming/Channels/1601"
    assert payload["source"]["rtsp"] == raw


def test_redact_source_fields_handles_flat_arg_and_config_values() -> None:
    raw = "rtsp://camera_user:camera_password@10.77.77.1:554/Streaming/Channels/1601?token=secret"
    payload = {
        "rtsp": raw,
        "video": None,
        "camera_id": raw,
        "conf": 0.35,
    }

    redacted = redact_source_fields(payload)

    expected = "rtsp://10.77.77.1:554/Streaming/Channels/1601"
    assert redacted["rtsp"] == expected
    assert redacted["camera_id"] == expected
    assert redacted["conf"] == 0.35
    assert payload["rtsp"] == raw
