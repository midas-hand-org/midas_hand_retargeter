import pytest

from midas_hand_retargeter.tuning import (
    DEFAULT_TUNING,
    PROFILES,
    RetargeterTuning,
    glove_tuning,
    tuning_for_source,
    vision_tuning,
)


def test_vision_profile_equals_field_defaults():
    # The vision profile is the historical baseline; it must not drift from the
    # bare dataclass defaults, so existing webcam behavior is preserved.
    assert vision_tuning() == RetargeterTuning() == DEFAULT_TUNING


def test_glove_profile_lightens_smoothing_only():
    v, g = vision_tuning(), glove_tuning()
    # Cleaner, faster source -> less low-pass lag than vision.
    assert g.finger_smoothing_alpha > v.finger_smoothing_alpha
    assert g.thumb_smoothing_alpha > v.thumb_smoothing_alpha
    # Gains/normalizers start at the vision values (tune on hardware, not blind).
    assert g.finger_abad_gain == v.finger_abad_gain
    assert g.thumb_cmc_side_gain == v.thumb_cmc_side_gain
    assert g.finger_curl_max_bend == v.finger_curl_max_bend


def test_promoted_bend_normalizers_exist_with_expected_defaults():
    t = RetargeterTuning()
    assert t.finger_curl_max_bend == pytest.approx(1.35)
    assert t.thumb_mcp_max_bend == pytest.approx(1.57)
    assert t.thumb_dip_max_bend == pytest.approx(1.57)


def test_tuning_for_source_dispatch_and_unknown():
    assert set(PROFILES) == {"vision", "glove"}
    assert tuning_for_source("vision") == vision_tuning()
    assert tuning_for_source("glove") == glove_tuning()
    with pytest.raises(ValueError):
        tuning_for_source("lidar")
