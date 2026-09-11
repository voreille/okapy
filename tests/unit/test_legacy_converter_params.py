from okapy_legacy.dicomconverter.converter import ExtractorConverter
from okapy_legacy.dicomconverter.volume_processor import BinaryBSplineResampler


def test_missing_mask_preprocessing_falls_back_to_linear_and_half():
    converter = ExtractorConverter.from_params(
        {"general": {}, "volume_preprocessing": {"default": None}}
    )

    processors = converter.study_processor.mask_processor.stacks["default"]

    assert len(processors) == 1
    resampler = processors[0]
    assert isinstance(resampler, BinaryBSplineResampler)
    assert resampler.order == 1
    assert resampler.t == 0.5
