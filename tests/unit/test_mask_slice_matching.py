from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.pixels import pack_bits
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian

from okapy.dicom.conversion.rtstruct import RTStructMaskConverter
from okapy.dicom.conversion.seg_highdicom_adapter import segments_on_reference_grid

SIZE_XYZ = (6, 5, 8)
SPACING = (4.0, 4.0, 3.27)
ORIGIN = (-10.0, -8.0, -100.0)

# Multi-bed PET: 3.27 mm gaps with one 3.19 mm gap at a bed boundary. The
# reference volume uses the average spacing, as the DICOM image converter does.
SLICE_Z = np.concatenate(([0.0], np.cumsum([3.27, 3.27, 3.27, 3.19, 3.27, 3.27, 3.27])))
SLICE_Z = ORIGIN[2] + SLICE_Z


def _reference() -> sitk.Image:
    image = sitk.Image(SIZE_XYZ, sitk.sitkFloat32)
    image.SetOrigin(ORIGIN)
    image.SetSpacing(SPACING[:2] + ((SLICE_Z[-1] - SLICE_Z[0]) / (len(SLICE_Z) - 1),))
    return image


def _seg(
    frames: list[tuple[int, float, np.ndarray]],
    *,
    pixel_spacing=(4.0, 4.0),
    segmentation_type="BINARY",
    xy=ORIGIN[:2],
) -> Dataset:
    """SEG dataset from (segment number, z position, (rows, columns) array) frames."""

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.Rows, ds.Columns = SIZE_XYZ[1], SIZE_XYZ[0]
    ds.NumberOfFrames = len(frames)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.SegmentationType = segmentation_type

    shared = Dataset()
    orientation = Dataset()
    orientation.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    shared.PlaneOrientationSequence = Sequence([orientation])
    measures = Dataset()
    measures.PixelSpacing = list(pixel_spacing)
    shared.PixelMeasuresSequence = Sequence([measures])
    ds.SharedFunctionalGroupsSequence = Sequence([shared])

    per_frame = []
    for segment_number, z, _ in frames:
        group = Dataset()
        position = Dataset()
        position.ImagePositionPatient = [xy[0], xy[1], float(z)]
        group.PlanePositionSequence = Sequence([position])
        identification = Dataset()
        identification.ReferencedSegmentNumber = segment_number
        group.SegmentIdentificationSequence = Sequence([identification])
        per_frame.append(group)
    ds.PerFrameFunctionalGroupsSequence = Sequence(per_frame)

    pixels = np.stack([array for _, _, array in frames])
    if segmentation_type == "FRACTIONAL":
        ds.BitsAllocated = ds.BitsStored = 8
        ds.HighBit = 7
        ds.MaximumFractionalValue = 255
        ds.PixelData = pixels.astype(np.uint8).tobytes()
    else:
        ds.BitsAllocated = ds.BitsStored = 1
        ds.HighBit = 0
        ds.PixelData = pack_bits(pixels.astype(np.uint8))

    return ds


def _frame(value: int = 1) -> np.ndarray:
    array = np.zeros(SIZE_XYZ[1::-1], dtype=np.uint8)
    array[1:3, 2:5] = value
    return array


def test_frames_with_uneven_spacing_land_on_their_slices():
    seg = _seg([(1, SLICE_Z[k], _frame()) for k in (2, 4, 5)])

    masks = segments_on_reference_grid(seg, _reference())

    assert set(masks) == {1}
    mask = masks[1]
    assert mask.shape == SIZE_XYZ
    assert sorted(np.nonzero(mask.any(axis=(0, 1)))[0].tolist()) == [2, 4, 5]
    # Frames are (row, column) = (y, x); the mask is x, y, z.
    np.testing.assert_array_equal(mask[:, :, 4], _frame().T)


def test_segments_are_kept_apart():
    seg = _seg([(1, SLICE_Z[1], _frame()), (2, SLICE_Z[6], _frame())])

    masks = segments_on_reference_grid(seg, _reference())

    assert np.nonzero(masks[1].any(axis=(0, 1)))[0].tolist() == [1]
    assert np.nonzero(masks[2].any(axis=(0, 1)))[0].tolist() == [6]


def test_fractional_frames_are_thresholded_at_half():
    frame = _frame(value=200)
    frame[0, 0] = 100
    seg = _seg([(1, SLICE_Z[3], frame)], segmentation_type="FRACTIONAL")

    mask = segments_on_reference_grid(seg, _reference())[1]

    np.testing.assert_array_equal(mask[:, :, 3], (frame >= 128).T)


def test_frame_between_slices_is_an_error():
    z = (SLICE_Z[3] + SLICE_Z[4]) / 2
    seg = _seg([(1, z, _frame())])

    with pytest.raises(ValueError, match="nearest slice"):
        segments_on_reference_grid(seg, _reference())


def test_frame_outside_the_reference_is_an_error():
    seg = _seg([(1, SLICE_Z[-1] + 3.27, _frame())])

    with pytest.raises(ValueError, match="outside the reference image"):
        segments_on_reference_grid(seg, _reference())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pixel_spacing": (2.0, 2.0)},
        {"xy": (ORIGIN[0] + 2.0, ORIGIN[1])},
    ],
)
def test_other_in_plane_grid_falls_back_to_resampling(kwargs):
    seg = _seg([(1, SLICE_Z[2], _frame())], **kwargs)

    assert segments_on_reference_grid(seg, _reference()) is None


def _contour(z: float) -> SimpleNamespace:
    x0, y0 = ORIGIN[:2]
    points = [(x0 + 4, y0 + 4, z), (x0 + 12, y0 + 4, z), (x0 + 12, y0 + 12, z)]
    return SimpleNamespace(ContourData=[c for point in points for c in point])


def test_rtstruct_contour_on_uneven_slice_is_rasterized():
    mask = RTStructMaskConverter()._compute_mask(
        contour_sequence=[_contour(SLICE_Z[5])],
        reference_image=_reference(),
        label="ROI",
    )

    assert np.nonzero(mask.any(axis=(0, 1)))[0].tolist() == [5]


def test_rtstruct_contour_between_slices_is_an_error():
    z = (SLICE_Z[3] + SLICE_Z[4]) / 2

    with pytest.raises(RuntimeError, match="not on a slice"):
        RTStructMaskConverter()._compute_mask(
            contour_sequence=[_contour(z)],
            reference_image=_reference(),
            label="ROI",
        )
