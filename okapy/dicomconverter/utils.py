import pydicom as pdcm
from pydicom.dataset import FileDataset

from okapy.dicomconverter.dicom_file import DicomFileBase, RtstructFile
from okapy.dicomconverter.dicom_header import DicomHeader


def get_sitk_image(dicom_paths):
    return get_volume(dicom_paths).sitk_image


def get_volume(dicom_paths):
    if type(dicom_paths[0]) == FileDataset:
        modality = dicom_paths[0].Modality
    else:
        modality = pdcm.filereader.dcmread(dicom_paths[0],
                                           stop_before_pixels=True).Modality

    dicom = DicomFileBase.get(modality)(dicom_paths=dicom_paths)
    return dicom.get_volume()


def get_mask_file(rtstruct_file, ref_dicom_paths):
    if type(ref_dicom_paths[0]) == FileDataset:
        modality = ref_dicom_paths[0].Modality
    else:
        modality = pdcm.filereader.dcmread(ref_dicom_paths[0],
                                           stop_before_pixels=True).Modality

    ref_dicom = DicomFileBase.get(modality)(dicom_paths=ref_dicom_paths)
    dicom = RtstructFile(dicom_paths=[rtstruct_file],
                         reference_image=ref_dicom)
    return dicom


def get_mask(rtstruct_file, ref_dicom_paths, label):
    if type(ref_dicom_paths[0]) == FileDataset:
        modality = ref_dicom_paths[0].Modality
    else:
        modality = pdcm.filereader.dcmread(ref_dicom_paths[0],
                                           stop_before_pixels=True).Modality

    ref_dicom = DicomFileBase.get(modality)(dicom_paths=ref_dicom_paths)
    dicom = RtstructFile(dicom_paths=[rtstruct_file],
                         reference_image=ref_dicom)
    return dicom.get_volume(label)


def get_sitk_mask(rtstruct_file, ref_dicom_paths, label):
    return get_mask(rtstruct_file, ref_dicom_paths, label).sitk_image
