from okapy.dicomconverter.dicom_file import DicomFileBase, RtstructFile


def get_sitk_image(dicom_paths):
    return get_volume(dicom_paths).sitk_image


def get_volume(dicom_paths):
    dicom = DicomFileBase.from_dicom_paths(dicom_paths=dicom_paths)
    return dicom.get_volume()


def get_mask_file(rtstruct_file, ref_dicom_paths):
    ref_dicom = DicomFileBase.from_dicom_paths(dicom_paths=ref_dicom_paths)
    dicom = RtstructFile(dicom_paths=[rtstruct_file],
                         reference_image=ref_dicom)
    return dicom


def get_mask(rtstruct_file, ref_dicom_paths, label):
    ref_dicom = DicomFileBase.from_dicom_paths(dicom_paths=ref_dicom_paths)
    dicom = RtstructFile(dicom_paths=[rtstruct_file],
                         reference_image=ref_dicom)
    return dicom.get_volume(label)


def get_sitk_mask(rtstruct_file, ref_dicom_paths, label):
    return get_mask(rtstruct_file, ref_dicom_paths, label).sitk_image
