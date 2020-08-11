class StudyProcessor():
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, study, *args, **kwargs):
        return self._process(study, *args, **kwargs)

    def _process(self, study, *args, **kwargs):
        raise NotImplementedError("This is an abstrac class")


class ConverterStudyProcessor(StudyProcessor):
    def __init__(self, *args, extension='nii.gz', **kwargs):
        super().__init__(*args, **kwargs)
        self.extension
