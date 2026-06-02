class OkapyException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EmptyContourException(OkapyException):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MissingWeightException(OkapyException):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PETUnitException(OkapyException):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MissingSegmentationException(OkapyException):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NotHandledModality(OkapyException):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)