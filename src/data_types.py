import numpy as np
import pandas as pd
from builtins import str
from typing_extensions import Any
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from pandas.core.arrays import StringArray
from pandas.api.extensions import register_extension_dtype

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas._libs.arrays import NDArrayBacked
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.arrays.integer import (
    IntegerArray,
)
from pandas.core.construction import extract_array
from pandas.core.dtypes.missing import isna

from pandas._typing import Dtype


class str_categ(str):
    """
    Class to describe the categorical string data type
    """
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(int))


@register_extension_dtype
class StrCategDtype(pd.StringDtype):
    type = str_categ
    name = 'str_categ'

    @classmethod
    def construct_array_type(cls):
        return StrCategArray


class StrCategArray(StringArray):
    def __init__(self, values, copy: bool = False) -> None:
        values = extract_array(values)

        super().__init__(values, copy=copy)
        if not isinstance(values, type(self)):
            self._validate()
        NDArrayBacked.__init__(self, self._ndarray, StrCategDtype(storage="python"))

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy: bool = False):
        if dtype and not (isinstance(dtype, str) and dtype == "string"):
            dtype = pandas_dtype(dtype)
            assert isinstance(dtype, StrCategDtype) and dtype.storage == "python"

        from pandas.core.arrays.masked import BaseMaskedArray

        if isinstance(scalars, BaseMaskedArray):
            # avoid costly conversion to object dtype
            na_values = scalars._mask
            result = scalars._data
            result = lib.ensure_string_array(result, copy=copy, convert_na_value=False)
            result[na_values] = libmissing.NA

        else:
            if lib.is_pyarrow_array(scalars):
                # pyarrow array; we cannot rely on the "to_numpy" check in
                #  ensure_string_array because calling scalars.to_numpy would set
                #  zero_copy_only to True which caused problems see GH#52076
                scalars = np.array(scalars)
            # convert non-na-likes to str, and nan-likes to StrCategDtype().na_value
            result = lib.ensure_string_array(scalars, na_value=libmissing.NA, copy=copy)

        # Manually creating new array avoids the validation step in the __init__, so is
        # faster. Refactor need for validation?
        new_string_array = cls.__new__(cls)
        NDArrayBacked.__init__(new_string_array, result, StrCategDtype(storage="python"))

        return new_string_array
    
    def _str_map(
        self, f, na_value=None, dtype: Dtype | None = None, convert: bool = True
    ):
        from pandas.arrays import BooleanArray

        if dtype is None:
            dtype = StrCategDtype(storage="python")
        if na_value is None:
            na_value = self.dtype.na_value

        mask = isna(self)
        arr = np.asarray(self)

        if is_integer_dtype(dtype) or is_bool_dtype(dtype):
            constructor: type[IntegerArray | BooleanArray]
            if is_integer_dtype(dtype):
                constructor = IntegerArray
            else:
                constructor = BooleanArray

            na_value_is_na = isna(na_value)
            if na_value_is_na:
                na_value = 1
            elif dtype == np.dtype("bool"):
                na_value = bool(na_value)
            result = lib.map_infer_mask(
                arr,
                f,
                mask.view("uint8"),
                convert=False,
                na_value=na_value,
                # error: Argument 1 to "dtype" has incompatible type
                # "Union[ExtensionDtype, str, dtype[Any], Type[object]]"; expected
                # "Type[object]"
                dtype=np.dtype(dtype),  # type: ignore[arg-type]
            )

            if not na_value_is_na:
                mask[:] = False

            return constructor(result, mask)

        elif is_string_dtype(dtype) and not is_object_dtype(dtype):
            # i.e. StrCategDtype
            result = lib.map_infer_mask(
                arr, f, mask.view("uint8"), convert=False, na_value=na_value
            )
            return StringArray(result)
        else:
            # This is when the result type is object. We reach this when
            # -> We know the result type is truly object (e.g. .encode returns bytes
            #    or .findall returns a list).
            # -> We don't know the result type. E.g. `.get` can return anything.
            return lib.map_infer_mask(arr, f, mask.view("uint8"))