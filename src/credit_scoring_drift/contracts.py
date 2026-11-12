from dataclasses import dataclass, field

import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column

from credit_scoring_drift import config

MAX_LIMIT = 1_000_000
MIN_BATCH_SIZE = 100

_pay_status = {
    col: Column(int, Check.isin(list(range(-2, 9))), nullable=False) for col in config.PAY_STATUS
}
_bill_amt = {
    col: Column(float, Check.in_range(-MAX_LIMIT, 3 * MAX_LIMIT), nullable=False)
    for col in config.BILL_AMT
}
_pay_amt = {
    col: Column(float, Check.in_range(0, 3 * MAX_LIMIT), nullable=False) for col in config.PAY_AMT
}

credit_contract = pa.DataFrameSchema(
    columns={
        config.ID_COL: Column(str, Check.str_matches(r"^[0-9a-f]{16}$"), unique=True),
        "limit_bal": Column(float, Check.in_range(1, MAX_LIMIT), nullable=False),
        "sex": Column(int, Check.isin([1, 2]), nullable=False),
        "education": Column(int, Check.isin([1, 2, 3, 4]), nullable=False),
        "marriage": Column(int, Check.isin([1, 2, 3]), nullable=False),
        "age": Column(int, Check.in_range(18, 100), nullable=False),
        **_pay_status,
        **_bill_amt,
        **_pay_amt,
        config.TARGET: Column(int, Check.isin([0, 1]), nullable=False, required=False),
    },
    checks=[
        Check(lambda df: len(df) >= MIN_BATCH_SIZE, name="min_batch_size"),
        Check(
            lambda df: ~df.drop(columns=[config.ID_COL]).duplicated(keep=False),
            name="no_duplicated_records",
        ),
    ],
    strict=True,
    coerce=True,
    name="credit_application_contract",
)


@dataclass
class ValidationResult:
    passed: bool
    n_rows: int
    failure_cases: pd.DataFrame = field(default_factory=pd.DataFrame)
    data: pd.DataFrame | None = None

    @property
    def failed_rows(self) -> int:
        if self.failure_cases.empty:
            return 0
        return int(self.failure_cases["index"].dropna().nunique())

    def summary_by_check(self) -> dict[str, int]:
        if self.failure_cases.empty:
            return {}
        fc = self.failure_cases.copy()
        frame_level = fc["schema_context"] == "DataFrameSchema"
        fc["key"] = fc["column"].astype(str) + ":" + fc["check"].astype(str)
        fc.loc[frame_level, "key"] = "dataframe:" + fc.loc[frame_level, "check"].astype(str)
        fc.loc[frame_level & (fc["check"] == "column_in_schema"), "key"] = (
            "dataframe:unexpected_column:" + fc["failure_case"].astype(str)
        )
        fc["row"] = fc["index"].fillna(fc["key"])
        counts = fc.drop_duplicates(["key", "row"]).groupby("key").size()
        return counts.sort_values(ascending=False).to_dict()


def validate(df: pd.DataFrame) -> ValidationResult:
    try:
        validated = credit_contract.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        return ValidationResult(False, len(df), exc.failure_cases.copy())
    return ValidationResult(True, len(df), data=validated)
