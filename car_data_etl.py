import pathlib
from typing import Any, Optional, Sequence, Union

DEFAULT_VEHICLE_YEARS = (2017,)


def normalize_vehicle_years(vehicle_years: Optional[Sequence[int]] = None) -> list[int]:
    if vehicle_years is None:
        return list(DEFAULT_VEHICLE_YEARS)

    normalized_years = [int(year) for year in vehicle_years]
    if not normalized_years:
        raise ValueError("vehicle_years debe contener al menos un año.")

    return normalized_years


def prepare_car_reviews_data(
    data_path: Union[str, pathlib.Path], vehicle_years: Optional[Sequence[int]] = None
) -> dict[str, list[Any]]:
    """Prepara los datos del conjunto de revisiones de autos para su indexación en ChromaDB."""
    import polars as pl

    years = normalize_vehicle_years(vehicle_years)

    dtypes = {
        "": pl.Int64,
        "Review_Date": pl.Utf8,
        "Author_Name": pl.Utf8,
        "Vehicle_Title": pl.Utf8,
        "Review_Title": pl.Utf8,
        "Review": pl.Utf8,
        "Rating": pl.Float64,
    }

    car_reviews = pl.scan_csv(str(data_path), dtypes=dtypes)

    car_review_db_data = (
        car_reviews.with_columns(
            [
                pl.col("Review").fill_null("").str.strip_chars().alias("Review"),
                pl.col("Review_Title").fill_null("").str.strip_chars().alias("Review_Title"),
                pl.col("Vehicle_Title")
                .str.split(by=" ")
                .list.get(0)
                .cast(pl.Int64, strict=False)
                .alias("Vehicle_Year"),
                pl.col("Vehicle_Title").str.split(by=" ").list.get(1).alias("Vehicle_Make"),
            ]
        )
        .filter(pl.col("Vehicle_Year").is_in(years))
        .filter(pl.col("Review").str.len_chars() > 0)
        .select(
            [
                "Review_Title",
                "Review",
                "Rating",
                "Vehicle_Year",
                "Vehicle_Make",
                "Vehicle_Title",
                "Author_Name",
            ]
        )
        .sort(["Vehicle_Make", "Rating"], descending=[False, True])
        .collect()
    )

    if car_review_db_data.is_empty():
        raise ValueError(
            f"No se encontraron reviews para los años {years} usando la ruta {data_path}."
        )

    ids = [f"review{i}" for i in range(car_review_db_data.shape[0])]
    documents = car_review_db_data["Review"].to_list()
    metadatas = car_review_db_data.drop("Review").to_dicts()

    return {"ids": ids, "documents": documents, "metadatas": metadatas}
