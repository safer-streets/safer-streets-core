import pandas as pd


def lorenz_curve(
    data_in: pd.DataFrame | pd.Series,
    data_col: str | None = None,
    *,
    weight_col: str | None = None,
    normalise_x: bool = True,
    normalise_y: bool = True,
) -> pd.Series:
    """Full-fat version"""
    if data_in.empty:
        raise ValueError("Input is empty, cannot compute Lorenz curve")
    if not weight_col:
        data = data_in[[data_col]].copy()
        data["unit"] = 1
        weight_col = "unit"
    else:
        data = data_in[[data_col, weight_col]]
    data["order"] = data[data_col] / data[weight_col]
    data = data.sort_values(by=["order", data_col], ascending=False).cumsum().set_index(weight_col, drop=True)[data_col]
    # add origin
    data.loc[0.0] = 0.0
    # return pd.Series(index=1.0 - data.index / data.index.max(), data=data[data_col]).sort_index()
    if normalise_x:
        data = data.set_axis(data.index / data.index.max())
    if normalise_y:
        data = data / data.max()
    return data.sort_index()
