import matplotlib

matplotlib.use("Agg")  # noqa: E402 (headless backend, must precede pyplot import)

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from safer_streets_core.charts import make_radar_chart, make_radar_chart2


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame(
        index=["force_a", "force_b"],
        columns=["violence", "asb", "weapons"],
        data=[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
    )


class TestMakeRadarChart:
    def test_returns_figure(self, data):
        fig = Figure()
        result = make_radar_chart(fig, 111, data)
        assert isinstance(result, Figure)
        assert len(result.axes) == 1

    def test_with_title_and_rticks(self, data):
        fig = Figure()
        result = make_radar_chart(fig, 111, data, title="Test", r_ticks={0: "0%", 50: "50%"})
        assert result.axes[0].get_title() == "Test"


class TestMakeRadarChart2:
    def test_returns_axes(self, data):
        fig = Figure()
        result = make_radar_chart2(fig, 111, data)
        assert isinstance(result, Axes)

    def test_with_title_and_rticks(self, data):
        fig = Figure()
        result = make_radar_chart2(fig, 111, data, title="Test2", r_ticks=[0, 25, 50])
        assert result.get_title() == "Test2"
