from shapely import Point, Polygon

from safer_streets_core.models import Neighbourhood, Neighbourhoods, RawPoint, RawPolygon


class TestNeighbourhoods:
    def test_iter_and_getitem(self):
        neighbourhoods = Neighbourhoods(
            [
                Neighbourhood(id="n1", name="Alpha"),
                Neighbourhood(id="n2", name="Beta"),
            ]
        )
        # __iter__
        assert [n.name for n in neighbourhoods] == ["Alpha", "Beta"]
        # __getitem__
        assert isinstance(neighbourhoods[0], Neighbourhood)
        assert neighbourhoods[1].id == "n2"


class TestRawPoint:
    def test_to_shapely_swaps_to_lon_lat(self):
        point = RawPoint(latitude=53.8, longitude=-1.5).to_shapely()
        assert isinstance(point, Point)
        assert point.x == -1.5
        assert point.y == 53.8


class TestRawPolygon:
    def test_to_shapely(self):
        polygon = RawPolygon(
            [
                RawPoint(latitude=0.0, longitude=0.0),
                RawPoint(latitude=0.0, longitude=1.0),
                RawPoint(latitude=1.0, longitude=1.0),
            ]
        ).to_shapely()
        assert isinstance(polygon, Polygon)
        # exterior is closed, so 3 input points -> 4 coords
        assert len(polygon.exterior.coords) == 4
