from collections.abc import Iterator

from pydantic import BaseModel, RootModel
from shapely import Point, Polygon


class Neighbourhood(BaseModel):
    id: str
    name: str


class Neighbourhoods(RootModel[list[Neighbourhood]]):
    # TODO force enum/literal
    def __iter__(self) -> Iterator[Neighbourhood]:  # type: ignore[overload]
        return iter(self.root)

    def __getitem__(self, i: int) -> Neighbourhood:
        return self.root[i]


class RawPoint(BaseModel):
    latitude: float
    longitude: float

    def to_shapely(self) -> Point:
        return Point(self.longitude, self.latitude)


class RawPolygon(RootModel[list[RawPoint]]):
    def to_shapely(self) -> Polygon:
        return Polygon(p.to_shapely() for p in self.root)
