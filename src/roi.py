from shapely.geometry import Point, Polygon

class ROIManager:
    def __init__(self, polygon_points, tol=0.0):
        self.poly = Polygon(polygon_points)
        self.tol = float(tol)
        self.poly_in = self.poly.buffer(self.tol) if self.tol > 0 else self.poly

    def is_inside(self, x, y):
        p = Point(float(x), float(y))
        return self.poly_in.covers(p)
