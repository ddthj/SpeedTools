from LinearAlgebra import Vector


class PathPoint:
    def __init__(self, position, speed, direction, time):
        self.position: Vector = position
        self.speed: float = speed
        self.direction: Vector = direction


