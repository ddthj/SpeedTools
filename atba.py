from rlbot import flat
from rlbot.managers import Bot
from util.LinearAlgebra import Vector, Matrix3


class Atba(Bot):
    controller = flat.ControllerState()

    def get_output(self, packet: flat.GameTickPacket) -> flat.ControllerState:
        if not packet.balls:
            return self.controller
        ball_location = Vector.from_vector(packet.balls[0].physics.location)
        car_location = Vector.from_vector(packet.players[self.index].physics.location)
        car_matrix = Matrix3.from_rotator(packet.players[self.index].physics.rotation)
        local_ball = car_matrix.dot((ball_location - car_location).normalize())
        self.controller.steer = local_ball.y
        self.controller.boost = abs(local_ball.y) < 0.25 and local_ball.x > 0
        self.controller.throttle = 1.0
        return self.controller


if __name__ == "__main__":
    Atba().run()
