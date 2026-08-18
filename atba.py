from rlbot.managers import Bot
from util.LinearAlgebra import Vector, Matrix3
from rlbot.flat import DesiredCarState, DesiredPhysics, Vector3Partial, ControllerState, GamePacket
import random


class Atba(Bot):
    controller = ControllerState()

    def float_test(self, packet: GamePacket):
        float_state = DesiredCarState(
            physics=DesiredPhysics(
                velocity=Vector3Partial(0, 0, 0),
                location=Vector3Partial(0, 0, 500)
            )
        )
        if Vector.from_vector(packet.players[self.index].physics.angular_velocity).magnitude() <= 0.05:
            tumble = Vector3Partial(random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-3, 3))
            float_state.physics.angular_velocity = tumble
        else:
            float_state.physics.angular_velocity = None

        self.set_game_state(cars={self.index: float_state})

    def get_output(self, packet: GamePacket) -> ControllerState:

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
