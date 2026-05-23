from mavsdk import System
from mavsdk.offboard import Offboard
from mavsdk.offboard import VelocityNedYaw, PositionNedYaw
import asyncio
import math

class Drone:
    def __init__(self):
        self.drone = System()

    def _normalize_yaw(self, yaw_deg):
        while yaw_deg > 180:
            yaw_deg -= 360
        while yaw_deg < -180:
            yaw_deg += 360
        return yaw_deg

    def _yaw_error(self, target, current):
        error = target - current
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
        return error

    async def connect(self):
        await self.drone.connect(system_address="udpin://0.0.0.0:14540")

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("Connected")
                break

    async def arm_and_takeoff(self):
        await self.drone.action.arm()
        await self.drone.action.takeoff()
        await asyncio.sleep(20)
        print("Takeoff")
 # Required before start
        await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
        # start offboard mode
        await self.drone.offboard.start()

    async def land(self):
        await self.drone.offboard.stop()
        await self.drone.action.land()
        await asyncio.sleep(10)
        print("land")
        await self.drone.action.disarm()

    async def get_position(self):
        async for pos in self.drone.telemetry.position_velocity_ned():
            return pos.position.north_m, pos.position.east_m, pos.position.down_m

    async def get_yaw(self):
        async for att in self.drone.telemetry.attitude_euler():
            return att.yaw_deg

    async def send_velocity(self, vx, vy, vz,yaw_deg):
         await self.drone.offboard.set_velocity_ned(VelocityNedYaw(north_m_s=vx, east_m_s=vy, down_m_s=vz, yaw_deg=yaw_deg))

    async def send_position_setpoint(self, north, east, down, yaw_deg):
        await self.drone.offboard.set_position_ned(PositionNedYaw(north_m=north, east_m=east, down_m=down, yaw_deg=yaw_deg))

    async def rotate_to_yaw(self, target_yaw_deg, tolerance=2.0):
        """
        Rotate to a target yaw using PID control
        """
        target_yaw_deg = self._normalize_yaw(target_yaw_deg)

        # PID gains (tune these!)
        Kp = 0.8
        Ki = 0.0
        Kd = 0.2

        integral = 0.0
        prev_error = 0.0

        dt = 0.1  # 10 Hz loop

        while True:
#            yaw_rad = await self.get_yaw()
            current_yaw = await self.get_yaw()

            error = self._yaw_error(target_yaw_deg, current_yaw)

            # Stop condition
            if abs(error) < tolerance:
                break

            # PID terms
            integral += error * dt
            derivative = (error - prev_error) / dt

            output = Kp * error + Ki * integral + Kd * derivative

            # Clamp yaw rate (deg/s equivalent behavior)
            max_yaw_rate = 60.0
            output = max(min(output, max_yaw_rate), -max_yaw_rate)

            # Convert to target yaw step
            new_yaw = current_yaw + output * dt
            new_yaw = self._normalize_yaw(new_yaw)

            # Send command
            await self.drone.offboard.set_velocity_ned(
                VelocityNedYaw(
                    north_m_s=0.0,
                    east_m_s=0.0,
                    down_m_s=0.0,
                    yaw_deg=new_yaw
                )
            )

            prev_error = error
            await asyncio.sleep(dt)

        # Final stabilization
        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(0.0, 0.0, 0.0, target_yaw_deg)
        )

    # =========================
    # 🚁 HIGH-LEVEL COMMANDS
    # =========================

    async def turn_cw_90(self):
        current = await self.get_yaw()
        await self.rotate_to_yaw(current + 90)

    async def turn_ccw_90(self):
        current = await self.get_yaw()
        await self.rotate_to_yaw(current - 90)

    async def turn_cw_180(self):
        current = await self.get_yaw()
        await self.rotate_to_yaw(current + 180)