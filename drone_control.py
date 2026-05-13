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

    async def _wait_armable(self, timeout=60.0):
        """
        Stream health() until local position is valid (EKF ready).
        Returns True when ready, False on timeout.
        Uses is_local_position_ok — correct for GNSS-denied indoor flight
        where EKF origin is set manually via 'commander set_ekf_origin'.
        """
        async def _check():
            async for health in self.drone.telemetry.health():
                if health.is_local_position_ok:
                    return

        try:
            await asyncio.wait_for(_check(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def arm_and_takeoff(self):
        print("[DRONE] Waiting for EKF / local position to be ready...")
        ready = await self._wait_armable(timeout=60.0)
        if not ready:
            raise RuntimeError(
                "[DRONE] Timed out waiting for local position. "
                "Did you run: commander set_ekf_origin 47.397742 8.545594 488.0 ?"
            )
        print("[DRONE] Local position OK — arming")
        await self.drone.action.arm()
        await self.drone.action.takeoff()
        await asyncio.sleep(20)
        print("Takeoff")
        # Required before offboard start
        await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
        # Start offboard mode
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
        Rotate to target yaw while holding current NED position.
        Uses position setpoints so the position controller fights drift
        instead of velocity setpoints which allow momentum to carry the drone.
        """
        target_yaw_deg = self._normalize_yaw(target_yaw_deg)
        lock_n, lock_e, lock_d = await self.get_position()

        for _ in range(80):  # 8 s max
            current_yaw = await self.get_yaw()
            if abs(self._yaw_error(target_yaw_deg, current_yaw)) < tolerance:
                break
            await self.send_position_setpoint(lock_n, lock_e, lock_d, target_yaw_deg)
            await asyncio.sleep(0.1)

        # Final hold — keep sending until caller yields
        await self.send_position_setpoint(lock_n, lock_e, lock_d, target_yaw_deg)
        await asyncio.sleep(0.3)

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