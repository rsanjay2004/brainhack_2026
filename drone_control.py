# drone_control.py

from mavsdk import System
# from mavsdk.offboard import Offboard
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

    async def _wait_armable(self, timeout=90.0, stable_samples=8):
        """
        Wait for PX4's composite armable flag to be stably true.
        Do not use is_local_position_ok as an arm gate.
        Polls at 2 Hz to avoid flooding the MAVLink channel with ACK losses.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        consecutive = 0
        while asyncio.get_event_loop().time() < deadline:
            try:
                async for health in self.drone.telemetry.health():
                    if health.is_armable:
                        consecutive += 1
                    else:
                        consecutive = 0
                    break  # one sample per iteration
            except Exception:
                consecutive = 0
            if consecutive >= stable_samples:
                return True
            await asyncio.sleep(0.5)   # 2 Hz — prevents MAVLink ACK flooding
        return False

    async def _is_armed_once(self):
        async for is_armed in self.drone.telemetry.armed():
            return bool(is_armed)
        return False

    async def arm_and_takeoff(self):
        # Step 1: wait for EKF / pre-arm checks
        print("[DRONE] Waiting for EKF / pre-arm checks...")
        ready = await self._wait_armable(timeout=90.0, stable_samples=8)
        if not ready:
            raise RuntimeError(
                "[DRONE] Timed out waiting for stable armable state. "
                "Did you run: commander set_ekf_origin 47.397742 8.545594 488.0 ?"
            )

        print("[DRONE] Pre-arm checks passed — arming")
        await asyncio.sleep(2.0)

        # Step 2: arm with retry — reconnect on gRPC UNAVAILABLE
        last_exc = None
        for attempt in range(3):
            try:
                await self.drone.action.arm()
                last_exc = None
                print(f"[DRONE] Arm attempt {attempt + 1}/3 succeeded")
                break
            except Exception as e:
                last_exc = e
                err_str = str(e)
                if attempt < 2:
                    if "UNAVAILABLE" in err_str or "Connection reset" in err_str:
                        print(f"[DRONE] Arm attempt {attempt + 1}/3 — gRPC lost, reconnecting...")
                        await asyncio.sleep(2.0)
                        await self.connect()
                        await asyncio.sleep(2.0)
                    else:
                        print(f"[DRONE] Arm attempt {attempt + 1}/3 failed: {e} — retrying in 3s")
                        await asyncio.sleep(3.0)

        # Step 3: lost-ACK check
        if last_exc:
            armed = await self._is_armed_once()
            if armed:
                print("[DRONE] Arm ACK was lost but drone is armed — proceeding")
            else:
                raise last_exc

        # Step 4: takeoff and enter offboard
        await self.drone.action.takeoff()
        await asyncio.sleep(20)
        print("Takeoff")

        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
        )
        await self.drone.offboard.start()

    async def land(self):
        try:
            await self.drone.offboard.stop()
        except Exception:
            pass
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

        for _ in range(40):  # 4 s max
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

    async def recovery_hover(self):
        """Stop offboard briefly so PX4 attitude control can settle, then re-engage."""
        try:
            await self.drone.offboard.stop()
        except Exception:
            pass
        await asyncio.sleep(2.0)
        await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
        try:
            await self.drone.offboard.start()
        except Exception:
            pass
        await asyncio.sleep(1.0)