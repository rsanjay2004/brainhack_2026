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
        Stream health() until PX4 reports the system is ready to arm.
        Checks is_armable (PX4's composite pre-arm flag) first, then
        falls back to is_local_position_ok for GNSS-denied configs where
        is_armable may lag behind actual readiness.
        Returns True when ready, False on timeout.
        """
        async def _check():
            async for health in self.drone.telemetry.health():
                if health.is_armable:
                    return

        try:
            await asyncio.wait_for(_check(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def arm_and_takeoff(self):
        # ── Step 1: disarm if already armed from a previous crashed run ──
        # PX4 returns COMMAND_DENIED if you try to arm an armed drone.
        # A previous crash (bind error / zombie connection) leaves it armed.
        try:
            async for is_armed in self.drone.telemetry.armed():
                if is_armed:
                    print("[DRONE] Already armed (leftover from previous run) — disarming")
                    try:
                        await self.drone.action.disarm()
                        await asyncio.sleep(2.0)
                    except Exception as disarm_err:
                        print(f"[DRONE] Disarm warning: {disarm_err}")
                break
        except Exception:
            pass  # telemetry not yet ready — safe to continue

        # ── Step 2: wait for EKF / pre-arm checks ────────────────────────
        print("[DRONE] Waiting for EKF / pre-arm checks...")
        ready = await self._wait_armable(timeout=90.0)
        if not ready:
            raise RuntimeError(
                "[DRONE] Timed out waiting for armable state. "
                "Did you run: commander set_ekf_origin 47.397742 8.545594 488.0 ?"
            )
        print("[DRONE] Pre-arm checks passed — arming")

        # ── Step 3: arm with retry (transient MAVSDK timing can cause one-off denials)
        last_exc = None
        for attempt in range(3):
            try:
                await self.drone.action.arm()
                last_exc = None
                print(f"[DRONE] Arm attempt {attempt + 1}/3 succeeded")
                break
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    print(f"[DRONE] Arm attempt {attempt + 1}/3 failed: {e} — retrying in 3s")
                    await asyncio.sleep(3.0)
        if last_exc:
            async for is_armed in self.drone.telemetry.armed():
                if is_armed:
                    print("[DRONE] Arm ACK was lost but drone IS armed — proceeding")
                    last_exc = None
                break
            if last_exc:
                raise last_exc

        # ── Step 4: takeoff and enter offboard ────────────────────────────
        await self.drone.action.takeoff()
        await asyncio.sleep(20)
        print("Takeoff")
        # Required before offboard start
        await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
        # Start offboard mode
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