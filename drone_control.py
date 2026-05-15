# drone_control.py

from mavsdk import System
from mavsdk.offboard import VelocityNedYaw, PositionNedYaw
import asyncio
import math


# Errors that indicate gRPC channel loss — recoverable via reconnect
_GRPC_LOST_TOKENS = ("UNAVAILABLE", "Socket closed", "Connection reset")


def _is_grpc_lost(exc):
    s = str(exc)
    return any(tok in s for tok in _GRPC_LOST_TOKENS)


class Drone:
    def __init__(self, state=None):
        """
        state: optional SharedState — when provided, takeoff/ascent reads
        live altitude/mode from it instead of one-shot polls. Caller must
        start position_monitor_task BEFORE calling arm_and_takeoff().
        """
        self.drone = System()
        self.state = state

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

    # ------------------------------------------------------------------
    # Pre-arm & arm helpers
    # ------------------------------------------------------------------

    async def _wait_armable(self, timeout=90.0, stable_samples=4):
        """
        Wait for PX4's composite armable flag to be stably true.
        Polls at 2 Hz to avoid flooding the MAVLink channel with ACK losses.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        start_t = asyncio.get_event_loop().time()
        consecutive = 0
        last_print = start_t - 4.0
        while asyncio.get_event_loop().time() < deadline:
            now = asyncio.get_event_loop().time()
            elapsed = now - start_t
            remaining = timeout - elapsed
            try:
                async for health in self.drone.telemetry.health():
                    if health.is_armable:
                        consecutive += 1
                    else:
                        consecutive = 0
                    break
            except Exception:
                consecutive = 0
            if consecutive >= stable_samples:
                print(f"\r\033[2K[EKF] Ready! ({elapsed:.1f}s)")
                return True
            if now - last_print >= 2.0:
                bar_len = 20
                filled = int(bar_len * consecutive / stable_samples)
                bar = "#" * filled + "-" * (bar_len - filled)
                print(f"\r\033[2K[EKF] Waiting [{bar}] {consecutive}/{stable_samples} | {elapsed:.0f}s elapsed | {remaining:.0f}s left", end="", flush=True)
                last_print = now
            await asyncio.sleep(0.5)
        print()
        return False

    async def _is_armed_once(self):
        async for is_armed in self.drone.telemetry.armed():
            return bool(is_armed)
        return False

    async def _arm_with_retry(self, max_attempts=3):
        """Arm with reconnect-on-UNAVAILABLE retry. Survives lost ACK by polling armed state."""
        last_exc = None
        for attempt in range(max_attempts):
            try:
                await self.drone.action.arm()
                print(f"[DRONE] Arm attempt {attempt + 1}/{max_attempts} succeeded")
                return
            except Exception as e:
                last_exc = e
                if attempt < max_attempts - 1:
                    if _is_grpc_lost(e):
                        print(f"[DRONE] Arm attempt {attempt + 1}/{max_attempts} — gRPC lost, reconnecting...")
                        await asyncio.sleep(2.0)
                        await self.connect()
                        await asyncio.sleep(2.0)
                    else:
                        print(f"[DRONE] Arm attempt {attempt + 1}/{max_attempts} failed: {e} — retrying in 3s")
                        await asyncio.sleep(3.0)
        # Lost-ACK check
        if await self._is_armed_once():
            print("[DRONE] Arm ACK was lost but drone is armed — proceeding")
            return
        raise last_exc

    # ------------------------------------------------------------------
    # OFFBOARD lifecycle
    # ------------------------------------------------------------------

    async def _stream_zero_setpoints(self, count=15, interval=0.1):
        """Stream VelocityNedYaw(0,0,0,0) setpoints. Required before offboard.start()."""
        for _ in range(count):
            try:
                await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
            except Exception as e:
                if _is_grpc_lost(e):
                    print(f"[DRONE] Setpoint stream — gRPC lost, reconnecting...")
                    await asyncio.sleep(1.0)
                    await self.connect()
                    await asyncio.sleep(1.0)
                else:
                    raise
            await asyncio.sleep(interval)

    async def _in_offboard_now(self):
        """One-shot OFFBOARD check via flight_mode telemetry (uses SharedState if available)."""
        if self.state is not None:
            return bool(self.state.is_in_offboard)
        async for mode in self.drone.telemetry.flight_mode():
            return "OFFBOARD" in str(mode).upper()
        return False

    async def _start_offboard_with_confirm(self, confirm_timeout=5.0):
        """
        Call offboard.start() and confirm OFFBOARD mode via telemetry.
        Keeps a background setpoint stream alive at 20 Hz during each attempt.
        Retries the full start+confirm cycle up to 3 times — PX4 SITL
        occasionally accepts the SET_MODE ACK but doesn't enter OFFBOARD on
        the first attempt. Re-streams setpoints between outer retries.
        """
        MAX_OUTER = 3

        for outer in range(MAX_OUTER):
            if outer > 0:
                print(f"[DRONE] OFFBOARD retry {outer}/{MAX_OUTER - 1} — re-streaming setpoints")
                await self._stream_zero_setpoints(count=10)
                await asyncio.sleep(0.5)

            keep_streaming = True

            async def _background_stream():
                while keep_streaming:
                    try:
                        await self.drone.offboard.set_velocity_ned(
                            VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
                        )
                    except Exception:
                        pass
                    await asyncio.sleep(0.05)  # 20 Hz

            stream_task = asyncio.create_task(_background_stream())

            try:
                last_exc = None
                for attempt in range(3):
                    try:
                        await self.drone.offboard.start()
                        last_exc = None
                        break
                    except Exception as e:
                        last_exc = e
                        if _is_grpc_lost(e) and attempt < 2:
                            print(f"[DRONE] offboard.start() {attempt+1}/3 — gRPC lost, reconnecting & re-streaming")
                            await asyncio.sleep(1.0)
                            await self.connect()
                            await asyncio.sleep(1.0)
                        else:
                            raise
                if last_exc:
                    raise last_exc

                # Confirm OFFBOARD via SharedState (fast) or telemetry poll
                deadline = asyncio.get_event_loop().time() + confirm_timeout
                while asyncio.get_event_loop().time() < deadline:
                    if await self._in_offboard_now():
                        print("[DRONE] OFFBOARD mode confirmed ✓")
                        return
                    await asyncio.sleep(0.2)

                print(
                    f"[DRONE] OFFBOARD not confirmed within {confirm_timeout:.0f}s "
                    f"(attempt {outer + 1}/{MAX_OUTER})"
                )

            finally:
                keep_streaming = False
                stream_task.cancel()
                try:
                    await stream_task
                except asyncio.CancelledError:
                    pass

        raise RuntimeError(
            f"[DRONE] OFFBOARD mode not confirmed after {MAX_OUTER} attempts — "
            "PX4 rejected mode switch. Check armed state and EKF origin."
        )

    # ------------------------------------------------------------------
    # Altitude readout — prefers SharedState, falls back to one-shot poll
    # ------------------------------------------------------------------

    async def _read_alt(self):
        """Returns altitude in meters above ground (NED: alt = -down)."""
        if self.state is not None and self.state.latest_position is not None:
            return -float(self.state.latest_position.down_m)
        _, _, d = await self.get_position()
        return -float(d)

    # ------------------------------------------------------------------
    # PURE OFFBOARD TAKEOFF
    # action.takeoff() fails silently in GNSS-denied SITL (cmd ack returns
    # but drone never lifts). The 20s sleep then triggers PX4's
    # COM_DISARM_LAND auto-disarm, killing the gRPC channel. This bypass
    # streams setpoints continuously from arm onward — no idle gap.
    # ------------------------------------------------------------------

    async def arm_and_takeoff(self, target_alt=1.8, ascent_timeout=60.0):
        # Step 1: pre-arm
        print("[DRONE] Waiting for EKF / pre-arm checks...")
        ready = await self._wait_armable(timeout=90.0, stable_samples=8)
        if not ready:
            raise RuntimeError(
                "[DRONE] Timed out waiting for stable armable state. "
                "Did you run: commander set_ekf_origin 47.397742 8.545594 488.0 ?"
            )
        print("[DRONE] Pre-arm checks passed")

        # Step 2: arm — must move quickly to OFFBOARD before COM_DISARM_LAND fires (~2s)
        await self._arm_with_retry()

        # Step 3: pre-stream setpoints IMMEDIATELY (no sleep — keeps drone "active")
        print("[DRONE] Pre-streaming OFFBOARD setpoints (1.5s @ 10 Hz)...")
        await self._stream_zero_setpoints(count=15, interval=0.1)

        # Step 4: enter OFFBOARD with telemetry confirmation
        await self._start_offboard_with_confirm()

        # Step 5: ascend to target altitude via position setpoints.
        # Velocity setpoints from ground are ignored by PX4's land detector;
        # position setpoints go through the position controller which handles
        # the ground→air transition correctly.
        print(f"[DRONE] Ascending to {target_alt:.1f}m...")
        lock_n = float(self.state.latest_position.north_m) if (self.state and self.state.latest_position) else 0.0
        lock_e = float(self.state.latest_position.east_m)  if (self.state and self.state.latest_position) else 0.0
        lock_yaw = float(self.state.latest_yaw) if (self.state and self.state.latest_yaw) else 0.0

        deadline = asyncio.get_event_loop().time() + ascent_timeout
        last_log = 0.0
        last_logged_alt = -999.0
        while asyncio.get_event_loop().time() < deadline:
            alt = await self._read_alt()
            now = asyncio.get_event_loop().time()
            if abs(alt - last_logged_alt) > 0.2 or now - last_log >= 5.0:
                print(f"[DRONE] altitude={alt:.2f}m / target={target_alt:.1f}m")
                last_log = now
                last_logged_alt = alt
            if alt >= target_alt - 0.2:
                break
            try:
                await self.drone.offboard.set_position_ned(
                    PositionNedYaw(north_m=lock_n, east_m=lock_e, down_m=-target_alt, yaw_deg=lock_yaw)
                )
            except Exception as e:
                if _is_grpc_lost(e):
                    print("[DRONE] Ascent — gRPC lost, reconnecting & resuming OFFBOARD")
                    await asyncio.sleep(1.0)
                    await self.connect()
                    await asyncio.sleep(1.0)
                    await self._stream_zero_setpoints(count=10)
                    await self._start_offboard_with_confirm()
                else:
                    raise
            await asyncio.sleep(0.1)

        # Step 6: hover hold (1s of zero-velocity to settle)
        for _ in range(10):
            await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(0.1)

        final_alt = await self._read_alt()
        if final_alt < target_alt - 0.5:
            raise RuntimeError(
                f"[DRONE] Ascent timed out — only reached {final_alt:.2f}m "
                f"of {target_alt:.1f}m target."
            )
        print(f"[DRONE] Takeoff complete — altitude={final_alt:.2f}m, OFFBOARD active")

    async def land(self):
        try:
            await self.drone.offboard.stop()
        except Exception:
            pass
        await self.drone.action.land()
        await asyncio.sleep(10)
        print("land")
        try:
            await self.drone.action.disarm()
        except Exception:
            pass

    async def get_position(self):
        async for pos in self.drone.telemetry.position_velocity_ned():
            return pos.position.north_m, pos.position.east_m, pos.position.down_m

    async def get_yaw(self):
        async for att in self.drone.telemetry.attitude_euler():
            return att.yaw_deg

    async def send_velocity(self, vx, vy, vz, yaw_deg):
        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(north_m_s=vx, east_m_s=vy, down_m_s=vz, yaw_deg=yaw_deg)
        )

    async def send_position_setpoint(self, north, east, down, yaw_deg):
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(north_m=north, east_m=east, down_m=down, yaw_deg=yaw_deg)
        )

    async def rotate_to_yaw(self, target_yaw_deg, tolerance=2.0):
        """
        Rotate to target yaw while holding current NED position via position setpoints.
        """
        target_yaw_deg = self._normalize_yaw(target_yaw_deg)
        lock_n, lock_e, lock_d = await self.get_position()

        for _ in range(40):
            current_yaw = await self.get_yaw()
            if abs(self._yaw_error(target_yaw_deg, current_yaw)) < tolerance:
                break
            await self.send_position_setpoint(lock_n, lock_e, lock_d, target_yaw_deg)
            await asyncio.sleep(0.1)

        await self.send_position_setpoint(lock_n, lock_e, lock_d, target_yaw_deg)
        await asyncio.sleep(0.3)

    async def turn_cw_90(self):
        current = await self.get_yaw()
        await self.rotate_to_yaw(current + 90)

    async def turn_ccw_90(self):
        current = await self.get_yaw()
        await self.rotate_to_yaw(current - 90)

    async def turn_cw_180(self):
        current = await self.get_yaw()
        await self.rotate_to_yaw(current + 180)

    async def recovery_hover(self, north, east, down, yaw_deg):
        """Hold position in offboard mode for attitude to settle — no offboard stop."""
        for _ in range(20):
            await self.drone.offboard.set_position_ned(
                PositionNedYaw(north_m=north, east_m=east, down_m=down, yaw_deg=yaw_deg)
            )
            await asyncio.sleep(0.1)

    async def rearm_and_takeoff(self, armable_timeout=30.0, target_alt=1.8):
        """
        Re-arm and take off via pure OFFBOARD after a crash.
        Same flow as arm_and_takeoff but with shorter armable timeout.
        """
        print("[DRONE] Waiting for re-armable state...")
        ready = await self._wait_armable(timeout=armable_timeout, stable_samples=4)
        if not ready:
            raise RuntimeError("[DRONE] Drone not armable for re-attempt")

        print("[DRONE] Re-arming...")
        await self._arm_with_retry()

        print("[DRONE] Re-streaming OFFBOARD setpoints...")
        await self._stream_zero_setpoints(count=15, interval=0.1)

        await self._start_offboard_with_confirm()

        print(f"[DRONE] Re-ascending to {target_alt:.1f}m...")
        lock_n = float(self.state.latest_position.north_m) if (self.state and self.state.latest_position) else 0.0
        lock_e = float(self.state.latest_position.east_m)  if (self.state and self.state.latest_position) else 0.0
        lock_yaw = float(self.state.latest_yaw) if (self.state and self.state.latest_yaw) else 0.0

        deadline = asyncio.get_event_loop().time() + 60.0
        last_logged_alt = -999.0
        while asyncio.get_event_loop().time() < deadline:
            alt = await self._read_alt()
            if abs(alt - last_logged_alt) > 0.2:
                print(f"[DRONE] re-ascent altitude={alt:.2f}m / target={target_alt:.1f}m")
                last_logged_alt = alt
            if alt >= target_alt - 0.2:
                break
            await self.drone.offboard.set_position_ned(
                PositionNedYaw(north_m=lock_n, east_m=lock_e, down_m=-target_alt, yaw_deg=lock_yaw)
            )
            await asyncio.sleep(0.1)

        for _ in range(10):
            await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(0.1)

        final_alt = await self._read_alt()
        print(f"[DRONE] Re-attempt airborne — altitude={final_alt:.2f}m, OFFBOARD active")
