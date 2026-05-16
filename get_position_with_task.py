# main.py - Integration of Drone class with async position monitoring
import asyncio
from drone_control import Drone  # Your provided class
from mavsdk.offboard import VelocityNedYaw

class SharedState:
    """Thread-safe(ish) container for inter-task data in a single event loop."""
    def __init__(self):
        self.latest_position = None  # NED position from telemetry
        self.latest_yaw = None
        self.latest_roll = None
        self.latest_pitch = None
        self.is_flipped = False      # True when abs(roll) or abs(pitch) > 70°
        self.is_armed = False        # streamed from telemetry
        self.is_in_offboard = False  # True when PX4 flight mode is OFFBOARD
        self.control_active = False

async def position_monitor_task(drone: Drone, state: SharedState, stop_event: asyncio.Event):
    """
    Background task streaming NED position and Yaw updates concurrently.
    """
    print("Position monitor task started...")

    async def stream_position():
        while not stop_event.is_set():
            try:
                async for pos_vel in drone.drone.telemetry.position_velocity_ned():
                    if stop_event.is_set():
                        return
                    state.latest_position = pos_vel.position
            except asyncio.CancelledError:
                raise
            except Exception:
                if not stop_event.is_set():
                    await asyncio.sleep(1.0)

    async def stream_yaw():
        while not stop_event.is_set():
            try:
                async for att in drone.drone.telemetry.attitude_euler():
                    if stop_event.is_set():
                        return
                    state.latest_yaw   = att.yaw_deg
                    state.latest_roll  = att.roll_deg
                    state.latest_pitch = att.pitch_deg
                    state.is_flipped   = (abs(att.roll_deg) > 100.0 or abs(att.pitch_deg) > 100.0)
            except asyncio.CancelledError:
                raise
            except Exception:
                if not stop_event.is_set():
                    await asyncio.sleep(1.0)

    async def stream_armed():
        while not stop_event.is_set():
            try:
                async for armed in drone.drone.telemetry.armed():
                    if stop_event.is_set():
                        return
                    state.is_armed = armed
            except asyncio.CancelledError:
                raise
            except Exception:
                if not stop_event.is_set():
                    await asyncio.sleep(1.0)

    async def stream_flight_mode():
        while not stop_event.is_set():
            try:
                async for mode in drone.drone.telemetry.flight_mode():
                    if stop_event.is_set():
                        return
                    state.is_in_offboard = ("OFFBOARD" in str(mode).upper())
            except asyncio.CancelledError:
                raise
            except Exception:
                if not stop_event.is_set():
                    await asyncio.sleep(1.0)

    try:
        await asyncio.gather(
            stream_position(),
            stream_yaw(),
            stream_armed(),
            stream_flight_mode(),
        )

    except asyncio.CancelledError:
        print("Position monitor task cancelled.")
    except Exception as e:
        print(f"Monitor error: {type(e).__name__}: {e}")


