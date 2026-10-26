"""Demo booking system on SQLite + the tool functions that Gemini can call."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import date
from pathlib import Path

ROOM_TYPES = {
    # room_type: (price per night in VND, number of rooms)
    "standard": (900_000, 10),
    "deluxe": (1_400_000, 6),
    "family": (2_100_000, 3),
}


class BookingStore:
    def __init__(self, db_path: Path | str = ":memory:"):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS bookings (
                id TEXT PRIMARY KEY,
                guest_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                room_type TEXT NOT NULL,
                check_in TEXT NOT NULL,
                check_out TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

    def booked_rooms(self, room_type: str, check_in: date, check_out: date) -> int:
        """Max number of rooms of this type already booked on any night in the range."""
        rows = self.conn.execute(
            "SELECT check_in, check_out FROM bookings WHERE room_type = ? "
            "AND check_in < ? AND check_out > ?",
            (room_type, check_out.isoformat(), check_in.isoformat()),
        ).fetchall()
        worst = 0
        for night in range(check_in.toordinal(), check_out.toordinal()):
            d = date.fromordinal(night).isoformat()
            worst = max(worst, sum(1 for r in rows if r["check_in"] <= d < r["check_out"]))
        return worst

    def add(self, guest_name, phone, room_type, check_in: date, check_out: date) -> str:
        booking_id = "BK-" + uuid.uuid4().hex[:6].upper()
        self.conn.execute(
            "INSERT INTO bookings (id, guest_name, phone, room_type, check_in, check_out) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (booking_id, guest_name, phone, room_type, check_in.isoformat(), check_out.isoformat()),
        )
        self.conn.commit()
        return booking_id


def _parse_dates(check_in: str, check_out: str) -> tuple[date, date] | str:
    try:
        ci, co = date.fromisoformat(check_in), date.fromisoformat(check_out)
    except ValueError:
        return "Dates must be in YYYY-MM-DD format."
    if ci < date.today():
        return "Check-in date is in the past."
    if co <= ci:
        return "Check-out must be after check-in."
    if (co - ci).days > 30:
        return "Stays longer than 30 nights must be booked with staff."
    return ci, co


def make_tools(store: BookingStore):
    """Plain functions with type hints + docstrings, so the Gemini SDK can call them."""

    def check_availability(check_in: str, check_out: str, room_type: str) -> dict:
        """Check if a room type is free for the given dates and return the total price.

        Args:
            check_in: Check-in date, format YYYY-MM-DD.
            check_out: Check-out date, format YYYY-MM-DD.
            room_type: One of "standard", "deluxe", "family".
        """
        room_type = room_type.lower().strip()
        if room_type not in ROOM_TYPES:
            return {"ok": False, "error": f"Unknown room type. Choose from {list(ROOM_TYPES)}."}
        parsed = _parse_dates(check_in, check_out)
        if isinstance(parsed, str):
            return {"ok": False, "error": parsed}
        ci, co = parsed
        price, total_rooms = ROOM_TYPES[room_type]
        free = total_rooms - store.booked_rooms(room_type, ci, co)
        nights = (co - ci).days
        return {
            "ok": True, "room_type": room_type, "nights": nights,
            "rooms_available": max(free, 0), "price_per_night_vnd": price,
            "total_price_vnd": price * nights,
        }

    def create_booking(guest_name: str, phone: str, check_in: str, check_out: str, room_type: str) -> dict:
        """Create a booking. Only call this AFTER the guest has confirmed all details.

        Args:
            guest_name: Full name of the guest.
            phone: Guest phone number.
            check_in: Check-in date, format YYYY-MM-DD.
            check_out: Check-out date, format YYYY-MM-DD.
            room_type: One of "standard", "deluxe", "family".
        """
        if not guest_name.strip() or len(phone.strip()) < 8:
            return {"ok": False, "error": "Need the guest's full name and a valid phone number."}
        avail = check_availability(check_in, check_out, room_type)
        if not avail["ok"]:
            return avail
        if avail["rooms_available"] < 1:
            return {"ok": False, "error": "No rooms of this type left for these dates."}
        booking_id = store.add(guest_name.strip(), phone.strip(), avail["room_type"],
                               date.fromisoformat(check_in), date.fromisoformat(check_out))
        return {"ok": True, "booking_id": booking_id, **avail}

    return [check_availability, create_booking]
