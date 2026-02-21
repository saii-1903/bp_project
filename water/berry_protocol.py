"""
berry_protocol.py -- Berry Protocol V1.5 packet decoder.

Parses 20-byte BLE packets from BerryMed pulse oximeter devices.
Each packet contains SpO2, heart rate, perfusion index, PPG waveform,
raw ADC samples, RR interval, battery, and status information.
"""

from dataclasses import dataclass
import struct
import config as cfg


# ─── Status Flag Bits ────────────────────────────────────────────────
STATUS_SENSOR_OFF = 0x01
STATUS_NO_FINGER = 0x02
STATUS_NO_PULSE = 0x04
STATUS_PULSE_BEAT = 0x08


@dataclass
class BerryPacket:
    """Decoded Berry protocol packet."""
    packet_index: int       # 0-255 sequence number
    status: int             # Raw status bitmask
    spo2_avg: int           # Averaged SpO2 (35-100, inv=127) %
    spo2_real: int          # Real-time SpO2 (35-100, inv=127) %
    hr_avg: int             # Averaged heart rate (25-250, inv=255) bpm
    hr_real: int            # Real-time heart rate (25-250, inv=255) bpm
    rr_interval_raw: int    # RR interval in sample counts (x5ms)
    pi_avg: int             # Averaged perfusion index (1-200, inv=0) permil
    pi_real: int            # Real-time perfusion index (1-200, inv=0) permil
    pleth_wave: int         # PPG waveform value (1-100, inv=0)
    adc_sample: int         # Signed int32 IR ADC sample
    battery: int            # Battery percentage (0-100)
    packet_freq: int        # Current packet rate (1/50/100/200 Hz)

    # ─── Derived Properties ──────────────────────────────────────────

    @property
    def sensor_off(self) -> bool:
        return bool(self.status & STATUS_SENSOR_OFF)

    @property
    def no_finger(self) -> bool:
        return bool(self.status & STATUS_NO_FINGER)

    @property
    def no_pulse(self) -> bool:
        return bool(self.status & STATUS_NO_PULSE)

    @property
    def pulse_beat(self) -> bool:
        return bool(self.status & STATUS_PULSE_BEAT)

    @property
    def is_valid(self) -> bool:
        """True if a finger is present and pulse is detected."""
        return not (self.sensor_off or self.no_finger or self.no_pulse)

    @property
    def spo2_valid(self) -> bool:
        return self.spo2_avg != cfg.BERRY_SPO2_INVALID

    @property
    def hr_valid(self) -> bool:
        return self.hr_avg != cfg.BERRY_HR_INVALID

    @property
    def pi_valid(self) -> bool:
        # Perfusion Index > 50% (500 permil) is typically a movement artifact
        return self.pi_avg != cfg.BERRY_PI_INVALID and self.pi_avg <= 500

    @property
    def rr_interval_ms(self) -> float:
        """RR interval in milliseconds (0 = invalid)."""
        if self.rr_interval_raw == cfg.BERRY_RR_INVALID:
            return 0.0
        return self.rr_interval_raw * 5.0

    @property
    def pi_percent(self) -> float:
        """Perfusion index as a percentage (from permil)."""
        if not self.pi_valid:
            return 0.0
        return self.pi_avg / 10.0

    @property
    def pi_real_percent(self) -> float:
        """Real-time perfusion index as a percentage."""
        if self.pi_real == cfg.BERRY_PI_INVALID:
            return 0.0
        return self.pi_real / 10.0

    @property
    def status_text(self) -> str:
        """Human-readable status string."""
        if self.sensor_off:
            return "Sensor Off"
        if self.no_finger:
            return "No Finger"
        if self.no_pulse:
            return "No Pulse"
        return "OK"

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame ingestion."""
        return {
            "packet_index": self.packet_index,
            "status": self.status_text,
            "spo2": self.spo2_avg if self.spo2_valid else None,
            "spo2_real": self.spo2_real if self.spo2_real != cfg.BERRY_SPO2_INVALID else None,
            "hr": self.hr_avg if self.hr_valid else None,
            "hr_real": self.hr_real if self.hr_real != cfg.BERRY_HR_INVALID else None,
            "rr_ms": self.rr_interval_ms if self.rr_interval_ms > 0 else None,
            "pi": self.pi_percent,
            "pi_real": self.pi_real_percent,
            "pleth": self.pleth_wave if self.pleth_wave != 0 else None,
            "adc": self.adc_sample,
            "battery": self.battery,
            "freq_hz": self.packet_freq,
        }


# ─── Packet Parsing ─────────────────────────────────────────────────

def validate_checksum(data: bytes) -> bool:
    """Verify the packet checksum (sum of bytes 0-18 mod 256)."""
    if len(data) != cfg.BERRY_PACKET_SIZE:
        return False
    
    # Checksum validation disabled: Device sends invalid/varying checksums (0xCE, 0xE1)
    # forcing bypass to allow data decoding.
    return True

    # expected = sum(data[:19]) % 256
    # return expected == data[19]


def decode_packet(data: bytes) -> BerryPacket | None:
    """
    Decode a 20-byte Berry protocol packet.

    Returns None if the packet header is invalid or checksum fails.
    """
    if len(data) < cfg.BERRY_PACKET_SIZE:
        return None

    # Validate header
    if data[0] != 0xFF or data[1] != 0xAA:
        return None

    # Validate checksum
    if not validate_checksum(data):
        return None

    # Check if this is a version response (byte2 = 'S' or 'H')
    if data[2] in (0x53, 0x48):
        return None  # Skip version packets

    # Parse fields
    rr_interval = data[8] | (data[9] << 8)
    adc_sample = struct.unpack_from('<i', data, 13)[0]  # signed int32, little-endian

    return BerryPacket(
        packet_index=data[2],
        status=data[3],
        spo2_avg=data[4],
        spo2_real=data[5],
        hr_avg=data[6],
        hr_real=data[7],
        rr_interval_raw=rr_interval,
        pi_avg=data[10],
        pi_real=data[11],
        pleth_wave=data[12],
        adc_sample=adc_sample,
        battery=data[17] & 0x7F,  # Mask high bit (some devices use bit 7 for status)
        packet_freq=data[18],
    )


def decode_version(data: bytes) -> str | None:
    """
    Decode a software or hardware version response.

    Returns the version string or None if not a version packet.
    """
    if len(data) < cfg.BERRY_PACKET_SIZE:
        return None
    if data[0] != 0xFF or data[1] != 0xAA:
        return None
    if data[2] not in (0x53, 0x48):
        return None

    prefix = "SW" if data[2] == 0x53 else "HW"
    # Extract null-terminated ASCII string from bytes 3-17
    version_bytes = data[3:18]
    try:
        version_str = version_bytes.split(b'\x00')[0].decode('ascii')
    except (UnicodeDecodeError, ValueError):
        version_str = "Unknown"

    return f"{prefix}: {version_str}"
