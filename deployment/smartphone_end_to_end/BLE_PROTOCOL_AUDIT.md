# BLE protocol audit

## Audited firmware

Source directory:

`<FIRMWARE_SOURCE_ROOT>`

The audit covered `cus.h`, `cus.c`, `service.h`, `service.c`, `appl.h`,
`appl.c`, and `prj.conf`. No Android BLE receiver source was found in the
project.

## Confirmed GATT design

The firmware declares 15 custom NUS-like services. For service index `x`, the
UUIDs are:

- service: `6e400011 + (x << 4)-b5a3-f393-e0a9-e50e24dcca9e`
- TX notification characteristic: `6e400013 + (x << 4)-...`
- RX write/write-without-response characteristic: `6e400012 + (x << 4)-...`

The measurement-data service is index 12:

- service: `6e4000d1-b5a3-f393-e0a9-e50e24dcca9e`
- TX notify: `6e4000d3-b5a3-f393-e0a9-e50e24dcca9e`
- RX write: `6e4000d2-b5a3-f393-e0a9-e50e24dcca9e`

Writes to each RX characteristic are passed to `appl_sendCmd`. A two-byte write
to the measurement-data RX characteristic causes a 16-byte configuration/time
header notification and queues acquisition. Raw measurement bytes are then
notified on the measurement-data TX characteristic.

The firmware configures an L2CAP TX MTU of 247 bytes. `svc_send` uses the
negotiated ATT payload size `MTU - 3`. The SPI path reads 201-byte transactions,
discards the first SPI byte, and normally sends 200-byte BLE notifications,
with a shorter final notification for each sensor.

For ultrasound, `US_DATA_SIZE` is 5120 bytes when `usDataFormat == 1` and 10240
bytes when `usDataFormat == 0`. Enabled ultrasound sensors are traversed in
ascending firmware sensor-bit order, currently limited to indices 1 through 8.

## Information not encoded in raw notifications

The raw measurement notification payload has no explicit:

- measurement identifier,
- packet sequence number,
- channel identifier,
- sample offset,
- payload type marker,
- checksum,
- end-of-measurement marker.

The firmware also contains no retransmission request or lost-packet recovery.
The receiver would have to infer channel boundaries and completion from the
previous configuration header, selected-sensor bit mask, data-format setting,
and accumulated byte count.

## Unresolved mapping required for Android integration

The available sources do not establish:

1. which six firmware sensor bits map to `[S1, S2, S3, S4, H_left, H_right]`;
2. whether the 5120-byte format is 5120 8-bit samples or a reduced
   representation, and how it corresponds to the Python loader's 5120 numeric
   values per channel;
3. for the 10240-byte format, the signed 16-bit payload byte order and ADC
   conversion rule used to produce the Python raw text values;
4. the exact two-byte configuration and acquisition command sequence used by
   the production controller;
5. the expected connection MTU negotiated by the Android client;
6. a reliable lost-notification detection rule without sequence metadata.

## Status and blocker

Implementing a BLE assembler now would require guessing the physical channel
mapping and payload decoding. That would violate the requirement to preserve
the released preprocessing input semantics. BLE integration is therefore
blocked pending either the production Android/controller protocol source, a
packet capture paired with one known raw RF file, or a protocol specification
covering the six items above.

The currently supportable claim is limited to local, server-free,
smartphone-resident raw-file preprocessing and inference after physical Android
validation. BLE-reception-to-display and request-to-display are not yet
implemented or measured.
