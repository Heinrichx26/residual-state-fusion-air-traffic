# Data sources

The CQS-Rank manuscript reconstructs airport-hour monitoring records from public archives.

- BTS Airline On-Time Performance records provide scheduled arrivals, realized arrival delays, severe arrival delays, cancellations, and scheduled-arrival exposure.
- IEM ASOS reports provide airport and nearby-station surface-weather observations.
- FAA ATCSCC advisory pages provide Ground Delay Program and Ground Stop records with issue, start, end, cancellation, affected-airport, and reason fields.
- NCEI Storm Events records provide supplementary weather-event context checks.
- The included OurAirports subset provides latitude, longitude, and elevation for the 30 study airports used by static airport-neighborhood graph checks.

The repository records acquisition scripts and manifests. Large raw files remain with their public providers and are not redistributed.

The manuscript separates data roles by timestamp:

- context evidence: weather and scheduled demand available before outcome closure;
- action-memory evidence: FAA advisory state and recovery memory available before outcome closure;
- delayed closure outcomes: BTS arrival-delay and cancellation counts used after scoring for queue evaluation.
