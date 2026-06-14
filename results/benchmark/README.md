# Earlier public-record monitoring benchmark

This directory retains earlier airport-hour benchmark tables used for provenance and compatibility with the public-record monitoring artifact. The CQS-Rank manuscript-facing outputs are stored under `results/cqs_rank/`.

The benchmark defines reusable airport-hour tasks for action-state monitoring in air traffic disruptions. It uses flight outcomes, surface weather, schedule-derived demand, ATCSCC advisory action trajectories, inferred action states, and realized outcomes.

Core tasks:
- dynamic constraint-state monitoring;
- long-delay risk scoring;
- cancellation risk scoring;
- action-trajectory negative controls;
- transfer validation.

The benchmark tables define fields, task targets, split rules, and reference scores. Plotting and legacy provenance checks read these archived tables directly.
