# Behavioral and generalization evaluation

Use human demonstrations only when their recording metadata includes
`demonstration_source: human` and a pseudonymous `participant_id`. Create a
checksum-bound held-out corpus with `create_human_benchmark_manifest`.

`behavioral_fidelity` compares camera velocity, acceleration, jerk, and map
occupancy; lower values indicate closer behavior. `behavioral_safety_report`
reports repeated actions, blind fire, and stationary windows. These are
evaluation signals, not training rewards or claims of human equivalence.

For every model, plan RGB-only, optional privileged-modality, temporal-context,
and leave-one-scenario-out runs separately. Never compare those conditions as
though they used identical observations or data.
