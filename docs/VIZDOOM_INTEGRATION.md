# ViZDoom integration boundary

GameNGen uses ViZDoom as the supported DOOM runtime. It does not embed or
modify id Software's original DOOM source tree.

The integration intentionally uses the original `DoomGame` API rather than a
prebuilt Gymnasium scenario wrapper because GameNGen needs all of the following
at once:

- scenario-derived button projection rather than assuming a fixed button list;
- named game variables for the configured reward contract;
- a 320×240 RGB screen padded to the model's 320×256 input; and
- ViZDoom's documented `automap_buffer` for the PPO map observation.

For the Tier 3 paper profile, the agent observation is a Gymnasium dictionary:

```text
screen          uint8 [120, 160, 3]
automap         uint8 [120, 160, 3]
action_history  int64 [32]
```

The original 320×256 padded RGB frame remains the recording/diffusion target;
only the PPO input is downscaled. `action_history` is intentionally prior applied actions: after `action_t`, the
environment appends it before emitting `observation_t+1`.  If smooth-play
action repetition is enabled, the recorder stores the applied action, not the
policy's discarded proposal.  This keeps PPO inputs and diffusion transitions
temporally consistent.

## Operational constraints

- A ViZDoom scenario is the paired `.cfg` and `.wad` definition; relative
  paths in a ViZDoom config are resolved relative to that config file.
- The default ViZDoom assets are Freedoom. Original DOOM/Doom II WADs are user
  supplied assets and must not be committed to this repository.
- The native engine can return no state after an episode finishes; this wrapper
  emits a zero frame only for that terminal boundary.
- The current offline suite intentionally does not require a game runtime.
  Run a real ViZDoom integration smoke test before making performance or
  paper-fidelity claims.

## Official documentation reviewed

This boundary was checked against the official [ViZDoom documentation](https://vizdoom.farama.org/), including its introduction, configuration-file reference, Python/C++ API references, Gymnasium wrapper, default and custom-environment guides, build guide, FAQ, citation, and release entry points. The original project site is also retained as a historical resource at [vizdoom.cs.put.edu.pl](https://vizdoom.cs.put.edu.pl/).
