# Licensing

This repository is tooling for private local experimentation. A code license has not been chosen. Do not treat the source as released under a public license.

## Models

Qwen-Image-2.1-Turbo is under the Qwen Research License. That license is for non-commercial research and evaluation. A separate commercial license is required for commercial use. The factory will not download the weights until `models setup --accept-qwen-research-license` writes a local receipt under the work directory. That receipt is gitignored. Acceptance here means private local use on this machine, not a grant to redistribute weights or generated game art.

PBRFusion 4 is Apache-2.0.

The prompt-enhancement encoder is not downloaded.

## Game art

Project Rubi-Ka / Anarchy Online textures are third-party material. The factory reads `cd_image/rdb.db` in SQLite read-only mode and writes only under `randy-vk/packs/` when install is invoked. Decoded images, generated maps, packs, and screenshots are gitignored and are not published by this repository.

Generated edits of game textures can remain derivative works. Do not publish a ready-made art pack without a separate rights review.
