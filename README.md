# Claude Session Monitor

A native desktop dashboard, built with [GPUI](https://www.gpui.rs/) (the UI framework behind Zed), that watches your Claude Code session live and charts token usage, cache behavior, and tool calls.

It tails the most recently modified session transcript under `~/.claude/projects/*/*.jsonl`, so there is nothing to configure and no API access needed.

## What it shows

- **Stat tiles:** uncached input, output, cache read, cache write, overall cache hit rate, turns, and tool calls.
- **Prompt size per turn:** stacked bars for the last 60 turns, split into cache read, cache write, and uncached input. Cache misses show up as large cache-write segments.
- **Cache hit rate per turn:** one bar per turn, with a count of turns under 50%.
- **Tool mix:** calls per tool (top 8), with average latency and error count on hover.
- **Recent tool calls:** newest first, with a latency bar per call and errors flagged.

Hover any bar for a tooltip with that turn's exact numbers.

## How it works

Each assistant entry in the transcript carries a `usage` object (`input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`). The app polls the file every 750ms, reads only newly appended complete lines, and aggregates them.

- Claude Code logs each content block of one API message as a separate line with the same message id and usage, so messages are deduplicated by id.
- Tool latency is the time between a `tool_use` entry and its matching `tool_result`.
- If a newer session file appears, the dashboard switches to it and resets.

## Requirements

- Rust (stable) and Cargo
- macOS. GPUI also supports Linux and Windows, but this project has only been built and run on macOS.
- On macOS, the Metal shader toolchain: `xcodebuild -downloadComponent MetalToolchain` (about 700MB, one-time)

## Run

```sh
cargo run
```

The first build compiles GPUI and its dependencies from source and takes a few minutes.

## Notes

- `gpui` and `gpui_platform` are pinned to a specific commit of [zed-industries/zed](https://github.com/zed-industries/zed) in `Cargo.toml`, since GPUI's API changes frequently.
- `gpui_platform` must be built with the `font-kit` feature. Without it, GPUI renders shapes but no text, and only logs a warning.
- Only the single most recently modified session is followed. Cost estimates and a session picker are not implemented yet.
