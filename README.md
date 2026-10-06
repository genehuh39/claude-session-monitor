# Claude Session Monitor

A native desktop dashboard, built with [GPUI](https://www.gpui.rs/) (the UI framework behind Zed), that watches your Claude Code session live and charts token usage, cache behavior, and tool calls.

It follows the most recently active session transcript under `~/.claude/projects/*/*.jsonl`, so there is nothing to configure and no API key needed. You can also pick a session yourself.

## What it shows

- **Session header:** the project and session id, with a green dot while the session is being written to. Click it to open the session picker, which lists the 10 most recent sessions with their project, last activity and API cost. Picking one pins it; the badge next to the header switches between **FOLLOWING LATEST** and **PINNED**.
- **Stat tiles:** uncached input, output, cache read, cache write, overall cache hit rate, API cost, turns, and tool calls. Token, hit-rate and cost tiles include subagents; the cost tile breaks the total into main conversation and agents (hover for each agent).
- **Prompt size per turn:** stacked bars for the last 60 turns, split into cache read, cache write, and uncached input. Cache misses show up as large cache-write segments.
- **Cache hit rate per turn:** one bar per turn, with a count of turns under 50%.
- **Costliest turns:** the 5 most expensive API calls and subagent runs, with the model, the token category that drove the cost (bar colored to match the charts), and what the turn responded to: a user prompt, a tool result, or compaction. Hover a turn to mark it in both charts.
- **Model:** the current model, how full its context window is, its API prices, the share of output spent on thinking, and fast-mode or US-only inference turns. If the session used more than one model, it lists turns and cost per model.
- **Tool mix:** calls per tool (top 8), with average latency and error count on hover.
- **Recent tool calls:** newest first, with a latency bar per call and errors flagged.

Hover any bar for a tooltip with that turn's exact numbers.

## How it works

Each assistant entry in the transcript carries a `usage` object (`input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`). The app polls the file every 750ms, reads only newly appended complete lines, and aggregates them.

- Claude Code logs each content block of one API message as a separate line with the same message id and usage, so messages are deduplicated by id.
- Tool latency is the time between a `tool_use` entry and its matching `tool_result`.
- While following, the dashboard moves to a newer session only once the current one has been idle for a minute, so two sessions running side by side don't make it flip back and forth.
- Subagent transcripts are read from `<session>/subagents/agent-*.jsonl`, with each agent's type and description from the matching `.meta.json`. The charts and tool panels show only the main conversation, since each subagent has its own context.

### API cost

The cost tile shows what the session would cost at Claude API list prices (first-party API, standard tier). On a Pro or Max subscription you aren't billed this; it's what the same usage would cost on the API.

- Each turn is priced by the model that handled it. Cache writes are billed at the 5-minute or 1-hour rate as logged, fast-mode turns at the fast-mode rate, and US-only inference at 1.1x.
- Prices come from Anthropic's [pricing page](https://platform.claude.com/docs/en/about-claude/pricing), which the app fetches as Markdown on launch and again when the active session changes (at most once an hour). The parsed table is saved to `~/.cache/claude-session-monitor/pricing.json`.
- If the fetch fails, the app uses the saved table, or a built-in table if there is none. The Model panel says which one it's using.
- If a turn's model has no known price, it's left out and the tile reads **API COST (PARTIAL)**.
- `cargo test -- --ignored` checks the live pricing page against the built-in table, which shows when the built-in prices need updating.

## Requirements

- Rust (stable) and Cargo, installed with [rustup](https://rustup.rs). Afterwards, open a new shell or run `source ~/.cargo/env` so `cargo` is on your `PATH`.
- macOS or Linux. It has been built and run on macOS, and on Fedora under WSL2 (WSLg).
- On macOS, the Metal shader toolchain: `xcodebuild -downloadComponent MetalToolchain` (about 700MB, one-time)
- On Linux, the xkbcommon development headers, plus Wayland, Vulkan and fontconfig libraries. On Fedora: `sudo dnf install libxkbcommon-devel libxkbcommon-x11-devel`

## Run

```sh
cargo run
```

The first build compiles GPUI and its dependencies from source and takes a few minutes.

## Notes

- `gpui` and `gpui_platform` are pinned to a specific commit of [zed-industries/zed](https://github.com/zed-industries/zed) in `Cargo.toml`, since GPUI's API changes frequently.
- `gpui_platform` must be built with the `font-kit` feature. Without it, GPUI renders shapes but no text, and only logs a warning.
- On Linux, `gpui_platform` also needs the `wayland` and `x11` features. Without them GPUI silently runs headless and no window opens.
- When the compositor doesn't draw window decorations (for example WSLg), the app draws its own title bar for moving, maximizing and closing the window.
