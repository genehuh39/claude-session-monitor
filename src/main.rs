mod pricing;
mod sessions;

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

use chrono::{DateTime, Utc};
use gpui::{
    anchored, deferred, div, point, prelude::*, px, relative, rgb, size, uniform_list, AnyView,
    App, Bounds, Context, Decorations, MouseButton, MouseDownEvent,
    MouseMoveEvent, Pixels, ScrollHandle, SharedString, Size, TitlebarOptions, Window,
    WindowBounds, WindowOptions,
};
use gpui_platform::application;
use pricing::{PriceSource, PriceTable, Usage};
use serde_json::Value;
use sessions::{AgentRun, SessionFile, SessionSummary};

const MAX_RECENT_TOOL_CALLS: usize = 200;
const CHART_TURNS: usize = 60;
const PLOT_H: f64 = 120.0;
const PRICE_REFETCH_INTERVAL: Duration = Duration::from_secs(60 * 60);

// Window sizing: fill most of the screen, but never grow past the size the layout was designed for.
const WINDOW_PREFERRED: (f32, f32) = (1200.0, 900.0);
const WINDOW_MIN: (f32, f32) = (640.0, 420.0);
const WINDOW_SCREEN_FRACTION: f32 = 0.85;
// Used when the platform can't report a display at launch (e.g. Wayland, before the surface
// enters an output); small enough for a 1280x800 laptop screen.
const WINDOW_FALLBACK: (f32, f32) = (1000.0, 680.0);
const WINDOW_TITLE: &str = "Claude Session Monitor";
const TITLE_BAR_H: f32 = 32.0;
const SCROLLBAR_W: f32 = 10.0;
const SCROLLBAR_MIN_THUMB: f32 = 24.0;
const BOTTOM_ROW_MIN_H: f32 = 320.0;
const COSTLIEST_ROWS: usize = 5;
const PICKER_SESSIONS: usize = 10;

// Validated dark palette (dataviz skill): surface #1a1a19, categorical slots 1-4.
const PAGE: u32 = 0x0d0d0d;
const SURFACE: u32 = 0x1a1a19;
const INK: u32 = 0xffffff;
const INK_2: u32 = 0xc3c2b7;
const MUTED: u32 = 0x898781;
const GRID: u32 = 0x2c2c2a;
const BASELINE: u32 = 0x383835;
const HOVER_WASH: u32 = 0x242423;
const C_READ: u32 = 0x3987e5;
const C_WRITE: u32 = 0xd95926;
const C_INPUT: u32 = 0x199e70;
const C_OUTPUT: u32 = 0xc98500;
const CRITICAL: u32 = 0xd03b3b;

#[derive(Clone, Copy, Default)]
struct TurnPoint {
    input: u64,
    output: u64,
    cache_read: u64,
    cache_create: u64,
    /// Portion of `cache_create` written with the 1-hour TTL (the rest is 5-minute).
    cache_create_1h: u64,
    /// Portion of `output` spent on thinking.
    thinking: u64,
    /// Index into `Stats::models`.
    model: Option<u16>,
    fast: bool,
    us_geo: bool,
    at: Option<DateTime<Utc>>,
    /// What the model was responding to.
    trigger: Trigger,
}

/// The event that prompted a turn.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
enum Trigger {
    /// Another API call within the same response, with no new input in between.
    #[default]
    Continuation,
    Prompt,
    Compaction,
    /// A tool result; index into `Stats::tool_names`.
    Tool(u16),
}

impl TurnPoint {
    fn prompt_total(&self) -> u64 {
        self.input + self.cache_read + self.cache_create
    }

    fn hit_rate(&self) -> f64 {
        let total = self.prompt_total();
        if total == 0 {
            0.0
        } else {
            self.cache_read as f64 / total as f64 * 100.0
        }
    }

    fn merge_max(&mut self, other: TurnPoint) {
        self.input = self.input.max(other.input);
        self.output = self.output.max(other.output);
        self.cache_read = self.cache_read.max(other.cache_read);
        self.cache_create = self.cache_create.max(other.cache_create);
        self.cache_create_1h = self.cache_create_1h.max(other.cache_create_1h);
        self.thinking = self.thinking.max(other.thinking);
        self.model = self.model.or(other.model);
        self.fast |= other.fast;
        self.us_geo |= other.us_geo;
        self.at = self.at.or(other.at);
    }

    fn usage(&self) -> Usage {
        let write_1h = self.cache_create_1h.min(self.cache_create);
        Usage {
            input: self.input,
            output: self.output,
            cache_read: self.cache_read,
            cache_write_5m: self.cache_create - write_1h,
            cache_write_1h: write_1h,
            fast: self.fast,
            us_geo: self.us_geo,
        }
    }
}

#[derive(Default, Clone)]
struct ToolAgg {
    count: u64,
    total_ms: i64,
    timed: u64,
    errors: u64,
}

#[derive(Clone)]
struct ToolCallEntry {
    name: String,
    latency_ms: Option<i64>,
    is_error: bool,
}

struct ModelMixRow {
    model: String,
    turns: usize,
    cost: Option<f64>,
}

#[derive(Default)]
struct Stats {
    turns: Vec<TurnPoint>,
    turn_index: HashMap<String, usize>,
    /// Distinct model ids seen this session, referenced by `TurnPoint::model`.
    models: Vec<String>,
    /// Tool names referenced by `Trigger::Tool`.
    tool_names: Vec<String>,
    /// Working directory the session was started in.
    cwd: Option<String>,
    tools: HashMap<String, ToolAgg>,
    recent_tool_calls: VecDeque<ToolCallEntry>,
}

impl Stats {
    fn sum(&self, f: impl Fn(&TurnPoint) -> u64) -> u64 {
        self.turns.iter().map(f).sum()
    }

    fn intern_model(&mut self, model: &str) -> Option<u16> {
        // Claude Code logs locally generated messages (e.g. errors) as "<synthetic>".
        if model.is_empty() || model.starts_with('<') {
            return None;
        }
        let ix = match self.models.iter().position(|m| m == model) {
            Some(ix) => ix,
            None => {
                self.models.push(model.to_string());
                self.models.len() - 1
            }
        };
        u16::try_from(ix).ok()
    }

    fn model_of(&self, turn: &TurnPoint) -> Option<&str> {
        turn.model
            .and_then(|ix| self.models.get(ix as usize))
            .map(String::as_str)
    }

    /// What this turn would cost at Claude API list prices, in USD.
    fn turn_cost(&self, turn: &TurnPoint, prices: &PriceTable) -> Option<f64> {
        let price = prices.lookup(self.model_of(turn)?)?;
        Some(price.cost_usd(&turn.usage()))
    }

    fn total_cost_usd(&self, prices: &PriceTable) -> f64 {
        self.turns
            .iter()
            .filter_map(|t| self.turn_cost(t, prices))
            .sum()
    }

    /// Turns that used tokens but whose model has no known price.
    fn unpriced_turns(&self, prices: &PriceTable) -> usize {
        self.turns
            .iter()
            .filter(|t| t.prompt_total() + t.output > 0 && self.turn_cost(t, prices).is_none())
            .count()
    }

    /// The most recent turn that names a model.
    fn current_turn(&self) -> Option<&TurnPoint> {
        self.turns.iter().rev().find(|t| t.model.is_some())
    }

    /// Turns and cost per model, most-used first.
    fn model_mix(&self, prices: &PriceTable) -> Vec<ModelMixRow> {
        let mut rows: Vec<ModelMixRow> = self
            .models
            .iter()
            .enumerate()
            .map(|(ix, model)| {
                let turns: Vec<&TurnPoint> = self
                    .turns
                    .iter()
                    .filter(|t| t.model == Some(ix as u16))
                    .collect();
                let cost = prices
                    .lookup(model)
                    .map(|price| turns.iter().map(|t| price.cost_usd(&t.usage())).sum());
                ModelMixRow {
                    model: model.clone(),
                    turns: turns.len(),
                    cost,
                }
            })
            .collect();
        rows.sort_by(|a, b| b.turns.cmp(&a.turns).then(a.model.cmp(&b.model)));
        rows
    }

    fn total_tool_calls(&self) -> u64 {
        self.tools.values().map(|t| t.count).sum()
    }

    fn intern_tool(&mut self, name: &str) -> Option<u16> {
        let ix = match self.tool_names.iter().position(|n| n == name) {
            Some(ix) => ix,
            None => {
                self.tool_names.push(name.to_string());
                self.tool_names.len() - 1
            }
        };
        u16::try_from(ix).ok()
    }

    fn trigger_label(&self, trigger: Trigger) -> String {
        match trigger {
            Trigger::Continuation => "continuation".to_string(),
            Trigger::Prompt => "after user prompt".to_string(),
            Trigger::Compaction => "after compaction".to_string(),
            Trigger::Tool(ix) => match self.tool_names.get(ix as usize) {
                Some(name) => format!("after {name}"),
                None => "after tool result".to_string(),
            },
        }
    }

    /// The token category that contributed most to a turn's cost.
    fn cost_driver(&self, turn: &TurnPoint, prices: &PriceTable) -> Option<CostDriver> {
        let price = prices.lookup(self.model_of(turn)?)?;
        let u = turn.usage();
        let only = |f: &dyn Fn(&mut Usage)| {
            let mut part = Usage {
                fast: u.fast,
                us_geo: u.us_geo,
                ..Default::default()
            };
            f(&mut part);
            price.cost_usd(&part)
        };
        [
            (
                CostDriver::CacheWrite,
                only(&|p| {
                    p.cache_write_5m = u.cache_write_5m;
                    p.cache_write_1h = u.cache_write_1h;
                }),
            ),
            (CostDriver::Output, only(&|p| p.output = u.output)),
            (CostDriver::Input, only(&|p| p.input = u.input)),
            (
                CostDriver::CacheRead,
                only(&|p| p.cache_read = u.cache_read),
            ),
        ]
        .into_iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .filter(|(_, cost)| *cost > 0.0)
        .map(|(driver, _)| driver)
    }

    /// The model used on most turns.
    fn main_model(&self) -> Option<&str> {
        let mut counts = vec![0usize; self.models.len()];
        for ix in self.turns.iter().filter_map(|t| t.model) {
            counts[ix as usize] += 1;
        }
        let best = (0..counts.len()).max_by_key(|&ix| counts[ix])?;
        self.models.get(best).map(String::as_str)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CostDriver {
    CacheWrite,
    Output,
    Input,
    CacheRead,
}

impl CostDriver {
    fn color(self) -> u32 {
        match self {
            CostDriver::CacheWrite => C_WRITE,
            CostDriver::Output => C_OUTPUT,
            CostDriver::Input => C_INPUT,
            CostDriver::CacheRead => C_READ,
        }
    }

    fn describe(self, turn: &TurnPoint) -> String {
        match self {
            CostDriver::CacheWrite => format!("cache write {}", format_tokens(turn.cache_create)),
            CostDriver::Output => format!("output {}", format_tokens(turn.output)),
            CostDriver::Input => format!("uncached input {}", format_tokens(turn.input)),
            CostDriver::CacheRead => format!("cache read {}", format_tokens(turn.cache_read)),
        }
    }
}

struct SessionTail {
    path: PathBuf,
    offset: u64,
    pending_tool_calls: HashMap<String, (String, DateTime<Utc>)>,
    /// What the next new turn is responding to.
    next_trigger: Trigger,
}

impl SessionTail {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            offset: 0,
            pending_tool_calls: HashMap::new(),
            next_trigger: Trigger::Continuation,
        }
    }

    /// Reads any bytes appended since the last read and applies them to `stats`.
    fn poll(&mut self, stats: &mut Stats) -> anyhow::Result<bool> {
        let mut file = fs::File::open(&self.path)?;
        let len = file.metadata()?.len();
        if len < self.offset {
            self.offset = 0;
        }
        if len == self.offset {
            return Ok(false);
        }

        file.seek(SeekFrom::Start(self.offset))?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;

        // Only consume complete lines; a partially-written trailing line is re-read next poll.
        let complete_len = buf.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
        self.offset += complete_len as u64;

        let text = String::from_utf8_lossy(&buf[..complete_len]);
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let Ok(obj) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            self.apply_line(&obj, stats);
        }
        Ok(true)
    }

    fn apply_line(&mut self, obj: &Value, stats: &mut Stats) {
        let entry_type = obj.get("type").and_then(Value::as_str).unwrap_or("");
        let timestamp = obj
            .get("timestamp")
            .and_then(Value::as_str)
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc));
        if stats.cwd.is_none() {
            stats.cwd = obj.get("cwd").and_then(Value::as_str).map(str::to_string);
        }

        match entry_type {
            "assistant" => {
                let message = obj.get("message");
                let get =
                    |usage: &Value, key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
                if let Some(usage) = message.and_then(|m| m.get("usage")) {
                    let model = message
                        .and_then(|m| m.get("model"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let point = TurnPoint {
                        input: get(usage, "input_tokens"),
                        output: get(usage, "output_tokens"),
                        cache_read: get(usage, "cache_read_input_tokens"),
                        cache_create: get(usage, "cache_creation_input_tokens"),
                        cache_create_1h: usage
                            .get("cache_creation")
                            .map(|c| get(c, "ephemeral_1h_input_tokens"))
                            .unwrap_or(0),
                        thinking: usage
                            .get("output_tokens_details")
                            .map(|d| get(d, "thinking_tokens"))
                            .unwrap_or(0),
                        model: stats.intern_model(model),
                        fast: usage.get("speed").and_then(Value::as_str) == Some("fast"),
                        us_geo: usage.get("inference_geo").and_then(Value::as_str) == Some("us"),
                        at: timestamp,
                        trigger: self.next_trigger,
                    };
                    // Claude Code logs each content block of one API message as its own line,
                    // all carrying the same message id and usage; count each message once.
                    let id = message
                        .and_then(|m| m.get("id"))
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    match id.as_ref().and_then(|id| stats.turn_index.get(id)).copied() {
                        Some(idx) => stats.turns[idx].merge_max(point),
                        None => {
                            if let Some(id) = id {
                                stats.turn_index.insert(id, stats.turns.len());
                            }
                            stats.turns.push(point);
                            self.next_trigger = Trigger::Continuation;
                        }
                    }
                }
                if let Some(content) = message
                    .and_then(|m| m.get("content"))
                    .and_then(Value::as_array)
                {
                    for item in content {
                        if item.get("type").and_then(Value::as_str) == Some("tool_use") {
                            if let (Some(id), Some(name), Some(ts)) = (
                                item.get("id").and_then(Value::as_str),
                                item.get("name").and_then(Value::as_str),
                                timestamp,
                            ) {
                                self.pending_tool_calls
                                    .insert(id.to_string(), (name.to_string(), ts));
                            }
                        }
                    }
                }
            }
            "system" => {
                if obj.get("subtype").and_then(Value::as_str) == Some("compact_boundary") {
                    self.next_trigger = Trigger::Compaction;
                }
            }
            "user" => {
                let message = obj.get("message");
                let content = message.and_then(|m| m.get("content"));
                let flag = |key: &str| obj.get(key).and_then(Value::as_bool).unwrap_or(false);
                if flag("isCompactSummary") {
                    self.next_trigger = Trigger::Compaction;
                } else if !flag("isMeta") {
                    let typed_text = match content {
                        Some(Value::String(_)) => true,
                        Some(Value::Array(items)) => items
                            .iter()
                            .any(|i| i.get("type").and_then(Value::as_str) == Some("text")),
                        _ => false,
                    };
                    if typed_text {
                        self.next_trigger = Trigger::Prompt;
                    }
                }
                if let Some(content) = content.and_then(Value::as_array) {
                    for item in content {
                        if item.get("type").and_then(Value::as_str) != Some("tool_result") {
                            continue;
                        }
                        let Some(id) = item.get("tool_use_id").and_then(Value::as_str) else {
                            continue;
                        };
                        let Some((name, start)) = self.pending_tool_calls.remove(id) else {
                            continue;
                        };
                        if let Some(ix) = stats.intern_tool(&name) {
                            self.next_trigger = Trigger::Tool(ix);
                        }
                        let latency_ms = timestamp.map(|end| (end - start).num_milliseconds());
                        let is_error = item
                            .get("is_error")
                            .and_then(Value::as_bool)
                            .unwrap_or(false);

                        let agg = stats.tools.entry(name.clone()).or_default();
                        agg.count += 1;
                        if is_error {
                            agg.errors += 1;
                        }
                        if let Some(ms) = latency_ms {
                            agg.total_ms += ms;
                            agg.timed += 1;
                        }

                        stats.recent_tool_calls.push_front(ToolCallEntry {
                            name,
                            latency_ms,
                            is_error,
                        });
                        while stats.recent_tool_calls.len() > MAX_RECENT_TOOL_CALLS {
                            stats.recent_tool_calls.pop_back();
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

#[derive(Clone, Copy)]
struct ScrollbarDrag {
    start_mouse_y: f32,
    start_scroll_y: f32,
    /// Content pixels scrolled per pixel of thumb movement.
    content_per_track_px: f32,
}

struct SessionMonitor {
    body_scroll: ScrollHandle,
    scrollbar_drag: Option<ScrollbarDrag>,
    session_path: Option<PathBuf>,
    session_modified: Option<SystemTime>,
    /// Stay on the chosen session instead of following the most recently active one.
    pinned: bool,
    tail: Option<SessionTail>,
    stats: Stats,
    agents: Vec<AgentRun>,
    /// Main-session turn under the pointer in the costliest-turns list.
    hovered_turn: Option<usize>,
    picker_open: bool,
    /// When a click outside last closed the picker, so the same click on the chip doesn't
    /// immediately reopen it.
    picker_dismissed_at: Option<Instant>,
    picker_sessions: Vec<SessionFile>,
    summaries: HashMap<PathBuf, SessionSummary>,
    summaries_in_flight: bool,
    prices: PriceTable,
    last_price_fetch: Option<Instant>,
    price_fetch_in_flight: bool,
    price_fetch_error: Option<String>,
}

impl SessionMonitor {
    fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let mut this = Self {
            body_scroll: ScrollHandle::new(),
            scrollbar_drag: None,
            session_path: None,
            session_modified: None,
            pinned: false,
            tail: None,
            stats: Stats::default(),
            agents: Vec::new(),
            hovered_turn: None,
            picker_open: false,
            picker_dismissed_at: None,
            picker_sessions: Vec::new(),
            summaries: HashMap::new(),
            summaries_in_flight: false,
            prices: PriceTable::load(),
            last_price_fetch: None,
            price_fetch_in_flight: false,
            price_fetch_error: None,
        };
        this.refresh();
        this.refresh_prices(cx);

        cx.spawn(async move |this, cx| loop {
            cx.background_executor()
                .timer(Duration::from_millis(750))
                .await;
            let result = this.update(cx, |this, cx| {
                if this.refresh() {
                    this.refresh_prices(cx);
                }
                cx.notify();
            });
            if result.is_err() {
                return;
            }
        })
        .detach();

        // Wayland only reports displays once the window is mapped, so the launch size may have
        // been a guess. Resize to fit as soon as the display is known.
        if !display_known_at_launch(cx) {
            cx.spawn_in(window, async move |this, cx| {
                for _ in 0..40 {
                    cx.background_executor()
                        .timer(Duration::from_millis(50))
                        .await;
                    let fitted = this.update_in(cx, |_, window, cx| {
                        let display = window
                            .display(cx)
                            .or_else(|| cx.displays().into_iter().next());
                        let Some(display) = display else {
                            return false;
                        };
                        let target = fit_window_to(display.visible_bounds().size);
                        if window.bounds().size != target {
                            window.resize(target);
                        }
                        true
                    });
                    if fitted.unwrap_or(true) {
                        return;
                    }
                }
            })
            .detach();
        }

        this
    }

    fn open_session(&mut self, path: PathBuf) {
        self.session_modified = sessions::modified(&path);
        self.session_path = Some(path.clone());
        self.tail = Some(SessionTail::new(path));
        self.stats = Stats::default();
        self.agents.clear();
        self.hovered_turn = None;
    }

    /// While following, moves to the most recently active session once the current one has
    /// gone quiet; returns whether it changed.
    fn rescan(&mut self) -> bool {
        if self.pinned && self.session_path.is_some() {
            return false;
        }
        let Some(newest) = sessions::list_sessions().into_iter().next() else {
            return false;
        };
        let follow = sessions::should_follow(
            self.session_path.as_deref(),
            self.session_path.as_deref().and_then(sessions::modified),
            &newest.path,
            SystemTime::now(),
        );
        if follow && self.session_path.as_ref() != Some(&newest.path) {
            self.open_session(newest.path);
            return true;
        }
        false
    }

    /// Returns whether the active session changed.
    fn refresh(&mut self) -> bool {
        let switched = self.rescan();
        if let Some(tail) = &mut self.tail {
            let _ = tail.poll(&mut self.stats);
        }
        if let Some(path) = &self.session_path {
            self.session_modified = sessions::modified(path);
            sessions::sync_agents(path, &mut self.agents);
        }
        switched
    }

    /// The main session followed by each subagent run.
    fn all_stats(&self) -> impl Iterator<Item = &Stats> {
        std::iter::once(&self.stats).chain(self.agents.iter().map(|a| &a.stats))
    }

    fn select_session(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        self.picker_open = false;
        self.pinned = true;
        if self.session_path.as_ref() != Some(&path) {
            self.open_session(path);
            self.refresh();
            self.refresh_prices(cx);
        }
        cx.notify();
    }

    fn toggle_picker(&mut self, cx: &mut Context<Self>) {
        let just_dismissed = self
            .picker_dismissed_at
            .is_some_and(|at| at.elapsed() < Duration::from_millis(300));
        if just_dismissed && !self.picker_open {
            self.picker_dismissed_at = None;
            return;
        }
        self.picker_open = !self.picker_open;
        if self.picker_open {
            self.picker_sessions = sessions::list_sessions();
            self.picker_sessions.truncate(PICKER_SESSIONS);
            self.summarize_picker_sessions(cx);
        }
        cx.notify();
    }

    /// Reads the picker's sessions in the background for their project and cost, skipping any
    /// already summarized at their current length.
    fn summarize_picker_sessions(&mut self, cx: &mut Context<Self>) {
        if self.summaries_in_flight {
            return;
        }
        let stale: Vec<PathBuf> = self
            .picker_sessions
            .iter()
            .filter(|s| {
                self.summaries
                    .get(&s.path)
                    .is_none_or(|sum| sum.len != sessions::file_len(&s.path))
            })
            .map(|s| s.path.clone())
            .collect();
        if stale.is_empty() {
            return;
        }
        self.summaries_in_flight = true;
        let prices = self.prices.clone();
        let task = cx.background_executor().spawn(async move {
            stale
                .into_iter()
                .map(|path| {
                    let summary = sessions::summarize(&path, &prices);
                    (path, summary)
                })
                .collect::<Vec<_>>()
        });
        cx.spawn(async move |this, cx| {
            let results = task.await;
            this.update(cx, |this, cx| {
                this.summaries_in_flight = false;
                this.summaries.extend(results);
                cx.notify();
            })
        })
        .detach();
    }

    fn price_source_label(&self) -> String {
        let base = match self.prices.source {
            PriceSource::Live { fetched_at } => {
                return format!("live API prices, fetched {}", format_ago(fetched_at));
            }
            PriceSource::Cached { fetched_at } => {
                format!("cached API prices from {}", format_ago(fetched_at))
            }
            PriceSource::BuiltIn => "built-in API prices (Oct 2026)".to_string(),
        };
        if self.price_fetch_in_flight {
            format!("{base}, updating")
        } else if self.price_fetch_error.is_some() {
            format!("{base}, offline")
        } else {
            base
        }
    }

    /// Fetches the live pricing page in the background, at most once per
    /// `PRICE_REFETCH_INTERVAL`; on failure the current (cached or built-in) prices stay.
    fn refresh_prices(&mut self, cx: &mut Context<Self>) {
        let due = self
            .last_price_fetch
            .is_none_or(|at| at.elapsed() >= PRICE_REFETCH_INTERVAL);
        if self.price_fetch_in_flight || !due {
            return;
        }
        self.price_fetch_in_flight = true;
        self.last_price_fetch = Some(Instant::now());
        let fetch = cx.background_executor().spawn(async {
            let table = PriceTable::fetch()?;
            // A failed cache write only costs the next offline launch its fresh prices.
            let _ = table.save_cache();
            anyhow::Ok(table)
        });
        cx.spawn(async move |this, cx| {
            let result = fetch.await;
            this.update(cx, |this, cx| {
                this.price_fetch_in_flight = false;
                match result {
                    Ok(table) => {
                        this.prices = table;
                        this.price_fetch_error = None;
                    }
                    Err(err) => this.price_fetch_error = Some(format!("{err:#}")),
                }
                cx.notify();
            })
        })
        .detach();
    }
}

// ---------- tooltip ----------

struct Tip(Vec<SharedString>);

impl Render for Tip {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let mut col = div()
            .flex()
            .flex_col()
            .px_2()
            .py_1()
            .bg(rgb(GRID))
            .border_1()
            .border_color(rgb(BASELINE))
            .rounded_md()
            .text_xs()
            .text_color(rgb(INK));
        for (i, line) in self.0.iter().enumerate() {
            let color = if i == 0 { INK } else { INK_2 };
            col = col.child(div().text_color(rgb(color)).child(line.clone()));
        }
        col
    }
}

fn tip_builder(lines: Vec<String>) -> impl Fn(&mut Window, &mut App) -> AnyView + 'static {
    let lines: Vec<SharedString> = lines.into_iter().map(Into::into).collect();
    move |_, cx| cx.new(|_| Tip(lines.clone())).into()
}

// ---------- building blocks ----------

fn stat_tile(label: &str, value: String, swatch: Option<u32>) -> gpui::Div {
    let mut label_row = div().flex().flex_row().items_center().gap_2();
    if let Some(color) = swatch {
        label_row = label_row.child(div().size(px(8.0)).rounded_sm().bg(rgb(color)));
    }
    label_row = label_row.child(
        div()
            .text_xs()
            .text_color(rgb(MUTED))
            .child(label.to_string()),
    );

    div()
        .flex()
        .flex_col()
        .gap_1()
        .px_4()
        .py_3()
        .bg(rgb(SURFACE))
        .rounded_md()
        .border_1()
        .border_color(rgb(GRID))
        .min_w(px(132.0))
        .child(label_row)
        .child(div().text_xl().text_color(rgb(INK)).child(value))
}

fn tile_detail(text: String) -> impl IntoElement {
    div().text_xs().text_color(rgb(MUTED)).child(text)
}

fn legend(items: &[(&str, u32)]) -> impl IntoElement {
    let mut row = div()
        .flex()
        .flex_row()
        .flex_wrap()
        .gap_x_3()
        .gap_y_1()
        .items_center();
    for (label, color) in items {
        row = row.child(
            div()
                .flex()
                .flex_row()
                .items_center()
                .gap_1()
                .child(div().size(px(8.0)).rounded_sm().bg(rgb(*color)))
                .child(
                    div()
                        .text_xs()
                        .text_color(rgb(INK_2))
                        .child(label.to_string()),
                ),
        );
    }
    row
}

fn chart_card(
    title: &str,
    subtitle: String,
    legend_items: Option<&[(&str, u32)]>,
    body: impl IntoElement,
) -> gpui::Div {
    // Wraps so a narrow card moves the legend under the title instead of overflowing.
    let mut header = div()
        .flex()
        .flex_row()
        .flex_wrap()
        .justify_between()
        .items_start()
        .gap_x_4()
        .gap_y_2()
        .child(
            div()
                .flex()
                .flex_col()
                .child(
                    div()
                        .text_sm()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(rgb(INK))
                        .child(title.to_string()),
                )
                .child(div().text_xs().text_color(rgb(MUTED)).child(subtitle)),
        );
    if let Some(items) = legend_items {
        header = header.child(legend(items));
    }

    div()
        .flex()
        .flex_col()
        .gap_3()
        .p_4()
        .bg(rgb(SURFACE))
        .rounded_md()
        .border_1()
        .border_color(rgb(GRID))
        // Anything that still doesn't fit is clipped to the card rather than drawn over its
        // neighbors.
        .overflow_hidden()
        .child(header)
        .child(body)
}

/// One column of a bar chart: stacked segments listed top-to-bottom, plus tooltip lines.
struct Col {
    segs: Vec<(f64, u32)>,
    tip: Vec<String>,
}

fn bar_chart(
    id: &'static str,
    cols: Vec<Col>,
    max: f64,
    y_labels: [String; 3],
    x_labels: (String, String),
    highlight: Option<usize>,
) -> impl IntoElement {
    let mut bars = div()
        .absolute()
        .inset_0()
        .flex()
        .flex_row()
        .items_end()
        .gap_0p5();

    let empty = cols.is_empty();
    for (i, col) in cols.into_iter().enumerate() {
        let mut stack = div()
            .id((id, i))
            .flex_1()
            .h_full()
            .flex()
            .flex_col()
            .justify_end()
            .gap_0p5()
            .hover(|s| s.bg(rgb(HOVER_WASH)))
            .tooltip(tip_builder(col.tip));
        if highlight == Some(i) {
            stack = stack.bg(rgb(BASELINE)).rounded_t_sm();
        }
        let visible: Vec<(f64, u32)> = col.segs.into_iter().filter(|(v, _)| *v > 0.0).collect();
        for (n, (value, color)) in visible.into_iter().enumerate() {
            let h = (value / max * (PLOT_H - 6.0)).max(2.0);
            let mut seg = div().w_full().h(px(h as f32)).bg(rgb(color));
            if n == 0 {
                seg = seg.rounded_t_sm();
            }
            stack = stack.child(seg);
        }
        bars = bars.child(stack);
    }

    let gridline = |bottom: f32| {
        div()
            .absolute()
            .left_0()
            .right_0()
            .bottom(px(bottom))
            .h(px(1.0))
            .bg(rgb(GRID))
    };

    let mut plot = div()
        .relative()
        .flex_1()
        .h(px(PLOT_H as f32))
        .child(gridline(PLOT_H as f32 / 2.0))
        .child(gridline(PLOT_H as f32 - 1.0))
        .child(gridline(0.0).bg(rgb(BASELINE)))
        .child(bars);
    if empty {
        plot = plot.child(
            div()
                .absolute()
                .inset_0()
                .flex()
                .items_center()
                .justify_center()
                .text_xs()
                .text_color(rgb(MUTED))
                .child("waiting for turns..."),
        );
    }

    let y_axis = div()
        .flex()
        .flex_col()
        .justify_between()
        .items_end()
        .h(px(PLOT_H as f32))
        .w(px(44.0))
        .text_xs()
        .text_color(rgb(MUTED))
        .child(y_labels[0].clone())
        .child(y_labels[1].clone())
        .child(y_labels[2].clone());

    div()
        .flex()
        .flex_col()
        .gap_1()
        .child(div().flex().flex_row().gap_2().child(y_axis).child(plot))
        .child(
            div()
                .flex()
                .flex_row()
                .justify_between()
                .pl(px(52.0))
                .text_xs()
                .text_color(rgb(MUTED))
                .child(x_labels.0)
                .child(x_labels.1),
        )
}

fn format_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_usd(amount: f64) -> String {
    if amount > 0.0 && amount < 0.01 {
        format!("${amount:.4}")
    } else {
        format!("${amount:.2}")
    }
}

fn format_ago(at: DateTime<Utc>) -> String {
    let secs = (Utc::now() - at).num_seconds().max(0);
    match secs {
        0..60 => "just now".to_string(),
        60..3600 => format!("{}m ago", secs / 60),
        3600..86400 => format!("{}h ago", secs / 3600),
        _ => format!("{}d ago", secs / 86400),
    }
}

fn format_rate(per_mtok: f64) -> String {
    if per_mtok.fract() == 0.0 {
        format!("${per_mtok:.0}")
    } else {
        format!("${per_mtok:.2}")
    }
}

fn model_panel(stats: &Stats, prices: &PriceTable) -> impl IntoElement {
    let muted = |text: String| div().text_xs().text_color(rgb(MUTED)).child(text);
    let Some(current) = stats.current_turn() else {
        return div().child(muted("no model activity yet".to_string()));
    };
    let model = stats.model_of(current).unwrap_or_default();
    let price = prices.lookup(model);
    let display = |id: &str| {
        prices
            .lookup(id)
            .map_or_else(|| id.to_string(), |p| p.name.clone())
    };

    let mut panel = div().flex().flex_col().gap_3().child(
        div()
            .flex()
            .flex_col()
            .child(div().text_lg().text_color(rgb(INK)).child(display(model)))
            .child(muted(model.to_string())),
    );

    // Context: the latest prompt is everything the model had to read on its last turn.
    let used = current.prompt_total();
    let context = match pricing::context_window(model) {
        Some(window) => {
            let frac = (used as f32 / window as f32).clamp(0.0, 1.0);
            let color = match frac {
                f if f >= 0.9 => CRITICAL,
                f if f >= 0.75 => C_OUTPUT,
                _ => C_READ,
            };
            div()
                .flex()
                .flex_col()
                .gap_1()
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .justify_between()
                        .child(muted("Context".to_string()))
                        .child(div().text_xs().text_color(rgb(INK_2)).child(format!(
                            "{} / {} ({:.0}%)",
                            format_tokens(used),
                            format_tokens(window),
                            frac * 100.0
                        ))),
                )
                .child(
                    div()
                        .h(px(8.0))
                        .w_full()
                        .bg(rgb(GRID))
                        .rounded_sm()
                        .child(div().h_full().w(relative(frac)).bg(rgb(color)).rounded_sm()),
                )
        }
        None => div().child(muted(format!(
            "Context {} (window unknown)",
            format_tokens(used)
        ))),
    };
    panel = panel.child(context);

    let mut details = div().flex().flex_col().gap_1();
    details = details.child(muted(match price {
        Some(p) => format!(
            "{} in, {} out, {} cache read per MTok",
            format_rate(p.input),
            format_rate(p.output),
            format_rate(p.cache_read)
        ),
        None => "no API price for this model".to_string(),
    }));
    let output = stats.sum(|t| t.output);
    let thinking = stats.sum(|t| t.thinking);
    if output > 0 {
        details = details.child(muted(format!(
            "Thinking {} of {} output ({:.0}%)",
            format_tokens(thinking),
            format_tokens(output),
            thinking as f64 / output as f64 * 100.0
        )));
    }
    let fast = stats.turns.iter().filter(|t| t.fast).count();
    if fast > 0 {
        details = details.child(muted(format!("Fast mode on {fast} turns")));
    }
    let us_geo = stats.turns.iter().filter(|t| t.us_geo).count();
    if us_geo > 0 {
        details = details.child(muted(format!("US-only inference on {us_geo} turns (1.1x)")));
    }
    panel = panel.child(details);

    let mix = stats.model_mix(prices);
    if mix.len() > 1 {
        let mut rows = div()
            .flex()
            .flex_col()
            .gap_1()
            .child(muted("Model mix".to_string()));
        for row in mix {
            rows = rows.child(
                div()
                    .flex()
                    .flex_row()
                    .gap_2()
                    .text_xs()
                    .child(
                        div()
                            .flex_1()
                            .text_color(rgb(INK_2))
                            .child(display(&row.model)),
                    )
                    .child(
                        div()
                            .text_color(rgb(MUTED))
                            .child(format!("{} turns", row.turns)),
                    )
                    .child(
                        div()
                            .w(px(64.0))
                            .flex()
                            .justify_end()
                            .text_color(rgb(INK))
                            .child(row.cost.map_or_else(|| "-".to_string(), format_usd)),
                    ),
            );
        }
        panel = panel.child(rows);
    }
    panel
}

fn tool_mix(stats: &Stats) -> impl IntoElement {
    let mut rows: Vec<(&String, &ToolAgg)> = stats.tools.iter().collect();
    rows.sort_by(|a, b| b.1.count.cmp(&a.1.count).then(a.0.cmp(b.0)));
    rows.truncate(8);
    let max = rows.first().map_or(1, |r| r.1.count).max(1) as f32;

    let mut list = div().flex().flex_col().gap_1();
    if rows.is_empty() {
        list = list.child(
            div()
                .text_xs()
                .text_color(rgb(MUTED))
                .child("no tool calls yet"),
        );
    }
    for (i, (name, agg)) in rows.into_iter().enumerate() {
        let avg = if agg.timed > 0 {
            format!("{}ms", agg.total_ms / agg.timed as i64)
        } else {
            "-".to_string()
        };
        let tip = vec![
            name.to_string(),
            format!("{} calls", agg.count),
            format!("avg latency {avg}"),
            format!("{} errors", agg.errors),
        ];
        list = list.child(
            div()
                .id(("toolmix", i))
                .flex()
                .flex_row()
                .items_center()
                .gap_2()
                .h(px(22.0))
                .hover(|s| s.bg(rgb(HOVER_WASH)))
                .tooltip(tip_builder(tip))
                .child(
                    div()
                        .w(px(96.0))
                        .text_xs()
                        .text_color(rgb(INK_2))
                        .child(name.to_string()),
                )
                .child(
                    div().flex_1().h(px(8.0)).child(
                        div()
                            .h_full()
                            .w(relative(agg.count as f32 / max))
                            .bg(rgb(C_READ))
                            .rounded_r_sm(),
                    ),
                )
                .child(
                    div()
                        .w(px(36.0))
                        .flex()
                        .justify_end()
                        .text_xs()
                        .text_color(rgb(INK))
                        .child(agg.count.to_string()),
                ),
        );
    }
    list
}

impl Render for SessionMonitor {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let stats = &self.stats;
        let prices = &self.prices;
        // Token and cost tiles cover the whole session, subagents included; charts and tool
        // panels show the main conversation.
        let total =
            |f: &dyn Fn(&TurnPoint) -> u64| -> u64 { self.all_stats().map(|s| s.sum(f)).sum() };
        let all_read = total(&|t| t.cache_read);
        let all_prompt = total(&|t| t.prompt_total());
        let all_hit_rate = if all_prompt == 0 {
            0.0
        } else {
            all_read as f64 / all_prompt as f64 * 100.0
        };
        let main_cost = stats.total_cost_usd(prices);
        let agent_costs: Vec<f64> = self
            .agents
            .iter()
            .map(|a| a.stats.total_cost_usd(prices))
            .collect();
        let agent_cost: f64 = agent_costs.iter().sum();
        let partial = self.all_stats().any(|s| s.unpriced_turns(prices) > 0);
        let agent_turns: usize = self.agents.iter().map(|a| a.stats.turns.len()).sum();

        let mut cost_tile = stat_tile(
            if partial {
                "API COST (PARTIAL)"
            } else {
                "API COST"
            },
            format_usd(main_cost + agent_cost),
            None,
        );
        let mut cost_tip = vec![format!("Main conversation {}", format_usd(main_cost))];
        if !self.agents.is_empty() {
            let n = self.agents.len();
            cost_tile = cost_tile.child(tile_detail(format!(
                "main {} · {n} agent{} {}",
                format_usd(main_cost),
                if n == 1 { "" } else { "s" },
                format_usd(agent_cost)
            )));
            for (agent, cost) in self.agents.iter().zip(&agent_costs) {
                cost_tip.push(format!(
                    "{}: {} turns, {}",
                    agent.label(),
                    agent.stats.turns.len(),
                    format_usd(*cost)
                ));
            }
        }
        let mut turns_tile = stat_tile("TURNS", stats.turns.len().to_string(), None);
        if agent_turns > 0 {
            turns_tile = turns_tile.child(tile_detail(format!("+{agent_turns} in agents")));
        }

        let n_turns = stats.turns.len();
        let start = n_turns.saturating_sub(CHART_TURNS);
        let window_turns = &stats.turns[start..];
        let x_labels = (
            if n_turns == 0 {
                String::new()
            } else {
                format!("turn {}", start + 1)
            },
            if n_turns == 0 {
                String::new()
            } else {
                format!("turn {n_turns}")
            },
        );

        // Hovering a costliest-turns row marks that turn in both charts.
        let highlight = self
            .hovered_turn
            .and_then(|ix| ix.checked_sub(start))
            .filter(|&ix| ix < window_turns.len());

        // Chart A: prompt composition per turn (stacked).
        let max_prompt = window_turns
            .iter()
            .map(|t| t.prompt_total())
            .max()
            .unwrap_or(1)
            .max(1) as f64;
        let composition_cols: Vec<Col> = window_turns
            .iter()
            .enumerate()
            .map(|(i, t)| Col {
                segs: vec![
                    (t.input as f64, C_INPUT),
                    (t.cache_create as f64, C_WRITE),
                    (t.cache_read as f64, C_READ),
                ],
                tip: vec![
                    format!("Turn {}", start + i + 1),
                    format!("Cache read {}", format_tokens(t.cache_read)),
                    format!("Cache write {}", format_tokens(t.cache_create)),
                    format!("Uncached input {}", format_tokens(t.input)),
                    format!("Output {}", format_tokens(t.output)),
                    format!("Hit rate {:.0}%", t.hit_rate()),
                    match stats.turn_cost(t, prices) {
                        Some(cost) => format!("Cost {}", format_usd(cost)),
                        None => "Cost unknown (unpriced model)".to_string(),
                    },
                ],
            })
            .collect();
        let composition = bar_chart(
            "composition",
            composition_cols,
            max_prompt,
            [
                format_tokens(max_prompt as u64),
                format_tokens(max_prompt as u64 / 2),
                "0".to_string(),
            ],
            x_labels.clone(),
            highlight,
        );

        // Chart B: cache hit rate per turn.
        let hit_cols: Vec<Col> = window_turns
            .iter()
            .enumerate()
            .map(|(i, t)| Col {
                segs: vec![(t.hit_rate(), C_READ)],
                tip: vec![
                    format!("Turn {}", start + i + 1),
                    format!("Hit rate {:.0}%", t.hit_rate()),
                    format!("Prompt {}", format_tokens(t.prompt_total())),
                ],
            })
            .collect();
        let hit_chart = bar_chart(
            "hitrate",
            hit_cols,
            100.0,
            ["100%".to_string(), "50%".to_string(), "0%".to_string()],
            x_labels,
            highlight,
        );
        let low_turns = window_turns.iter().filter(|t| t.hit_rate() < 50.0).count();

        // Chart D: recent tool calls with latency bars.
        let tool_count = stats.recent_tool_calls.len();

        let body = div()
            .id("dashboard")
            .size_full()
            .overflow_y_scroll()
            .track_scroll(&self.body_scroll)
            .flex()
            .flex_col()
            .p_4()
            .pr(px(SCROLLBAR_W + 8.0))
            .gap_3()
            .child(self.session_header(cx))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap_3()
                    .flex_wrap()
                    .child(stat_tile(
                        "UNCACHED INPUT",
                        format_tokens(total(&|t| t.input)),
                        Some(C_INPUT),
                    ))
                    .child(stat_tile(
                        "OUTPUT",
                        format_tokens(total(&|t| t.output)),
                        Some(C_OUTPUT),
                    ))
                    .child(stat_tile(
                        "CACHE READ",
                        format_tokens(all_read),
                        Some(C_READ),
                    ))
                    .child(stat_tile(
                        "CACHE WRITE",
                        format_tokens(total(&|t| t.cache_create)),
                        Some(C_WRITE),
                    ))
                    .child(stat_tile(
                        "CACHE HIT RATE",
                        format!("{all_hit_rate:.1}%"),
                        None,
                    ))
                    .child(cost_tile.id("cost-tile").tooltip(tip_builder(cost_tip)))
                    .child(turns_tile)
                    .child(stat_tile(
                        "TOOL CALLS",
                        stats.total_tool_calls().to_string(),
                        None,
                    )),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap_3()
                    .child(
                        chart_card(
                            "Prompt size per turn",
                            format!("last {} turns, tokens by cache status", window_turns.len()),
                            Some(&[
                                ("Cache read", C_READ),
                                ("Cache write", C_WRITE),
                                ("Uncached input", C_INPUT),
                            ]),
                            composition,
                        )
                        .flex_1()
                        .min_w(px(0.0)),
                    )
                    .child(
                        chart_card(
                            "Cache hit rate per turn",
                            format!("{low_turns} of {} turns under 50%", window_turns.len()),
                            None,
                            hit_chart,
                        )
                        .flex_1()
                        .min_w(px(0.0)),
                    ),
            )
            .child(self.costliest_card(cx))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap_3()
                    .flex_1()
                    // Below this the tool lists are too short to read; scroll instead.
                    .min_h(px(BOTTOM_ROW_MIN_H))
                    .child(
                        chart_card(
                            "Model",
                            self.price_source_label(),
                            None,
                            model_panel(stats, prices),
                        )
                        .w(px(300.0)),
                    )
                    .child(
                        chart_card(
                            "Tool mix",
                            "calls by tool, top 8".to_string(),
                            None,
                            tool_mix(stats),
                        )
                        .w(px(340.0)),
                    )
                    .child(
                        chart_card(
                            "Recent tool calls",
                            format!("latency, newest first ({tool_count})"),
                            None,
                            uniform_list(
                                "tool-calls",
                                tool_count,
                                cx.processor(
                                    move |this: &mut SessionMonitor, range, _window, _cx| {
                                        let entries = &this.stats.recent_tool_calls;
                                        let max_ms = entries
                                            .iter()
                                            .filter_map(|e| e.latency_ms)
                                            .max()
                                            .unwrap_or(1)
                                            .max(1)
                                            as f32;
                                        let mut rows = Vec::new();
                                        for ix in range {
                                            let Some(entry) = entries.get(ix) else {
                                                continue;
                                            };
                                            let frac = entry
                                                .latency_ms
                                                .map_or(0.0, |ms| ms.max(0) as f32 / max_ms);
                                            let label = match (entry.is_error, entry.latency_ms) {
                                                (true, Some(ms)) => format!("x error  {ms}ms"),
                                                (true, None) => "x error".to_string(),
                                                (false, Some(ms)) => format!("{ms}ms"),
                                                (false, None) => "-".to_string(),
                                            };
                                            let label_color =
                                                if entry.is_error { CRITICAL } else { INK_2 };
                                            rows.push(
                                                div()
                                                    .id(ix)
                                                    .flex()
                                                    .flex_row()
                                                    .items_center()
                                                    .gap_3()
                                                    .h(px(24.0))
                                                    .hover(|s| s.bg(rgb(HOVER_WASH)))
                                                    .child(
                                                        div()
                                                            .w(px(110.0))
                                                            .text_xs()
                                                            .text_color(rgb(INK_2))
                                                            .child(entry.name.clone()),
                                                    )
                                                    .child(
                                                        div().flex_1().h(px(6.0)).child(
                                                            div()
                                                                .h_full()
                                                                .w(relative(frac))
                                                                .bg(rgb(C_READ))
                                                                .rounded_r_sm(),
                                                        ),
                                                    )
                                                    .child(
                                                        div()
                                                            .w(px(110.0))
                                                            .flex()
                                                            .justify_end()
                                                            .text_xs()
                                                            .text_color(rgb(label_color))
                                                            .child(label),
                                                    ),
                                            );
                                        }
                                        rows
                                    },
                                ),
                            )
                            .h_full(),
                        )
                        .flex_1()
                        .min_w(px(0.0))
                        .min_h(px(0.0)),
                    ),
            );

        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(PAGE))
            .text_color(rgb(INK))
            .children(title_bar(window))
            .child(
                div()
                    .relative()
                    .flex_1()
                    .min_h(px(0.0))
                    .child(body)
                    .children(self.scrollbar(cx)),
            )
            .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _, cx| {
                let Some(drag) = this.scrollbar_drag else {
                    return;
                };
                if event.pressed_button != Some(MouseButton::Left) {
                    this.scrollbar_drag = None;
                    return;
                }
                let delta = f32::from(event.position.y) - drag.start_mouse_y;
                this.set_scroll_y(drag.start_scroll_y + delta * drag.content_per_track_px);
                cx.notify();
            }))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _, _, _| this.scrollbar_drag = None),
            )
    }
}

/// A row in the costliest-turns card.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Costly {
    /// Index into the main session's turns.
    Turn(usize),
    /// Index into the session's subagent runs, ranked by the run's total cost.
    Agent(usize),
}

/// The `n` most expensive main-session turns and subagent runs, most expensive first.
fn costliest(
    stats: &Stats,
    agent_costs: &[f64],
    prices: &PriceTable,
    n: usize,
) -> Vec<(Costly, f64)> {
    let mut rows: Vec<(Costly, f64)> = stats
        .turns
        .iter()
        .enumerate()
        .filter_map(|(ix, t)| Some((Costly::Turn(ix), stats.turn_cost(t, prices)?)))
        .chain(
            agent_costs
                .iter()
                .enumerate()
                .map(|(ix, cost)| (Costly::Agent(ix), *cost)),
        )
        .filter(|(_, cost)| *cost > 0.0)
        .collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    rows.truncate(n);
    rows
}

fn short_model_name(model: Option<&str>, prices: &PriceTable) -> String {
    let Some(model) = model else {
        return "-".to_string();
    };
    match prices.lookup(model) {
        Some(price) => price.name.trim_start_matches("Claude ").to_string(),
        None => model.to_string(),
    }
}

fn local_time(at: Option<DateTime<Utc>>, fmt: &str) -> String {
    at.map(|at| at.with_timezone(&chrono::Local).format(fmt).to_string())
        .unwrap_or_default()
}

impl SessionMonitor {
    /// The session chip (opens the session picker), the follow/pin toggle, and a subagent note.
    fn session_header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let label = match &self.session_path {
            Some(path) => format!(
                "{} · {}",
                sessions::project_name(self.stats.cwd.as_deref(), path),
                sessions::short_id(path)
            ),
            None => "no session found".to_string(),
        };
        let live = sessions::is_live(self.session_modified);
        let chip = div()
            .id("session-chip")
            .flex()
            .flex_row()
            .items_center()
            .gap_2()
            .px_2()
            .py_1()
            .rounded_md()
            .border_1()
            .border_color(rgb(GRID))
            .bg(rgb(SURFACE))
            .cursor_pointer()
            .hover(|s| s.bg(rgb(HOVER_WASH)))
            .tooltip(tip_builder(vec![
                "Choose a session".to_string(),
                match &self.session_path {
                    Some(path) => path.display().to_string(),
                    None => "~/.claude/projects is empty".to_string(),
                },
            ]))
            .on_click(cx.listener(|this, _, _, cx| this.toggle_picker(cx)))
            .child(status_dot(live))
            .child(div().text_xs().text_color(rgb(INK)).child(label))
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(MUTED))
                    .child(if self.picker_open { "▴" } else { "▾" }),
            );

        let (mode, mode_tip) = if self.pinned {
            (
                "PINNED",
                "Staying on this session. Click to follow the latest session.",
            )
        } else {
            (
                "FOLLOWING LATEST",
                "Switches to the newest session once this one has been idle for a minute. \
                 Click to pin this session.",
            )
        };
        let mode_badge = div()
            .id("follow-mode")
            .px_2()
            .py_0p5()
            .rounded_sm()
            .border_1()
            .border_color(rgb(if self.pinned { C_OUTPUT } else { BASELINE }))
            .text_xs()
            .text_color(rgb(if self.pinned { C_OUTPUT } else { INK_2 }))
            .cursor_pointer()
            .hover(|s| s.bg(rgb(HOVER_WASH)))
            .tooltip(tip_builder(vec![mode_tip.to_string()]))
            .on_click(cx.listener(|this, _, _, cx| {
                this.pinned = !this.pinned;
                this.refresh();
                cx.notify();
            }))
            .child(mode);

        let mut row = div()
            .flex()
            .flex_row()
            .flex_wrap()
            .items_center()
            .gap_3()
            .child(chip)
            .child(mode_badge);
        let n = self.agents.len();
        if n > 0 {
            row = row.child(tile_detail(format!(
                "totals include {n} subagent{}",
                if n == 1 { "" } else { "s" }
            )));
        }

        let mut header = div().flex().flex_col().child(row);
        if self.picker_open {
            header = header.child(
                deferred(
                    anchored()
                        .snap_to_window_with_margin(px(8.0))
                        .child(self.session_menu(cx)),
                )
                .with_priority(1),
            );
        }
        header
    }

    fn session_menu(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut menu = div()
            .id("session-menu")
            .occlude()
            .mt_1()
            .w(px(480.0))
            .flex()
            .flex_col()
            .p_1()
            .bg(rgb(SURFACE))
            .border_1()
            .border_color(rgb(BASELINE))
            .rounded_md()
            .shadow_lg()
            .on_mouse_down_out(cx.listener(|this, _, _, cx| {
                this.picker_open = false;
                this.picker_dismissed_at = Some(Instant::now());
                cx.notify();
            }));

        menu = menu.child(
            menu_row("follow-latest", !self.pinned)
                .on_click(cx.listener(|this, _, _, cx| {
                    this.picker_open = false;
                    this.pinned = false;
                    this.refresh();
                    cx.notify();
                }))
                .child(
                    div()
                        .w(px(12.0))
                        .text_xs()
                        .text_color(rgb(C_READ))
                        .child(if self.pinned { "" } else { "✓" }),
                )
                .child(
                    div()
                        .flex_1()
                        .text_xs()
                        .text_color(rgb(INK))
                        .child("Follow the latest session"),
                ),
        );
        menu = menu.child(div().my_1().h(px(1.0)).bg(rgb(GRID)));

        if self.picker_sessions.is_empty() {
            menu = menu.child(
                div()
                    .p_2()
                    .text_xs()
                    .text_color(rgb(MUTED))
                    .child("no sessions in ~/.claude/projects"),
            );
        }
        for (ix, session) in self.picker_sessions.iter().enumerate() {
            let current = self.session_path.as_ref() == Some(&session.path);
            let summary = self.summaries.get(&session.path);
            let project =
                sessions::project_name(summary.and_then(|s| s.cwd.as_deref()), &session.path);
            let cost = match summary {
                Some(s) if s.partial => format!("{}+", format_usd(s.cost)),
                Some(s) => format_usd(s.cost),
                None => "…".to_string(),
            };
            let path = session.path.clone();
            menu = menu.child(
                menu_row(("session", ix), current)
                    .on_click(
                        cx.listener(move |this, _, _, cx| this.select_session(path.clone(), cx)),
                    )
                    .child(
                        div()
                            .w(px(12.0))
                            .child(status_dot(sessions::is_live(Some(session.modified)))),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(0.0))
                            .truncate()
                            .text_xs()
                            .text_color(rgb(INK))
                            .child(project),
                    )
                    .child(
                        div()
                            .w(px(72.0))
                            .text_xs()
                            .text_color(rgb(MUTED))
                            .child(session.short_id()),
                    )
                    .child(
                        div()
                            .w(px(64.0))
                            .text_xs()
                            .text_color(rgb(MUTED))
                            .child(format_ago(DateTime::<Utc>::from(session.modified))),
                    )
                    .child(
                        div()
                            .w(px(64.0))
                            .flex()
                            .justify_end()
                            .text_xs()
                            .text_color(rgb(INK))
                            .child(cost),
                    ),
            );
        }
        menu
    }

    /// The most expensive turns and subagent runs, with what drove each one's cost.
    fn costliest_card(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let stats = &self.stats;
        let prices = &self.prices;
        let agent_costs: Vec<f64> = self
            .agents
            .iter()
            .map(|a| a.stats.total_cost_usd(prices))
            .collect();
        let rows = costliest(stats, &agent_costs, prices, COSTLIEST_ROWS);
        let session_cost = stats.total_cost_usd(prices) + agent_costs.iter().sum::<f64>();
        let top_cost: f64 = rows.iter().map(|(_, cost)| cost).sum();
        let max = rows
            .first()
            .map_or(1.0, |(_, cost)| *cost)
            .max(f64::EPSILON);
        let chart_start = stats.turns.len().saturating_sub(CHART_TURNS);

        let subtitle = if rows.is_empty() {
            "per API call, at API prices".to_string()
        } else {
            format!(
                "top {} = {:.0}% of {} session cost",
                rows.len(),
                top_cost / session_cost.max(f64::EPSILON) * 100.0,
                format_usd(session_cost)
            )
        };

        let mut list = div().flex().flex_col().gap_0p5();
        if rows.is_empty() {
            list = list.child(tile_detail("no priced turns yet".to_string()));
        }
        let cell = |w: f32| div().w(px(w)).flex_none().truncate().text_xs();
        for (rank, (item, cost)) in rows.into_iter().enumerate() {
            let frac = (cost / max) as f32;
            let row = div()
                .id(("costly", rank))
                .flex()
                .flex_row()
                .items_center()
                .gap_3()
                .h(px(24.0))
                .px_1()
                .rounded_sm()
                .hover(|s| s.bg(rgb(HOVER_WASH)));
            let bar = |color: u32| {
                div().flex_1().min_w(px(40.0)).h(px(8.0)).child(
                    div()
                        .h_full()
                        .w(relative(frac))
                        .bg(rgb(color))
                        .rounded_r_sm(),
                )
            };
            let cost_cell = cell(64.0)
                .flex()
                .justify_end()
                .text_color(rgb(INK))
                .child(format_usd(cost));

            let row = match item {
                Costly::Turn(ix) => {
                    let t = &stats.turns[ix];
                    let driver = stats.cost_driver(t, prices);
                    let trigger = stats.trigger_label(t.trigger);
                    let mut tip = vec![
                        format!("Turn {} at {}", ix + 1, local_time(t.at, "%H:%M:%S")),
                        format!("Cost {}", format_usd(cost)),
                        format!("Cache write {}", format_tokens(t.cache_create)),
                        format!("Output {}", format_tokens(t.output)),
                        format!("Uncached input {}", format_tokens(t.input)),
                        format!("Cache read {}", format_tokens(t.cache_read)),
                        trigger.clone(),
                    ];
                    if ix < chart_start {
                        tip.push(format!("older than the last {CHART_TURNS} charted turns"));
                    }
                    row.tooltip(tip_builder(tip))
                        .on_hover(cx.listener(move |this, hovered: &bool, _, cx| {
                            if *hovered {
                                this.hovered_turn = Some(ix);
                            } else if this.hovered_turn == Some(ix) {
                                this.hovered_turn = None;
                            }
                            cx.notify();
                        }))
                        .child(
                            cell(44.0)
                                .text_color(rgb(INK_2))
                                .child(format!("#{}", ix + 1)),
                        )
                        .child(
                            cell(40.0)
                                .text_color(rgb(MUTED))
                                .child(local_time(t.at, "%H:%M")),
                        )
                        .child(bar(driver.map_or(MUTED, CostDriver::color)))
                        .child(cost_cell)
                        .child(
                            cell(88.0)
                                .text_color(rgb(INK_2))
                                .child(short_model_name(stats.model_of(t), prices)),
                        )
                        .child(
                            cell(130.0)
                                .text_color(rgb(INK_2))
                                .child(driver.map_or_else(String::new, |d| d.describe(t))),
                        )
                        .child(cell(150.0).text_color(rgb(MUTED)).child(trigger))
                }
                Costly::Agent(ix) => {
                    let agent = &self.agents[ix];
                    let turns = agent.stats.turns.len();
                    let mut tip = vec![
                        format!("Subagent {} ({})", agent.label(), agent.id),
                        format!("Cost {} over {turns} turns", format_usd(cost)),
                    ];
                    if let Some(description) = &agent.description {
                        tip.push(description.clone());
                    }
                    let started = agent.stats.turns.first().and_then(|t| t.at);
                    row.tooltip(tip_builder(tip))
                        .child(cell(44.0).text_color(rgb(MUTED)).child("⤷ agent"))
                        .child(
                            cell(40.0)
                                .text_color(rgb(MUTED))
                                .child(local_time(started, "%H:%M")),
                        )
                        .child(bar(MUTED))
                        .child(cost_cell)
                        .child(
                            cell(88.0)
                                .text_color(rgb(INK_2))
                                .child(short_model_name(agent.stats.main_model(), prices)),
                        )
                        .child(
                            cell(130.0)
                                .text_color(rgb(INK_2))
                                .child(agent.label().to_string()),
                        )
                        .child(cell(150.0).text_color(rgb(MUTED)).child(format!(
                            "{turns} turns{}",
                            agent
                                .description
                                .as_ref()
                                .map_or_else(String::new, |d| format!(" · {d}"))
                        )))
                }
            };
            list = list.child(row);
        }

        chart_card("Costliest turns", subtitle, None, list)
    }
}

fn status_dot(live: bool) -> impl IntoElement {
    div()
        .size(px(8.0))
        .rounded_full()
        .bg(rgb(if live { C_INPUT } else { MUTED }))
}

fn menu_row(id: impl Into<gpui::ElementId>, selected: bool) -> gpui::Stateful<gpui::Div> {
    let mut row = div()
        .id(id)
        .flex()
        .flex_row()
        .items_center()
        .gap_3()
        .h(px(28.0))
        .px_2()
        .rounded_sm()
        .cursor_pointer()
        .hover(|s| s.bg(rgb(HOVER_WASH)));
    if selected {
        row = row.bg(rgb(GRID));
    }
    row
}

impl SessionMonitor {
    fn set_scroll_y(&self, y: f32) {
        let max = f32::from(self.body_scroll.max_offset().y);
        let offset = self.body_scroll.offset();
        self.body_scroll
            .set_offset(point(offset.x, px(-y.clamp(0.0, max.max(0.0)))));
    }

    /// GPUI scroll containers don't paint a scrollbar, so draw one over the dashboard's right
    /// edge: drag the thumb, or click the track to jump there.
    fn scrollbar(&self, cx: &mut Context<Self>) -> Option<impl IntoElement> {
        let viewport = self.body_scroll.bounds();
        let view_h = f32::from(viewport.size.height);
        let max = f32::from(self.body_scroll.max_offset().y);
        if max <= 0.5 || view_h <= 0.0 {
            return None;
        }
        let scrolled = -f32::from(self.body_scroll.offset().y);
        let thumb_h = (view_h * view_h / (view_h + max)).max(SCROLLBAR_MIN_THUMB);
        let travel = (view_h - thumb_h).max(1.0);
        let thumb_top = scrolled / max * travel;
        let content_per_track_px = max / travel;
        let dragging = self.scrollbar_drag.is_some();
        let track_top = f32::from(viewport.origin.y);

        Some(
            div()
                .id("scrollbar-track")
                .absolute()
                .top_0()
                .bottom_0()
                .right_0()
                .w(px(SCROLLBAR_W))
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, event: &MouseDownEvent, _, cx| {
                        // Center the thumb on the click, then let the press keep dragging.
                        let click = f32::from(event.position.y) - track_top;
                        let target = (click - thumb_h / 2.0) * content_per_track_px;
                        this.set_scroll_y(target);
                        this.scrollbar_drag = Some(ScrollbarDrag {
                            start_mouse_y: f32::from(event.position.y),
                            start_scroll_y: target.clamp(0.0, max),
                            content_per_track_px,
                        });
                        cx.stop_propagation();
                        cx.notify();
                    }),
                )
                .child(
                    div()
                        .id("scrollbar-thumb")
                        .absolute()
                        .top(px(thumb_top))
                        .right(px(2.0))
                        .w(px(SCROLLBAR_W - 4.0))
                        .h(px(thumb_h))
                        .rounded_full()
                        .bg(rgb(if dragging { MUTED } else { BASELINE }))
                        .hover(|s| s.bg(rgb(MUTED)))
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |this, event: &MouseDownEvent, _, cx| {
                                this.scrollbar_drag = Some(ScrollbarDrag {
                                    start_mouse_y: f32::from(event.position.y),
                                    start_scroll_y: scrolled,
                                    content_per_track_px,
                                });
                                cx.stop_propagation();
                                cx.notify();
                            }),
                        ),
                ),
        )
    }
}

/// A draggable title bar, drawn only when the compositor leaves decorations to the app
/// (e.g. WSLg/Weston). Drag to move, double-click to maximize, right-click for the window menu.
fn title_bar(window: &Window) -> Option<impl IntoElement> {
    if matches!(window.window_decorations(), Decorations::Server) {
        return None;
    }
    let maximize_glyph = if window.is_maximized() { "❐" } else { "□" };
    Some(
        div()
            .id("title-bar")
            .flex()
            .flex_row()
            .items_center()
            .justify_between()
            .h(px(TITLE_BAR_H))
            .flex_none()
            .pl_4()
            .bg(rgb(SURFACE))
            .border_b_1()
            .border_color(rgb(GRID))
            .on_mouse_down(MouseButton::Left, |event, window, _| {
                if event.click_count >= 2 {
                    window.zoom_window();
                } else {
                    window.start_window_move();
                }
            })
            .on_mouse_down(MouseButton::Right, |event, window, _| {
                window.show_window_menu(event.position);
            })
            .child(div().text_xs().text_color(rgb(INK_2)).child(WINDOW_TITLE))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .h_full()
                    .child(title_bar_button(
                        "minimize",
                        "–",
                        HOVER_WASH,
                        |window, _| window.minimize_window(),
                    ))
                    .child(title_bar_button(
                        "maximize",
                        maximize_glyph,
                        HOVER_WASH,
                        |window, _| window.zoom_window(),
                    ))
                    .child(title_bar_button("close", "×", CRITICAL, |window, _| {
                        window.remove_window()
                    })),
            ),
    )
}

fn title_bar_button(
    id: &'static str,
    glyph: &'static str,
    hover_bg: u32,
    action: impl Fn(&mut Window, &mut App) + 'static,
) -> impl IntoElement {
    div()
        .id(id)
        .flex()
        .items_center()
        .justify_center()
        .w(px(44.0))
        .h_full()
        .text_sm()
        .text_color(rgb(INK_2))
        .hover(move |s| s.bg(rgb(hover_bg)).text_color(rgb(INK)))
        // Keep the press from reaching the title bar, which would start a window move.
        .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
        .on_click(move |_, window, cx| action(window, cx))
        .child(glyph)
}

fn display_known_at_launch(cx: &App) -> bool {
    cx.primary_display().is_some() || !cx.displays().is_empty()
}

/// Largest window up to the preferred size that fits within `WINDOW_SCREEN_FRACTION` of the screen,
/// without going below the minimum unless the screen itself is smaller.
fn fit_window_to(visible: Size<Pixels>) -> Size<Pixels> {
    let fit = |avail: Pixels, preferred: f32, min: f32| {
        let avail = f32::from(avail);
        px((avail * WINDOW_SCREEN_FRACTION)
            .min(preferred)
            .max(min.min(avail)))
    };
    size(
        fit(visible.width, WINDOW_PREFERRED.0, WINDOW_MIN.0),
        fit(visible.height, WINDOW_PREFERRED.1, WINDOW_MIN.1),
    )
}

fn initial_window_size(cx: &App) -> Size<Pixels> {
    cx.primary_display()
        .or_else(|| cx.displays().into_iter().next())
        .map(|display| fit_window_to(display.visible_bounds().size))
        .unwrap_or_else(|| size(px(WINDOW_FALLBACK.0), px(WINDOW_FALLBACK.1)))
}

fn main() {
    application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, initial_window_size(cx), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                window_min_size: Some(size(px(WINDOW_MIN.0), px(WINDOW_MIN.1))),
                titlebar: Some(TitlebarOptions {
                    title: Some(SharedString::from(WINDOW_TITLE)),
                    ..Default::default()
                }),
                ..Default::default()
            },
            |window, cx| cx.new(|cx| SessionMonitor::new(window, cx)),
        )
        .unwrap();
        cx.activate(true);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(stats: &mut Stats, model: &str, input: u64) -> TurnPoint {
        TurnPoint {
            input,
            model: stats.intern_model(model),
            ..Default::default()
        }
    }

    #[test]
    fn model_mix_and_unpriced_turns() {
        let prices = PriceTable::built_in();
        let mut stats = Stats::default();
        for t in [
            turn(&mut stats, "claude-opus-5-5", 1_000_000),
            turn(&mut stats, "claude-opus-5-5", 1_000_000),
            turn(&mut stats, "claude-sonnet-5-5", 1_000_000),
            turn(&mut stats, "mystery-model", 10),
            // Synthetic, token-free turns are neither a model nor unpriced.
            turn(&mut stats, "<synthetic>", 0),
        ] {
            stats.turns.push(t);
        }
        assert_eq!(
            stats.models,
            ["claude-opus-5-5", "claude-sonnet-5-5", "mystery-model"]
        );
        assert!((stats.total_cost_usd(&prices) - 10.0).abs() < 1e-9);
        assert_eq!(stats.unpriced_turns(&prices), 1);
        let mix = stats.model_mix(&prices);
        assert_eq!(
            (mix[0].model.as_str(), mix[0].turns),
            ("claude-opus-5-5", 2)
        );
        assert!((mix[0].cost.unwrap() - 8.0).abs() < 1e-9);
        assert_eq!(
            mix.iter()
                .find(|r| r.model == "mystery-model")
                .unwrap()
                .cost,
            None
        );
        assert_eq!(
            stats.model_of(stats.current_turn().unwrap()),
            Some("mystery-model")
        );
    }

    fn feed(tail: &mut SessionTail, stats: &mut Stats, lines: &[serde_json::Value]) {
        for line in lines {
            tail.apply_line(line, stats);
        }
    }

    fn assistant(
        id: &str,
        usage: serde_json::Value,
        content: serde_json::Value,
    ) -> serde_json::Value {
        serde_json::json!({
            "type": "assistant",
            "timestamp": "2026-10-05T14:03:00Z",
            "message": { "id": id, "model": "claude-opus-5-5", "usage": usage, "content": content }
        })
    }

    #[test]
    fn turns_record_what_triggered_them() {
        let mut tail = SessionTail::new(PathBuf::from("/dev/null"));
        let mut stats = Stats::default();
        let usage = serde_json::json!({ "input_tokens": 10, "output_tokens": 5 });
        feed(
            &mut tail,
            &mut stats,
            &[
                serde_json::json!({ "type": "user", "cwd": "/home/me/app", "message": { "content": "hi" } }),
                assistant(
                    "m1",
                    usage.clone(),
                    serde_json::json!([
                        { "type": "tool_use", "id": "t1", "name": "Read" }
                    ]),
                ),
                // A second content block of the same message must not consume the next trigger.
                assistant("m1", usage.clone(), serde_json::json!([])),
                serde_json::json!({ "type": "user", "timestamp": "2026-10-05T14:03:01Z",
                "message": { "content": [ { "type": "tool_result", "tool_use_id": "t1" } ] } }),
                assistant("m2", usage.clone(), serde_json::json!([])),
                assistant("m3", usage.clone(), serde_json::json!([])),
                serde_json::json!({ "type": "user", "isMeta": true, "message": { "content": "caveat" } }),
                serde_json::json!({ "type": "system", "subtype": "compact_boundary" }),
                assistant("m4", usage.clone(), serde_json::json!([])),
            ],
        );
        let triggers: Vec<String> = stats
            .turns
            .iter()
            .map(|t| stats.trigger_label(t.trigger))
            .collect();
        assert_eq!(
            triggers,
            [
                "after user prompt",
                "after Read",
                "continuation",
                "after compaction"
            ]
        );
        assert_eq!(stats.cwd.as_deref(), Some("/home/me/app"));
        assert!(stats.turns[0].at.is_some());
    }

    #[test]
    fn costliest_ranks_turns_and_agents_with_their_driver() {
        let prices = PriceTable::built_in();
        let mut stats = Stats::default();
        let model = stats.intern_model("claude-opus-5-5");
        // Opus 5.5: 1M 1h cache writes = $8, 100k output = $2, 1M cache reads = $0.20.
        stats.turns.push(TurnPoint {
            cache_create: 1_000_000,
            cache_create_1h: 1_000_000,
            model,
            ..Default::default()
        });
        stats.turns.push(TurnPoint {
            output: 100_000,
            model,
            ..Default::default()
        });
        stats.turns.push(TurnPoint {
            cache_read: 1_000_000,
            model,
            ..Default::default()
        });
        let mystery = stats.intern_model("mystery");
        stats.turns.push(TurnPoint {
            model: mystery,
            input: 5,
            ..Default::default()
        });

        let rows = costliest(&stats, &[3.0, 0.0], &prices, 3);
        let items: Vec<Costly> = rows.iter().map(|r| r.0).collect();
        assert_eq!(items, [Costly::Turn(0), Costly::Agent(0), Costly::Turn(1)]);
        assert!((rows[0].1 - 8.0).abs() < 1e-9);

        let drivers: Vec<Option<CostDriver>> = stats
            .turns
            .iter()
            .map(|t| stats.cost_driver(t, &prices))
            .collect();
        assert_eq!(
            drivers,
            [
                Some(CostDriver::CacheWrite),
                Some(CostDriver::Output),
                Some(CostDriver::CacheRead),
                None
            ]
        );
        assert_eq!(stats.main_model(), Some("claude-opus-5-5"));
    }
}
