use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Utc};
use gpui::{
    AnyView, App, AppContext as _, Bounds, Context, SharedString, TitlebarOptions, Window,
    WindowBounds, WindowOptions, div, prelude::*, px, relative, rgb, size, uniform_list,
};
use gpui_platform::application;
use serde_json::Value;

const MAX_RECENT_TOOL_CALLS: usize = 200;
const CHART_TURNS: usize = 60;
const PLOT_H: f64 = 120.0;

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

#[derive(Default)]
struct Stats {
    turns: Vec<TurnPoint>,
    turn_index: HashMap<String, usize>,
    tools: HashMap<String, ToolAgg>,
    recent_tool_calls: VecDeque<ToolCallEntry>,
}

impl Stats {
    fn sum(&self, f: impl Fn(&TurnPoint) -> u64) -> u64 {
        self.turns.iter().map(f).sum()
    }

    fn cache_hit_rate(&self) -> f64 {
        let read = self.sum(|t| t.cache_read);
        let total = self.sum(|t| t.prompt_total());
        if total == 0 {
            0.0
        } else {
            read as f64 / total as f64 * 100.0
        }
    }

    fn total_tool_calls(&self) -> u64 {
        self.tools.values().map(|t| t.count).sum()
    }
}

/// Finds the most recently modified session transcript across all Claude Code projects.
fn find_active_session() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let projects_dir = Path::new(&home).join(".claude/projects");
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;

    for project_entry in fs::read_dir(&projects_dir).ok()? {
        let project_entry = project_entry.ok()?;
        if !project_entry.path().is_dir() {
            continue;
        }
        let Ok(entries) = fs::read_dir(project_entry.path()) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            let Ok(modified) = metadata.modified() else {
                continue;
            };
            if newest.as_ref().map_or(true, |(t, _)| modified > *t) {
                newest = Some((modified, path));
            }
        }
    }

    newest.map(|(_, path)| path)
}

struct SessionTail {
    path: PathBuf,
    offset: u64,
    pending_tool_calls: HashMap<String, (String, DateTime<Utc>)>,
}

impl SessionTail {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            offset: 0,
            pending_tool_calls: HashMap::new(),
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

        match entry_type {
            "assistant" => {
                let message = obj.get("message");
                let get = |usage: &Value, key: &str| {
                    usage.get(key).and_then(Value::as_u64).unwrap_or(0)
                };
                if let Some(usage) = message.and_then(|m| m.get("usage")) {
                    let point = TurnPoint {
                        input: get(usage, "input_tokens"),
                        output: get(usage, "output_tokens"),
                        cache_read: get(usage, "cache_read_input_tokens"),
                        cache_create: get(usage, "cache_creation_input_tokens"),
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
            "user" => {
                let message = obj.get("message");
                if let Some(content) = message
                    .and_then(|m| m.get("content"))
                    .and_then(Value::as_array)
                {
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

struct SessionMonitor {
    session_path: Option<PathBuf>,
    tail: Option<SessionTail>,
    stats: Stats,
}

impl SessionMonitor {
    fn new(_window: &mut Window, cx: &mut Context<Self>) -> Self {
        let mut this = Self {
            session_path: None,
            tail: None,
            stats: Stats::default(),
        };
        this.refresh();

        cx.spawn(async move |this, cx| loop {
            cx.background_executor()
                .timer(Duration::from_millis(750))
                .await;
            let result = this.update(cx, |this, cx| {
                this.refresh();
                cx.notify();
            });
            if result.is_err() {
                return;
            }
        })
        .detach();

        this
    }

    fn rescan(&mut self) {
        if let Some(path) = find_active_session() {
            if self.session_path.as_ref() != Some(&path) {
                self.session_path = Some(path.clone());
                self.tail = Some(SessionTail::new(path));
                self.stats = Stats::default();
            }
        }
    }

    fn refresh(&mut self) {
        self.rescan();
        if let Some(tail) = &mut self.tail {
            let _ = tail.poll(&mut self.stats);
        }
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

fn stat_tile(label: &str, value: String, swatch: Option<u32>) -> impl IntoElement {
    let mut label_row = div().flex().flex_row().items_center().gap_2();
    if let Some(color) = swatch {
        label_row = label_row.child(div().size(px(8.0)).rounded_sm().bg(rgb(color)));
    }
    label_row = label_row.child(div().text_xs().text_color(rgb(MUTED)).child(label.to_string()));

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

fn legend(items: &[(&str, u32)]) -> impl IntoElement {
    let mut row = div().flex().flex_row().gap_3().items_center();
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
    let mut header = div()
        .flex()
        .flex_row()
        .justify_between()
        .items_start()
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
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let stats = &self.stats;
        let session_name = self
            .session_path
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "no active session found".to_string());

        let n_turns = stats.turns.len();
        let start = n_turns.saturating_sub(CHART_TURNS);
        let window_turns = &stats.turns[start..];
        let x_labels = (
            if n_turns == 0 { String::new() } else { format!("turn {}", start + 1) },
            if n_turns == 0 { String::new() } else { format!("turn {n_turns}") },
        );

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
        );
        let low_turns = window_turns.iter().filter(|t| t.hit_rate() < 50.0).count();

        // Chart D: recent tool calls with latency bars.
        let tool_count = stats.recent_tool_calls.len();

        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(PAGE))
            .text_color(rgb(INK))
            .p_4()
            .gap_3()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(MUTED))
                    .child(format!("session: {session_name}")),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap_3()
                    .flex_wrap()
                    .child(stat_tile(
                        "UNCACHED INPUT",
                        format_tokens(stats.sum(|t| t.input)),
                        Some(C_INPUT),
                    ))
                    .child(stat_tile(
                        "OUTPUT",
                        format_tokens(stats.sum(|t| t.output)),
                        Some(C_OUTPUT),
                    ))
                    .child(stat_tile(
                        "CACHE READ",
                        format_tokens(stats.sum(|t| t.cache_read)),
                        Some(C_READ),
                    ))
                    .child(stat_tile(
                        "CACHE WRITE",
                        format_tokens(stats.sum(|t| t.cache_create)),
                        Some(C_WRITE),
                    ))
                    .child(stat_tile(
                        "CACHE HIT RATE",
                        format!("{:.1}%", stats.cache_hit_rate()),
                        None,
                    ))
                    .child(stat_tile("TURNS", n_turns.to_string(), None))
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
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap_3()
                    .flex_1()
                    .min_h(px(0.0))
                    .child(
                        chart_card(
                            "Tool mix",
                            "calls by tool, top 8".to_string(),
                            None,
                            tool_mix(stats),
                        )
                        .w(px(380.0)),
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
                                            .max(1) as f32;
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
            )
    }
}

fn main() {
    application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(1200.0), px(900.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(TitlebarOptions {
                    title: Some(SharedString::from("Claude Session Monitor")),
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
