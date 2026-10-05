//! Claude API list prices: parsed from Anthropic's pricing page, cached on disk, with a built-in
//! table as the offline fallback.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context as _, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub const PRICING_URL: &str = "https://platform.claude.com/docs/en/about-claude/pricing.md";
const FETCH_TIMEOUT: Duration = Duration::from_secs(15);
// US-only inference (`inference_geo: "us"`) multiplies every token category.
const US_GEO_MULT: f64 = 1.1;

/// USD per million tokens for one model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelPrice {
    pub name: String,
    pub input: f64,
    pub cache_write_5m: f64,
    pub cache_write_1h: f64,
    pub cache_read: f64,
    pub output: f64,
    /// Fast-mode (input, output) rates; None where fast mode bills at standard rates.
    pub fast: Option<(f64, f64)>,
}

/// Token counts for one API call, as logged by Claude Code.
#[derive(Clone, Copy, Debug, Default)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write_5m: u64,
    pub cache_write_1h: u64,
    pub fast: bool,
    pub us_geo: bool,
}

impl ModelPrice {
    pub fn cost_usd(&self, u: &Usage) -> f64 {
        // Cache multipliers apply on top of fast-mode pricing, so scale every input category.
        let (input_scale, output_rate) = match (u.fast, self.fast) {
            (true, Some((fast_in, fast_out))) => (fast_in / self.input, fast_out),
            _ => (1.0, self.output),
        };
        let input_side = u.input as f64 * self.input
            + u.cache_write_5m as f64 * self.cache_write_5m
            + u.cache_write_1h as f64 * self.cache_write_1h
            + u.cache_read as f64 * self.cache_read;
        let geo = if u.us_geo { US_GEO_MULT } else { 1.0 };
        (input_side * input_scale + u.output as f64 * output_rate) * geo / 1_000_000.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PriceSource {
    BuiltIn,
    Cached { fetched_at: DateTime<Utc> },
    Live { fetched_at: DateTime<Utc> },
}

#[derive(Clone, Debug)]
pub struct PriceTable {
    pub models: HashMap<String, ModelPrice>,
    pub source: PriceSource,
}

impl PriceTable {
    pub fn built_in() -> Self {
        let p = |name: &str, input, w5, w1, read, output, fast| ModelPrice {
            name: name.to_string(),
            input,
            cache_write_5m: w5,
            cache_write_1h: w1,
            cache_read: read,
            output,
            fast,
        };
        // Snapshot of the pricing page as of 2026-10.
        let rows = [
            p("Claude Fable 5.1", 10.0, 12.5, 20.0, 0.25, 50.0, None),
            p("Claude Mythos 5.1", 10.0, 12.5, 20.0, 0.25, 50.0, None),
            p("Claude Fable 5", 10.0, 12.5, 20.0, 1.0, 50.0, None),
            p("Claude Mythos 5", 10.0, 12.5, 20.0, 1.0, 50.0, None),
            p(
                "Claude Opus 5.5",
                4.0,
                5.0,
                8.0,
                0.20,
                20.0,
                Some((8.0, 40.0)),
            ),
            p(
                "Claude Opus 5",
                5.0,
                6.25,
                10.0,
                0.50,
                25.0,
                Some((10.0, 50.0)),
            ),
            p(
                "Claude Opus 4.8",
                5.0,
                6.25,
                10.0,
                0.50,
                25.0,
                Some((10.0, 50.0)),
            ),
            p("Claude Opus 4.7", 5.0, 6.25, 10.0, 0.50, 25.0, None),
            p("Claude Opus 4.6", 5.0, 6.25, 10.0, 0.50, 25.0, None),
            p("Claude Opus 4.5", 5.0, 6.25, 10.0, 0.50, 25.0, None),
            p("Claude Opus 4.1", 15.0, 18.75, 30.0, 1.50, 75.0, None),
            p("Claude Opus 4", 15.0, 18.75, 30.0, 1.50, 75.0, None),
            p("Claude Sonnet 5.5", 2.0, 2.5, 4.0, 0.20, 10.0, None),
            p("Claude Sonnet 5", 2.0, 2.5, 4.0, 0.20, 10.0, None),
            p("Claude Sonnet 4.6", 3.0, 3.75, 6.0, 0.30, 15.0, None),
            p("Claude Sonnet 4.5", 3.0, 3.75, 6.0, 0.30, 15.0, None),
            p("Claude Sonnet 4", 3.0, 3.75, 6.0, 0.30, 15.0, None),
            p("Claude Haiku 4.5", 1.0, 1.25, 2.0, 0.10, 5.0, None),
            p("Claude Haiku 3.5", 0.80, 1.0, 1.60, 0.08, 4.0, None),
        ];
        Self {
            models: rows
                .into_iter()
                .map(|m| (model_id_from_name(&m.name), m))
                .collect(),
            source: PriceSource::BuiltIn,
        }
    }

    /// The cached table if one exists, otherwise the built-in one.
    pub fn load() -> Self {
        Self::load_cache().unwrap_or_else(|_| Self::built_in())
    }

    pub fn lookup(&self, model: &str) -> Option<&ModelPrice> {
        self.models.get(&normalize_model_id(model))
    }

    /// Fetches and parses the live pricing page. Blocking; call off the UI thread.
    pub fn fetch() -> Result<Self> {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(FETCH_TIMEOUT))
            .build()
            .into();
        let body = agent
            .get(PRICING_URL)
            .call()
            .context("requesting pricing page")?
            .body_mut()
            .read_to_string()
            .context("reading pricing page")?;
        let models = parse_pricing_page(&body)?;
        Ok(Self {
            models,
            source: PriceSource::Live {
                fetched_at: Utc::now(),
            },
        })
    }

    pub fn save_cache(&self) -> Result<()> {
        let fetched_at = match self.source {
            PriceSource::Live { fetched_at } | PriceSource::Cached { fetched_at } => fetched_at,
            PriceSource::BuiltIn => return Ok(()),
        };
        let path = cache_path().context("no cache directory")?;
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let file = CacheFile {
            fetched_at: fetched_at.timestamp(),
            models: self.models.clone(),
        };
        fs::write(&path, serde_json::to_vec_pretty(&file)?)?;
        Ok(())
    }

    fn load_cache() -> Result<Self> {
        let path = cache_path().context("no cache directory")?;
        let file: CacheFile = serde_json::from_slice(&fs::read(path)?)?;
        let fetched_at = DateTime::from_timestamp(file.fetched_at, 0).context("bad timestamp")?;
        Ok(Self {
            models: file.models,
            source: PriceSource::Cached { fetched_at },
        })
    }
}

#[derive(Serialize, Deserialize)]
struct CacheFile {
    fetched_at: i64,
    models: HashMap<String, ModelPrice>,
}

fn cache_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("claude-session-monitor").join("pricing.json"))
}

/// Maps a logged model id to the pricing table's key, tolerating provider prefixes
/// (`us.anthropic.`), date-pinned snapshots (`-20251001`) and suffixes like `[1m]`.
pub fn normalize_model_id(model: &str) -> String {
    let model = model.rsplit("anthropic.").next().unwrap_or(model);
    let model = model.split('[').next().unwrap_or(model).trim();
    let model = match model.rsplit_once('-') {
        Some((head, tail)) if tail.len() == 8 && tail.bytes().all(|b| b.is_ascii_digit()) => head,
        _ => model,
    };
    model.to_ascii_lowercase()
}

/// "Claude Opus 5.5 ([note](...))" -> "claude-opus-5-5".
fn model_id_from_name(name: &str) -> String {
    let name = name.split(" (").next().unwrap_or(name);
    let name = name.split('<').next().unwrap_or(name);
    name.trim().to_ascii_lowercase().replace([' ', '.'], "-")
}

fn display_name(cell: &str) -> String {
    let name = cell.split(" (").next().unwrap_or(cell);
    name.split('<').next().unwrap_or(name).trim().to_string()
}

/// "$12.50 / MTok<sup>1</sup>" -> 12.5
fn parse_price(cell: &str) -> Option<f64> {
    let after = &cell[cell.find('$')? + 1..];
    let end = after
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(after.len());
    after[..end].parse().ok()
}

fn cells(line: &str) -> Vec<&str> {
    let line = line.trim().trim_start_matches('|').trim_end_matches('|');
    line.split('|').map(str::trim).collect()
}

/// Rows of the first Markdown table at or after `start` whose header contains every
/// `required` column; returns (header cells, data rows).
fn find_table<'a>(
    lines: &[&'a str],
    start: usize,
    required: &[&str],
) -> Option<(Vec<&'a str>, Vec<Vec<&'a str>>)> {
    let header_ix = (start..lines.len()).find(|&i| {
        let line = lines[i];
        line.trim_start().starts_with('|') && required.iter().all(|r| line.contains(r))
    })?;
    let header = cells(lines[header_ix]);
    let rows = lines[header_ix + 1..]
        .iter()
        .take_while(|l| l.trim_start().starts_with('|'))
        .map(|l| cells(l))
        .filter(|row| {
            !row.iter()
                .all(|c| c.chars().all(|ch| matches!(ch, '-' | ':' | ' ')))
        })
        .collect();
    Some((header, rows))
}

pub fn parse_pricing_page(text: &str) -> Result<HashMap<String, ModelPrice>> {
    let lines: Vec<&str> = text.lines().collect();
    let (header, rows) = find_table(&lines, 0, &["Model", "Base input", "1h cache", "Output"])
        .context("model pricing table not found")?;
    let col = |needle: &str| {
        header
            .iter()
            .position(|h| h.to_ascii_lowercase().contains(needle))
            .with_context(|| format!("pricing table has no '{needle}' column"))
    };
    let (c_model, c_in, c_w5, c_w1, c_read, c_out) = (
        col("model")?,
        col("base input")?,
        col("5m cache")?,
        col("1h cache")?,
        col("cache hits")?,
        col("output")?,
    );

    let mut models = HashMap::new();
    for row in rows {
        let get = |i: usize| row.get(i).copied().and_then(parse_price);
        let (Some(input), Some(w5), Some(w1), Some(read), Some(output)) =
            (get(c_in), get(c_w5), get(c_w1), get(c_read), get(c_out))
        else {
            continue;
        };
        let Some(name_cell) = row.get(c_model) else {
            continue;
        };
        models.insert(
            model_id_from_name(name_cell),
            ModelPrice {
                name: display_name(name_cell),
                input,
                cache_write_5m: w5,
                cache_write_1h: w1,
                cache_read: read,
                output,
                fast: None,
            },
        );
    }
    if models.len() < 3 {
        bail!("parsed only {} models from pricing page", models.len());
    }

    // Fast mode has its own table under its own heading; models absent from it bill at standard
    // rates even when a request asks for fast mode.
    if let Some(start) = lines
        .iter()
        .position(|l| l.trim_start().starts_with("### Fast mode"))
    {
        if let Some((_, rows)) = find_table(&lines, start, &["Model", "Input", "Output"]) {
            for row in rows {
                let (Some(name_cell), Some(fin), Some(fout)) = (
                    row.first(),
                    row.get(1).copied().and_then(parse_price),
                    row.get(2).copied().and_then(parse_price),
                ) else {
                    continue;
                };
                for name in name_cell.split(" / ") {
                    if let Some(m) = models.get_mut(&model_id_from_name(name)) {
                        m.fast = Some((fin, fout));
                    }
                }
            }
        }
    }
    Ok(models)
}

/// Context window in tokens, for the models whose window is known.
pub fn context_window(model: &str) -> Option<u64> {
    let id = normalize_model_id(model);
    const SMALL: &[&str] = &[
        "claude-haiku-4-5",
        "claude-haiku-3-5",
        "claude-opus-4-5",
        "claude-opus-4-1",
        "claude-opus-4",
        "claude-sonnet-4-5",
        "claude-sonnet-4",
    ];
    const LARGE_PREFIXES: &[&str] = &[
        "claude-fable-",
        "claude-mythos-",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
    ];
    if SMALL.contains(&id.as_str()) {
        Some(200_000)
    } else if LARGE_PREFIXES.iter().any(|p| id.starts_with(p)) {
        Some(1_000_000)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
## Model pricing

| Model | Base input tokens | 5m cache writes | 1h cache writes | Cache hits and refreshes | Output tokens |
| :---- | :---------------- | :-------------- | :-------------- | :----------------------- | :------------ |
| Claude Fable 5.1 | $10 / MTok | $12.50 / MTok | $20 / MTok | $0.25 / MTok<sup>1</sup> | $50 / MTok |
| Claude Mythos 5.1 ([limited availability](https://anthropic.com/glasswing)) | $10 / MTok | $12.50 / MTok | $20 / MTok | $0.25 / MTok | $50 / MTok |
| Claude Opus 5.5 | $4 / MTok | $5 / MTok | $8 / MTok | $0.20 / MTok<sup>2</sup> | $20 / MTok |
| Claude Opus 5 | $5 / MTok | $6.25 / MTok | $10 / MTok | $0.50 / MTok | $25 / MTok |
| Claude Opus 4.8 | $5 / MTok | $6.25 / MTok | $10 / MTok | $0.50 / MTok | $25 / MTok |
| Claude Sonnet 5 | $2 / MTok<sup>3</sup> | $2.50 / MTok | $4 / MTok | $0.20 / MTok | $10 / MTok<sup>3</sup> |

### Fast mode pricing

| Model                           | Input      | Output     |
| ------------------------------- | ---------- | ---------- |
| Claude Opus 5.5                 | $8 / MTok  | $40 / MTok |
| Claude Opus 5 / Claude Opus 4.8 | $10 / MTok | $50 / MTok |
";

    #[test]
    fn parses_model_and_fast_tables() {
        let models = parse_pricing_page(SAMPLE).unwrap();
        assert_eq!(models.len(), 6);
        let opus = &models["claude-opus-5-5"];
        assert_eq!(opus.name, "Claude Opus 5.5");
        assert_eq!(
            (opus.input, opus.cache_write_1h, opus.cache_read),
            (4.0, 8.0, 0.20)
        );
        assert_eq!(opus.fast, Some((8.0, 40.0)));
        assert_eq!(models["claude-opus-4-8"].fast, Some((10.0, 50.0)));
        assert_eq!(models["claude-mythos-5-1"].name, "Claude Mythos 5.1");
        assert_eq!(models["claude-sonnet-5"].input, 2.0);
        assert_eq!(models["claude-fable-5-1"].fast, None);
    }

    #[test]
    fn rejects_page_without_table() {
        assert!(parse_pricing_page("# Pricing\n\nnothing here").is_err());
    }

    #[test]
    fn built_in_matches_parsed_layout() {
        let table = PriceTable::built_in();
        let parsed = parse_pricing_page(SAMPLE).unwrap();
        for (id, price) in parsed {
            assert_eq!(table.models.get(&id), Some(&price), "{id}");
        }
    }

    #[test]
    fn lookup_handles_prefixes_and_snapshots() {
        let table = PriceTable::built_in();
        assert_eq!(
            table.lookup("us.anthropic.claude-opus-5-5").unwrap().input,
            4.0
        );
        assert_eq!(
            table.lookup("claude-haiku-4-5-20251001").unwrap().input,
            1.0
        );
        assert_eq!(table.lookup("claude-opus-5").unwrap().input, 5.0);
        assert_eq!(table.lookup("claude-sonnet-4-6[1m]").unwrap().input, 3.0);
        assert!(table.lookup("<synthetic>").is_none());
    }

    #[test]
    fn cost_applies_cache_ttls_fast_mode_and_geo() {
        let table = PriceTable::built_in();
        let opus = table.lookup("claude-opus-5-5").unwrap();
        let usage = Usage {
            input: 1_000_000,
            output: 1_000_000,
            cache_read: 1_000_000,
            cache_write_5m: 1_000_000,
            cache_write_1h: 1_000_000,
            fast: false,
            us_geo: false,
        };
        // 4 input + 20 output + 0.20 read + 5 (5m write) + 8 (1h write)
        assert!((opus.cost_usd(&usage) - 37.20).abs() < 1e-9);
        let fast = Usage {
            fast: true,
            ..usage
        };
        assert!((opus.cost_usd(&fast) - 74.40).abs() < 1e-9);
        let geo = Usage {
            us_geo: true,
            ..usage
        };
        assert!((opus.cost_usd(&geo) - 40.92).abs() < 1e-9);
        // Opus 4.6 has no fast-mode price: fast requests bill at standard rates.
        let opus46 = table.lookup("claude-opus-4-6").unwrap();
        assert_eq!(opus46.cost_usd(&fast), opus46.cost_usd(&usage));
    }

    #[test]
    fn context_windows() {
        assert_eq!(context_window("claude-opus-5-5"), Some(1_000_000));
        assert_eq!(context_window("claude-haiku-4-5-20251001"), Some(200_000));
        assert_eq!(context_window("claude-opus-4"), Some(200_000));
        assert_eq!(context_window("mystery"), None);
    }
}

#[cfg(test)]
mod live_check {
    #[test]
    #[ignore]
    fn live_page_matches_built_in() {
        let live = super::PriceTable::fetch().unwrap();
        let built_in = super::PriceTable::built_in();
        for (id, price) in &live.models {
            assert_eq!(built_in.models.get(id), Some(price), "{id}");
        }
        assert_eq!(live.models.len(), built_in.models.len());
    }
}
