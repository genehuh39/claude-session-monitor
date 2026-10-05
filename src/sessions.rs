//! Finding session transcripts on disk: the session list for the picker, a session's subagent
//! transcripts, and whole-session summaries.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde_json::Value;

use crate::pricing::PriceTable;
use crate::{SessionTail, Stats};

/// While following the latest session, stay on the current one until it has been quiet this
/// long, so two sessions running side by side don't make the dashboard flip between them.
pub const FOLLOW_IDLE_SWITCH: Duration = Duration::from_secs(60);

/// A session written to within this window counts as live.
pub const LIVE_WINDOW: Duration = Duration::from_secs(60);

#[derive(Clone, Debug)]
pub struct SessionFile {
    pub path: PathBuf,
    pub modified: SystemTime,
}

impl SessionFile {
    /// The session id, shortened for display.
    pub fn short_id(&self) -> String {
        short_id(&self.path)
    }
}

pub fn short_id(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().chars().take(8).collect())
        .unwrap_or_default()
}

/// Main-session transcripts across all Claude Code projects, most recently modified first.
/// Subagent transcripts live one level deeper and are not included.
pub fn list_sessions() -> Vec<SessionFile> {
    let Ok(home) = std::env::var("HOME") else {
        return Vec::new();
    };
    let Ok(projects) = fs::read_dir(Path::new(&home).join(".claude/projects")) else {
        return Vec::new();
    };
    let mut sessions = Vec::new();
    for project in projects.flatten() {
        let Ok(entries) = fs::read_dir(project.path()) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let Ok(modified) = entry.metadata().and_then(|m| m.modified()) else {
                continue;
            };
            let modified = latest_subagent_write(&path).map_or(modified, |m| m.max(modified));
            sessions.push(SessionFile { path, modified });
        }
    }
    sessions.sort_by(|a, b| b.modified.cmp(&a.modified));
    sessions
}

/// When the session was last written to, counting its subagents: while a subagent runs, the
/// main transcript can sit untouched for minutes.
pub fn last_activity(path: &Path) -> Option<SystemTime> {
    let main = fs::metadata(path).and_then(|m| m.modified()).ok();
    match (main, latest_subagent_write(path)) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}

fn latest_subagent_write(session: &Path) -> Option<SystemTime> {
    subagent_files(session)
        .iter()
        .filter_map(|p| fs::metadata(p).and_then(|m| m.modified()).ok())
        .max()
}

fn subagent_files(session: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(subagent_dir(session)) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .collect()
}

pub fn is_live(modified: Option<SystemTime>) -> bool {
    modified
        .and_then(|m| m.elapsed().ok())
        .is_some_and(|age| age < LIVE_WINDOW)
}

/// Whether to leave the current session for `newest` while following the latest session.
pub fn should_follow(
    current: Option<&Path>,
    current_modified: Option<SystemTime>,
    newest: &Path,
    now: SystemTime,
) -> bool {
    match (current, current_modified) {
        (None, _) | (_, None) => true,
        (Some(current), _) if current == newest => false,
        (Some(_), Some(modified)) => now
            .duration_since(modified)
            .is_ok_and(|idle| idle >= FOLLOW_IDLE_SWITCH),
    }
}

/// The last path component of a session's working directory.
pub fn project_name(cwd: Option<&str>, path: &Path) -> String {
    if let Some(name) = cwd.and_then(|c| Path::new(c).file_name()) {
        return name.to_string_lossy().to_string();
    }
    // Fall back to the project directory, which encodes the cwd with '/' replaced by '-'.
    path.parent()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().trim_start_matches('-').to_string())
        .unwrap_or_default()
}

/// One subagent run spawned from the session, read from `<session>/subagents/agent-<id>.jsonl`.
pub struct AgentRun {
    pub id: String,
    pub agent_type: Option<String>,
    pub description: Option<String>,
    pub tail: SessionTail,
    pub stats: Stats,
}

impl AgentRun {
    pub fn label(&self) -> &str {
        self.agent_type.as_deref().unwrap_or("agent")
    }

    fn load_meta(&mut self) {
        let meta = self.tail.path.with_extension("meta.json");
        let Some(value) = fs::read_to_string(meta)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        else {
            return;
        };
        let field = |key: &str| value.get(key).and_then(Value::as_str).map(str::to_string);
        self.agent_type = field("agentType");
        self.description = field("description");
    }
}

fn subagent_dir(session: &Path) -> PathBuf {
    session.with_extension("").join("subagents")
}

/// Picks up subagent transcripts that appeared since the last call and reads new lines from all
/// of them.
pub fn sync_agents(session: &Path, agents: &mut Vec<AgentRun>) {
    let mut found: Vec<PathBuf> = subagent_files(session)
        .into_iter()
        .filter(|p| !agents.iter().any(|a| a.tail.path == *p))
        .collect();
    found.sort();
    for path in found {
        let id = path
            .file_stem()
            .map(|s| s.to_string_lossy().trim_start_matches("agent-").to_string())
            .unwrap_or_default();
        agents.push(AgentRun {
            id,
            agent_type: None,
            description: None,
            tail: SessionTail::new(path),
            stats: Stats::default(),
        });
    }
    for agent in agents.iter_mut() {
        // The metadata file can land after the transcript starts.
        if agent.agent_type.is_none() {
            agent.load_meta();
        }
        let _ = agent.tail.poll(&mut agent.stats);
    }
}

/// A whole session, including its subagents, reduced to what the session picker shows.
#[derive(Clone, Debug)]
pub struct SessionSummary {
    /// The transcripts' sizes when summarized; a different fingerprint means it's stale.
    pub fingerprint: Fingerprint,
    pub cwd: Option<String>,
    pub cost: f64,
    pub partial: bool,
}

/// Total length and count of a session's transcripts, its subagents' included. Transcripts are
/// append-only, so any new usage changes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fingerprint {
    bytes: u64,
    files: usize,
}

pub fn fingerprint(path: &Path) -> Fingerprint {
    let len = |p: &Path| fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    let agents = subagent_files(path);
    Fingerprint {
        bytes: len(path) + agents.iter().map(|p| len(p)).sum::<u64>(),
        files: 1 + agents.len(),
    }
}

pub fn summarize(path: &Path, prices: &PriceTable) -> SessionSummary {
    let fingerprint = fingerprint(path);
    let mut tail = SessionTail::new(path.to_path_buf());
    let mut stats = Stats::default();
    let _ = tail.poll(&mut stats);
    let mut agents = Vec::new();
    sync_agents(path, &mut agents);
    let all = std::iter::once(&stats).chain(agents.iter().map(|a| &a.stats));
    let (mut cost, mut partial) = (0.0, false);
    for s in all {
        cost += s.total_cost_usd(prices);
        partial |= s.unpriced_turns(prices) > 0;
    }
    SessionSummary {
        fingerprint,
        cwd: stats.cwd,
        cost,
        partial,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_only_after_the_current_session_goes_quiet() {
        let now = SystemTime::now();
        let a = Path::new("/p/a.jsonl");
        let b = Path::new("/p/b.jsonl");
        assert!(should_follow(None, None, b, now));
        assert!(!should_follow(Some(b), Some(now), b, now));
        assert!(!should_follow(
            Some(a),
            Some(now - Duration::from_secs(5)),
            b,
            now
        ));
        assert!(should_follow(
            Some(a),
            Some(now - FOLLOW_IDLE_SWITCH),
            b,
            now
        ));
        // A current session that vanished (no mtime) is abandoned.
        assert!(should_follow(Some(a), None, b, now));
    }

    #[test]
    fn subagents_count_toward_activity_and_fingerprint() {
        let dir = std::env::temp_dir().join(format!("csm-test-{}", std::process::id()));
        let session = dir.join("abc.jsonl");
        let agents = dir.join("abc/subagents");
        fs::create_dir_all(&agents).unwrap();
        fs::write(&session, "main\n").unwrap();
        let old = SystemTime::now() - Duration::from_secs(600);
        fs::File::options()
            .write(true)
            .open(&session)
            .unwrap()
            .set_modified(old)
            .unwrap();
        let before = fingerprint(&session);

        // A subagent writing while the main transcript sits idle keeps the session active.
        fs::write(agents.join("agent-x.jsonl"), "agent\n").unwrap();
        assert!(last_activity(&session).unwrap() > old + Duration::from_secs(300));
        assert!(is_live(last_activity(&session)));
        assert_ne!(fingerprint(&session), before);
        assert_eq!(fingerprint(&session).files, 2);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn project_name_prefers_cwd() {
        let path = Path::new("/h/.claude/projects/-home-me-my-app/abc.jsonl");
        assert_eq!(project_name(Some("/home/me/my-app"), path), "my-app");
        assert_eq!(project_name(None, path), "home-me-my-app");
        assert_eq!(short_id(Path::new("/x/d5d22496-6a4c.jsonl")), "d5d22496");
    }
}
