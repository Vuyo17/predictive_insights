import { useMemo, useState } from "react";
import submissionsData from "./data/submissions.json";
import serverLogsData from "./data/server_logs.json";

const SCORE_STORE_KEY = "submission_real_scores_v1";
const SUBMITTED_STORE_KEY = "submission_submitted_status_v1";

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function scoreLabel(scores) {
  if (scores.leaderboardAuc !== null) {
    return "Leaderboard AUC";
  }
  if (scores.cvAucMean !== null) {
    return "CV AUC";
  }
  return "Proxy Score";
}

function formatOrdinal(value) {
  const mod100 = value % 100;
  if (mod100 >= 11 && mod100 <= 13) {
    return `${value}th`;
  }

  const mod10 = value % 10;
  if (mod10 === 1) {
    return `${value}st`;
  }
  if (mod10 === 2) {
    return `${value}nd`;
  }
  if (mod10 === 3) {
    return `${value}rd`;
  }
  return `${value}th`;
}

function getIndicatorScore(scores) {
  if (!scores || typeof scores !== "object") {
    return null;
  }

  const candidates = [
    scores.leaderboardIndicator,
    scores.leaderboardAuc,
    scores.cvAucMean,
    scores.proxyScore,
  ];

  for (const value of candidates) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }

  return null;
}

function sortIndicatorValue(file) {
  const value = getIndicatorScore(file?.scores);
  return typeof value === "number" && Number.isFinite(value) ? value : -1;
}

function loadLocalScores() {
  try {
    const raw = localStorage.getItem(SCORE_STORE_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function saveLocalScores(payload) {
  localStorage.setItem(SCORE_STORE_KEY, JSON.stringify(payload));
}

function loadSubmittedState() {
  try {
    const raw = localStorage.getItem(SUBMITTED_STORE_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function saveSubmittedState(payload) {
  localStorage.setItem(SUBMITTED_STORE_KEY, JSON.stringify(payload));
}

function formatDateTime(value) {
  if (!value) {
    return "-";
  }

  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) {
    return "-";
  }
  return dt.toLocaleString();
}

function summarizeRoles(server) {
  const roles = [];
  if (server?.roles?.frontend) {
    roles.push("Frontend");
  }
  if (server?.roles?.backend) {
    roles.push("Backend");
  }
  return roles.join(" + ") || "-";
}

function App() {
  const submissions = submissionsData.submissions || [];
  const generatedAt = submissionsData.generatedAt ? new Date(submissionsData.generatedAt) : null;
  const allImproving = submissions.every((s) => s.improvement?.improved !== false);
  const serverLogs = Array.isArray(serverLogsData?.servers) ? serverLogsData.servers : [];
  const serverLogsGeneratedAt = formatDateTime(serverLogsData?.generatedAt);

  const [localRealScores, setLocalRealScores] = useState(() => loadLocalScores());
  const [submittedState, setSubmittedState] = useState(() => loadSubmittedState());
  const [draftScores, setDraftScores] = useState({});

  const enriched = useMemo(
    () => {
      const ranked = submissions.map((file) => {
        const localValueRaw = localRealScores[file.fileName];
        const localValue =
          typeof localValueRaw === "number" && Number.isFinite(localValueRaw) ? localValueRaw : null;
        const liveRealScore = localValue ?? file.scores.leaderboardAuc;
        return {
          ...file,
          liveRealScore,
          hasLocalOverride: localValue !== null,
          submitted: Boolean(submittedState[file.fileName]),
        };
      });

      return ranked
        .sort((a, b) => {
          const indicatorDiff = sortIndicatorValue(b) - sortIndicatorValue(a);
          if (indicatorDiff !== 0) {
            return indicatorDiff;
          }

          return b.scores.proxyScore - a.scores.proxyScore;
        })
        .map((file, index) => ({
          ...file,
          rank: index + 1,
          rankLabel: formatOrdinal(index + 1),
        }));
    },
    [submissions, localRealScores, submittedState]
  );

  const bestRealScore = enriched
    .map((s) => s.liveRealScore)
    .filter((v) => typeof v === "number")
    .sort((a, b) => b - a)[0];

  const bestIndicatorScore = enriched
    .map((s) => getIndicatorScore(s.scores))
    .filter((v) => typeof v === "number")
    .sort((a, b) => b - a)[0];

  function onChangeDraft(fileName, value) {
    setDraftScores((prev) => ({
      ...prev,
      [fileName]: value,
    }));
  }

  function onSaveRealScore(fileName) {
    const raw = (draftScores[fileName] ?? "").trim();
    const value = Number(raw);

    if (!raw || Number.isNaN(value) || value < 0 || value > 1) {
      return;
    }

    const next = {
      ...localRealScores,
      [fileName]: value,
    };
    setLocalRealScores(next);
    saveLocalScores(next);
    setDraftScores((prev) => ({
      ...prev,
      [fileName]: "",
    }));
  }

  function onClearRealScore(fileName) {
    const next = { ...localRealScores };
    delete next[fileName];
    setLocalRealScores(next);
    saveLocalScores(next);
  }

  function onToggleSubmitted(fileName) {
    const next = {
      ...submittedState,
      [fileName]: !submittedState[fileName],
    };
    setSubmittedState(next);
    saveSubmittedState(next);
  }

  return (
    <div className="page-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Predictive Insights Hackathon</p>
          <h1>Submission Metrics Dashboard</h1>
          <p className="hero-subtitle">
            Track every submission file, compare scores, and download the exact file you want to upload.
          </p>
        </div>
        <div className={`status-chip ${allImproving ? "good" : "warn"}`}>
          {allImproving ? "All submissions improving" : "Improvement rule broken"}
        </div>
      </header>

      <section className="panel">
        <div className="panel-header">
          <h2>Submission Tracker</h2>
          <p>
            Generated: {generatedAt ? generatedAt.toLocaleString() : "-"} | Files: {submissions.length} | Best
            Real Score: {bestRealScore !== undefined ? formatNumber(bestRealScore, 6) : "-"} | Best Indicator: {bestIndicatorScore !== undefined ? formatNumber(bestIndicatorScore, 6) : "-"}
          </p>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>File</th>
                <th>Indicator</th>
                <th>Source</th>
                <th>Proxy</th>
                <th>Recorded Real</th>
                <th>Live Real</th>
                <th>Submitted</th>
                <th>Improved</th>
                <th>Delta</th>
                <th>Rows</th>
                <th>Updated</th>
                <th>Download</th>
                <th>New Real Score</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {enriched.map((file) => (
                <tr key={`row-${file.fileName}`}>
                  <td>{file.rankLabel}</td>
                  <td className="file-cell">{file.fileName}</td>
                  <td>{formatNumber(getIndicatorScore(file.scores), 6)}</td>
                  <td>{scoreLabel(file.scores)}</td>
                  <td>{formatNumber(file.scores.proxyScore, 6)}</td>
                  <td>{formatNumber(file.scores.leaderboardAuc, 6)}</td>
                  <td>
                    {formatNumber(file.liveRealScore, 6)} {file.hasLocalOverride ? "(local)" : ""}
                  </td>
                  <td>
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={file.submitted}
                        onChange={() => onToggleSubmitted(file.fileName)}
                      />
                      <span>{file.submitted ? "Yes" : "No"}</span>
                    </label>
                  </td>
                  <td>
                    <span className={`improve-tag ${file.improvement?.improved ? "up" : "down"}`}>
                      {file.improvement?.improved ? "Yes" : "No"}
                    </span>
                  </td>
                  <td>
                    {file.improvement?.delta === null
                      ? "-"
                      : `${file.improvement.delta > 0 ? "+" : ""}${formatNumber(file.improvement.delta, 6)}`}
                  </td>
                  <td>{file.metrics.rowCount}</td>
                  <td>{new Date(file.modifiedAt).toLocaleString()}</td>
                  <td>
                    <a href={`/${file.downloadPath}`} download>
                      Download
                    </a>
                  </td>
                  <td>
                    <input
                      className="score-input"
                      type="number"
                      min="0"
                      max="1"
                      step="0.00001"
                      placeholder="0.62719"
                      value={draftScores[file.fileName] ?? ""}
                      onChange={(e) => onChangeDraft(file.fileName, e.target.value)}
                    />
                  </td>
                  <td className="action-cell">
                    <button type="button" onClick={() => onSaveRealScore(file.fileName)}>
                      Save
                    </button>
                    <button type="button" className="ghost" onClick={() => onClearRealScore(file.fileName)}>
                      Clear
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Server Runtime Logs</h2>
          <p>Snapshot: {serverLogsGeneratedAt} | Servers: {serverLogs.length}</p>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Server</th>
                <th>Roles</th>
                <th>Backend</th>
                <th>Best CV</th>
                <th>Iterations</th>
                <th>Backend Updated</th>
                <th>Frontend</th>
                <th>URL</th>
                <th>Backend Log Tail</th>
                <th>Frontend Log Tail</th>
              </tr>
            </thead>
            <tbody>
              {serverLogs.map((server) => {
                const backendState = server?.backend?.state || {};
                const backendRunning = Boolean(server?.backend?.running);
                const frontendRunning = Boolean(server?.frontend?.running);

                return (
                  <tr key={`server-${server.name}`}>
                    <td>
                      {server.name}
                      <div className="server-subtext">
                        {server.user}@{server.host}
                      </div>
                    </td>
                    <td>{summarizeRoles(server)}</td>
                    <td>
                      <span className={`improve-tag ${backendRunning ? "up" : "down"}`}>
                        {backendRunning ? "Running" : "Stopped"}
                      </span>
                    </td>
                    <td>{formatNumber(backendState.current_best_cv_auc, 6)}</td>
                    <td>{backendState.iterations_completed ?? "-"}</td>
                    <td>{formatDateTime(backendState.updated_at)}</td>
                    <td>
                      {server.roles?.frontend ? (
                        <span className={`improve-tag ${frontendRunning ? "up" : "down"}`}>
                          {frontendRunning ? "Running" : "Stopped"}
                        </span>
                      ) : (
                        "-"
                      )}
                    </td>
                    <td>
                      {server?.frontend?.url ? (
                        <a href={server.frontend.url} target="_blank" rel="noreferrer">
                          Open
                        </a>
                      ) : (
                        "-"
                      )}
                    </td>
                    <td>
                      <pre className="log-tail">{server?.backend?.logTail || "(no backend logs yet)"}</pre>
                    </td>
                    <td>
                      <pre className="log-tail">
                        {server.roles?.frontend
                          ? server?.frontend?.logTail || "(no frontend logs yet)"
                          : "-"}
                      </pre>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

export default App;
