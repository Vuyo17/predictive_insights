import fs from "node:fs";
import path from "node:path";

const rootDir = process.cwd();
const outputsDir = path.resolve(rootDir, "..", "outputs");
const publicSubmissionsDir = path.resolve(rootDir, "public", "submissions");
const dataOutFile = path.resolve(rootDir, "src", "data", "submissions.json");
const manualScoresFile = path.resolve(rootDir, "src", "data", "leaderboard_scores.json");
const cvMetricsFile = path.resolve(path.join(outputsDir, "cv_metrics.json"));
const watchMode = process.argv.includes("--watch");

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function parseCsvMetrics(filePath) {
  const raw = fs.readFileSync(filePath, "utf-8").trim();
  if (!raw) {
    return {
      rowCount: 0,
      duplicateIds: 0,
      missingPredictions: 0,
      invalidRangeCount: 0,
      minPrediction: null,
      maxPrediction: null,
      meanPrediction: null,
      stdPrediction: null,
      schemaValid: false,
    };
  }

  const lines = raw.split(/\r?\n/);
  const headers = lines[0].split(",").map((h) => h.trim());
  const idIdx = headers.indexOf("anonymised_id");
  const predIdx = headers.indexOf("employed_status");
  const schemaValid = idIdx >= 0 && predIdx >= 0;

  const ids = new Set();
  let duplicateIds = 0;
  let missingPredictions = 0;
  let invalidRangeCount = 0;

  const values = [];

  for (let i = 1; i < lines.length; i += 1) {
    if (!lines[i].trim()) {
      continue;
    }
    const cells = lines[i].split(",");
    const id = schemaValid ? (cells[idIdx] ?? "").trim() : "";
    const predRaw = schemaValid ? (cells[predIdx] ?? "").trim() : "";

    if (id) {
      if (ids.has(id)) {
        duplicateIds += 1;
      }
      ids.add(id);
    }

    if (predRaw === "") {
      missingPredictions += 1;
      continue;
    }

    const pred = Number(predRaw);
    if (Number.isNaN(pred)) {
      missingPredictions += 1;
      continue;
    }

    values.push(pred);

    if (pred < 0 || pred > 1) {
      invalidRangeCount += 1;
    }
  }

  const rowCount = lines.length - 1;

  let minPrediction = null;
  let maxPrediction = null;
  let meanPrediction = null;
  let stdPrediction = null;

  if (values.length > 0) {
    minPrediction = Math.min(...values);
    maxPrediction = Math.max(...values);
    meanPrediction = values.reduce((acc, v) => acc + v, 0) / values.length;
    const variance = values.reduce((acc, v) => acc + (v - meanPrediction) ** 2, 0) / values.length;
    stdPrediction = Math.sqrt(variance);
  }

  return {
    rowCount,
    duplicateIds,
    missingPredictions,
    invalidRangeCount,
    minPrediction,
    maxPrediction,
    meanPrediction,
    stdPrediction,
    schemaValid,
  };
}

function computeProxyScore(metrics) {
  if (!metrics.schemaValid || metrics.rowCount === 0) {
    return 0;
  }

  const dupPenalty = Math.min(1, metrics.duplicateIds / Math.max(1, metrics.rowCount));
  const missPenalty = Math.min(1, metrics.missingPredictions / Math.max(1, metrics.rowCount));
  const rangePenalty = Math.min(1, metrics.invalidRangeCount / Math.max(1, metrics.rowCount));

  const spread = metrics.stdPrediction ?? 0;
  const spreadComponent = Math.max(0, 1 - Math.abs(spread - 0.17) / 0.17);

  const quality = 1 - (0.5 * dupPenalty + 0.35 * missPenalty + 0.15 * rangePenalty);
  const score = 0.7 * quality + 0.3 * spreadComponent;
  return Number(Math.max(0, Math.min(1, score)).toFixed(6));
}

function readJsonSafe(filePath, fallbackValue) {
  try {
    if (!fs.existsSync(filePath)) {
      return fallbackValue;
    }
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return fallbackValue;
  }
}

function inferReportScore(fileName, cvMean) {
  if (fileName === "submission.csv") {
    return cvMean;
  }

  let reportPath = null;
  if (fileName.startsWith("submission_cont_")) {
    reportPath = path.join(outputsDir, fileName.replace(/^submission_/, "continuous_report_").replace(/\.csv$/i, ".json"));
  } else if (/^submission_.+\.csv$/i.test(fileName) && fileName !== "submission_.csv") {
    reportPath = path.join(outputsDir, fileName.replace(/^submission_/, "auto_report_").replace(/\.csv$/i, ".json"));
  }

  if (!reportPath || !fs.existsSync(reportPath)) {
    return null;
  }

  const report = readJsonSafe(reportPath, null);
  if (!report || typeof report !== "object") {
    return null;
  }

  const winnerAuc = report?.winner?.cv_auc_mean;
  const reportAuc = report?.cv_auc_mean;

  if (typeof winnerAuc === "number") {
    return winnerAuc;
  }
  if (typeof reportAuc === "number") {
    return reportAuc;
  }
  return null;
}

function main() {
  ensureDir(publicSubmissionsDir);

  const manualScoresData = readJsonSafe(manualScoresFile, { scores: {} });
  const manualScores = manualScoresData.scores || {};
  const cvMetrics = readJsonSafe(cvMetricsFile, {});
  const cvMean = typeof cvMetrics.cv_auc_mean === "number" ? cvMetrics.cv_auc_mean : null;

  const files = fs
    .readdirSync(outputsDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /^submission.*\.csv$/i.test(entry.name))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b));

  const submissions = files.map((fileName) => {
    const fullPath = path.join(outputsDir, fileName);
    const stats = fs.statSync(fullPath);
    const metrics = parseCsvMetrics(fullPath);

    const copyTarget = path.join(publicSubmissionsDir, fileName);
    fs.copyFileSync(fullPath, copyTarget);

    const manualScore = typeof manualScores[fileName] === "number" ? manualScores[fileName] : null;
    const cvScore = inferReportScore(fileName, cvMean);
    const proxyScore = computeProxyScore(metrics);
    const leaderboardIndicator = manualScore ?? cvScore ?? proxyScore;
    const effectiveScore = leaderboardIndicator;

    return {
      fileName,
      relativeOutputPath: path.posix.join("outputs", fileName),
      downloadPath: path.posix.join("submissions", fileName),
      modifiedAt: new Date(stats.mtimeMs).toISOString(),
      fileSizeKb: Number((stats.size / 1024).toFixed(2)),
      metrics,
      scores: {
        leaderboardAuc: manualScore,
        cvAucMean: cvScore,
        proxyScore,
        leaderboardIndicator,
        effectiveScore,
      },
    };
  });

  const chronological = [...submissions].sort(
    (a, b) => new Date(a.modifiedAt).getTime() - new Date(b.modifiedAt).getTime()
  );

  for (let i = 0; i < chronological.length; i += 1) {
    const current = chronological[i];
    const prev = chronological[i - 1];
    const improved = i === 0 ? true : current.scores.effectiveScore > prev.scores.effectiveScore;
    current.improvement = {
      previousFile: prev ? prev.fileName : null,
      improved,
      delta:
        i === 0
          ? null
          : Number((current.scores.effectiveScore - prev.scores.effectiveScore).toFixed(6)),
    };
  }

  const result = {
    generatedAt: new Date().toISOString(),
    submissions: chronological,
    improvementRule: {
      mode: "chronological",
      scoreField: "effectiveScore",
      note: "leaderboardIndicator uses leaderboardAuc if present, else report/cv AUC, else proxyScore",
    },
  };

  fs.writeFileSync(dataOutFile, JSON.stringify(result, null, 2), "utf-8");
  console.log(`Indexed ${submissions.length} submission files.`);
}

function shouldRefresh(fileName) {
  if (!fileName) {
    return true;
  }

  return (
    /^submission.*\.csv$/i.test(fileName) ||
    /^continuous_report_.*\.json$/i.test(fileName) ||
    /^auto_report_.*\.json$/i.test(fileName) ||
    fileName === "cv_metrics.json"
  );
}

function startWatch() {
  main();
  console.log("Watching for submission/score changes...");

  let timer = null;
  const debounceRefresh = () => {
    if (timer) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      try {
        main();
      } catch (error) {
        console.error("Refresh failed:", error.message);
      }
    }, 250);
  };

  fs.watch(outputsDir, (eventType, fileName) => {
    if (eventType && shouldRefresh(fileName)) {
      debounceRefresh();
    }
  });

  fs.watch(path.dirname(manualScoresFile), (eventType, fileName) => {
    if (eventType && fileName === path.basename(manualScoresFile)) {
      debounceRefresh();
    }
  });
}

if (watchMode) {
  startWatch();
} else {
  main();
}
