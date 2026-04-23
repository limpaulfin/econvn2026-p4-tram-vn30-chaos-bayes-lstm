# R verification: Lyapunov via nonlinearTseries, cross-check vs Python output.
# Runs same rolling window to compare per-ticker correlation.

largest_lyapunov_R <- function(x, m = 5, tau = 1) {
  if (length(x) < 50 || sd(x) == 0) return(NA_real_)
  tryCatch({
    res <- nonlinearTseries::maxLyapunov(
      time.series = x,
      min.embedding.dim = m, max.embedding.dim = m,
      time.lag = tau, radius = 0.1,
      theiler.window = 10, min.neighs = 5,
      min.ref.points = 500, max.time.steps = 20
    )
    mean(res$s.function[, 1], na.rm = TRUE)
  }, error = function(e) NA_real_)
}

rolling_lyap_R <- function(x, window = 250, step = 20) {
  idx <- seq(window, length(x), by = step)
  vapply(idx, function(i) largest_lyapunov_R(x[(i - window + 1):i]), numeric(1))
}

analysis_run_chaos <- function(params) {
  panel <- load_panel(params)
  out <- list()
  for (tkr in unique(panel$ticker)) {
    sub <- panel[panel$ticker == tkr, ]
    if (nrow(sub) < 260) next
    rets <- sub$ret_log
    rets <- rets[!is.na(rets)]
    lyap_series <- rolling_lyap_R(rets, window = 250, step = 20)
    out[[tkr]] <- data.frame(
      ticker = tkr,
      step_idx = seq_along(lyap_series),
      lyap_R = lyap_series
    )
    log_info(sprintf("R Lyapunov %s: %d points", tkr, length(lyap_series)))
  }
  all_df <- do.call(rbind, out)
  p <- paths(params)
  write.csv(all_df, file.path(p$output, "chaos_lyap_R.csv"), row.names = FALSE)
  log_info(sprintf("Saved %d rows to chaos_lyap_R.csv", nrow(all_df)))
}

crosscheck_vs_python <- function(params) {
  p <- paths(params)
  py_path <- file.path(p$root, "python", "output", "chaos_indicators_1D.csv")
  r_path <- file.path(p$output, "chaos_lyap_R.csv")
  if (!file.exists(py_path) || !file.exists(r_path)) {
    log_info("Missing one side. Run both pipelines first.")
    return(invisible(NULL))
  }
  py <- read.csv(py_path, stringsAsFactors = FALSE)
  rr <- read.csv(r_path, stringsAsFactors = FALSE)
  log_info(sprintf("Python rows: %d | R rows: %d", nrow(py), nrow(rr)))
  # Compare mean Lyapunov per ticker as sanity check
  py_mean <- tapply(py$lyap, py$ticker, mean, na.rm = TRUE)
  r_mean <- tapply(rr$lyap_R, rr$ticker, mean, na.rm = TRUE)
  common <- intersect(names(py_mean), names(r_mean))
  merged <- data.frame(ticker = common, py = py_mean[common], R = r_mean[common])
  log_info("Per-ticker mean Lyapunov (Python vs R):")
  print(merged)
  write.csv(merged, file.path(p$output, "crosscheck_lyap_py_vs_R.csv"), row.names = FALSE)
}
