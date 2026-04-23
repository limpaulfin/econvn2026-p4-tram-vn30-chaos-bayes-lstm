# Load panel parquet produced by Python data_prep.
# R-side SSoT = same files. No refetching.

# arrow package fails without libarrow-dev; use CSV as portable interchange.

data_prep_run <- function(params) {
  p <- paths(params)
  csv_path <- file.path(p$processed, "vn30_panel_1D.csv")
  if (!file.exists(csv_path)) {
    log_info("Panel CSV not found. Run python export_csv helper first.")
    return(invisible(NULL))
  }
  panel <- read.csv(csv_path, stringsAsFactors = FALSE)
  panel$time <- as.POSIXct(panel$time)
  log_info(sprintf("Loaded panel: %d rows x %d cols, %d tickers",
                   nrow(panel), ncol(panel),
                   length(unique(panel$ticker))))
  invisible(panel)
}

load_panel <- function(params) {
  p <- paths(params)
  panel <- read.csv(file.path(p$processed, "vn30_panel_1D.csv"), stringsAsFactors = FALSE)
  panel$time <- as.POSIXct(panel$time)
  panel
}
