# Shared R utilities.
# Provides: paths(), log_info(), load_parquet().

paths <- function(params) {
  root <- normalizePath(file.path(dirname(dirname(sys.frame(1)$ofile)), ".."))
  list(
    root      = root,
    raw       = file.path(root, "data", "raw"),
    processed = file.path(root, "data", "processed"),
    output    = file.path(root, "R", "output"),
    logs      = file.path(root, "R", "logs")
  )
}

log_info <- function(msg) {
  cat(sprintf("%s | INFO | %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))
}

load_parquet <- function(path) {
  arrow::read_parquet(path)
}
