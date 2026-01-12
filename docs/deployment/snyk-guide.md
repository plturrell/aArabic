# Snyk Scan Procedure

The project mandate requires running Snyk Code scans whenever first-party code (Python, Rust, Node, etc.) is modified. Follow this checklist:

## 1. Install Snyk CLI (if needed)
```bash
brew tap snyk/tap
brew install snyk
snyk auth    # opens browser flow
```

## 2. Run scans from repo root
Use absolute paths per automation policy:
```bash
snyk code test /Users/user/Documents/arabic_folder
```
Add `--severity-threshold=medium` if you only need actionable results.

## 3. Address findings
1. Review the reported file/line and implement the suggested fix.
2. If the issue originates from vendored code that you do not control, document the rationale in the PR and consider suppressing via `.snyk` policy.

## 4. Re-scan to confirm
After fixes, run the same command to ensure the issue count returns to zero.

## 5. Report prevention/fixes (optional)
When collaborating with Cascade, use the `mcp0_snyk_send_feedback` tool so automated records show counts of fixed/prevented issues for the current session.

Keep this process in CI by adding a pipeline stage that executes `snyk code test` against the repo root before merge.
