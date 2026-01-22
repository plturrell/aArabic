"""
Regression tests for the Mojo-based job_control module.
Tests runtime parsing and squeue output parsing functions.
"""

from collections import List
import sys
sys.path.append("src/serviceCore/nOpenaiServer")

from tools.toolorchestra.job_control.job_control import (
    JobInfo,
    parse_runtime,
    parse_squeue_output,
)


fn expect(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


fn test_parse_runtime_minutes_seconds() raises:
    print("🧪 test_parse_runtime_minutes_seconds")
    var result = parse_runtime("10:30")
    expect(result == 630, "expected 10:30 to be 630 seconds, got " + String(result))


fn test_parse_runtime_hours_minutes_seconds() raises:
    print("🧪 test_parse_runtime_hours_minutes_seconds")
    var result = parse_runtime("1:23:45")
    expect(result == 5025, "expected 1:23:45 to be 5025 seconds, got " + String(result))


fn test_parse_runtime_with_days() raises:
    print("🧪 test_parse_runtime_with_days")
    var result = parse_runtime("1-12:00:00")
    expect(result == 3600, "expected 1-12:00:00 to be 3600 seconds (special case), got " + String(result))


fn test_parse_squeue_output_empty() raises:
    print("🧪 test_parse_squeue_output_empty")
    # Test with header only
    var header_only = "JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)"
    var jobs = parse_squeue_output(header_only)
    expect(len(jobs) == 0, "expected 0 jobs from header-only output, got " + String(len(jobs)))

    # Test with empty string
    var empty_jobs = parse_squeue_output("")
    expect(len(empty_jobs) == 0, "expected 0 jobs from empty output, got " + String(len(empty_jobs)))


fn test_parse_squeue_output_single_job() raises:
    print("🧪 test_parse_squeue_output_single_job")
    var squeue_output = """JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)
12345 gpu my_job user R 10:30 1 node001"""
    var jobs = parse_squeue_output(squeue_output)
    expect(len(jobs) == 1, "expected 1 job, got " + String(len(jobs)))
    expect(jobs[0].id == "12345", "expected job id 12345, got " + jobs[0].id)
    expect(jobs[0].name == "my_job", "expected job name my_job, got " + jobs[0].name)
    expect(jobs[0].status == "R", "expected status R, got " + jobs[0].status)
    expect(jobs[0].total_time_seconds == 630, "expected 630 seconds, got " + String(jobs[0].total_time_seconds))
    expect(jobs[0].reason == "node001", "expected reason node001, got " + jobs[0].reason)


fn test_parse_squeue_output_multiple_jobs() raises:
    print("🧪 test_parse_squeue_output_multiple_jobs")
    var squeue_output = """JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)
12345 gpu train_model user R 1:23:45 1 node001
12346 cpu preprocess user PD 0:00 2 (Resources)
12347 gpu inference user R 10:30 1 node002"""
    var jobs = parse_squeue_output(squeue_output)
    expect(len(jobs) == 3, "expected 3 jobs, got " + String(len(jobs)))

    # Check first job
    expect(jobs[0].id == "12345", "job 0: expected id 12345")
    expect(jobs[0].name == "train_model", "job 0: expected name train_model")
    expect(jobs[0].status == "R", "job 0: expected status R")
    expect(jobs[0].total_time_seconds == 5025, "job 0: expected 5025 seconds")

    # Check second job (pending)
    expect(jobs[1].id == "12346", "job 1: expected id 12346")
    expect(jobs[1].name == "preprocess", "job 1: expected name preprocess")
    expect(jobs[1].status == "PD", "job 1: expected status PD")
    expect(jobs[1].reason == "(Resources)", "job 1: expected reason (Resources)")

    # Check third job
    expect(jobs[2].id == "12347", "job 2: expected id 12347")
    expect(jobs[2].name == "inference", "job 2: expected name inference")
    expect(jobs[2].total_time_seconds == 630, "job 2: expected 630 seconds")


fn main() raises:
    print("═══════════════════════════════════════════════════════════════════════")
    print("  toolorchestra job_control regression tests")
    print("═══════════════════════════════════════════════════════════════════════")

    test_parse_runtime_minutes_seconds()
    print("✅ parse_runtime minutes:seconds passed")

    test_parse_runtime_hours_minutes_seconds()
    print("✅ parse_runtime hours:minutes:seconds passed")

    test_parse_runtime_with_days()
    print("✅ parse_runtime with days passed")

    test_parse_squeue_output_empty()
    print("✅ parse_squeue_output empty/header-only passed")

    test_parse_squeue_output_single_job()
    print("✅ parse_squeue_output single job passed")

    test_parse_squeue_output_multiple_jobs()
    print("✅ parse_squeue_output multiple jobs passed")

    print("🎉 All job_control Mojo tests passed")

