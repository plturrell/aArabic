"""
Mojo abstraction over Slurm job control commands.
Provides a clean API for listing, submitting, and canceling jobs via squeue/sbatch/scancel.
"""

from sys.process import Process
from collections import List
from io import write_file, read_file, file_exists


struct JobInfo:
    id: String
    name: String
    status: String
    total_time_seconds: Int
    reason: String


fn run_command(args: List[String]) -> String raises:
    var process = Process(args)
    let result = process.run()
    if result.exit_code != 0:
        raise Error("Command failed: " + " ".join(args) + " stderr=" + result.stderr)
    return result.stdout


fn parse_squeue_output(text: String) -> List[JobInfo]:
    var jobs = List[JobInfo]()
    let lines = text.strip().split("\n")
    if len(lines) <= 1:
        return jobs
    for line in lines[1:]:
        if len(line.strip()) == 0:
            continue
        var parts = List[String]()
        var current = String("")
        for ch in line:
            if ch.isspace():
                if len(current) > 0:
                    parts.append(current)
                    current = ""
            else:
                current += ch
        if len(current) > 0:
            parts.append(current)
        if len(parts) < 6:
            continue
        var job = JobInfo(
            id=parts[0],
            name=parts[2],
            status=parts[4],
            total_time_seconds=parse_runtime(parts[5]),
            reason=parts[-1]
        )
        jobs.append(job)
    return jobs


fn parse_runtime(runtime: String) -> Int:
    if "-" in runtime:
        return 3600
    let sections = runtime.split(":")
    if len(sections) == 2:
        return Int(sections[0]) * 60 + Int(sections[1])
    if len(sections) == 3:
        return Int(sections[0]) * 3600 + Int(sections[1]) * 60 + Int(sections[2])
    return 0


fn list_jobs(user: String) raises -> List[JobInfo]:
    var args = List[String]()
    args.append("squeue")
    args.append("-u")
    args.append(user)
    let output = run_command(args)
    return parse_squeue_output(output)


fn submit(script_path: String) raises -> String:
    var args = List[String]()
    args.append("sbatch")
    args.append(script_path)
    let output = run_command(args)
    return output.strip()


fn cancel(job_id: String) raises:
    var args = List[String]()
    args.append("scancel")
    args.append(job_id)
    _ = run_command(args)


fn write_script(path: String, content: String) raises:
    write_file(path, content)


fn read_script(path: String) -> String raises:
    if not file_exists(path):
        raise Error("script not found: " + path)
    return read_file(path)
