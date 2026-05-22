import typer

from rageval.cli import (
    ci_check,
    compare,
    evaluate_retrieval,
    generate_evalset,
    ingest,
    inspect_cmd,
    report,
    retrieve,
    run_cmd,
)
from rageval.cli import init_cmd

app = typer.Typer(
    name="rageval",
    help="CLI-first RAG evaluation and regression-testing framework.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

app.command("init")(init_cmd.init)
app.command("ingest")(ingest.ingest)
app.command("generate-evalset")(generate_evalset.generate_evalset)
app.command("run")(run_cmd.run)
app.command("compare")(compare.compare)
app.command("report")(report.report)
app.command("ci-check")(ci_check.ci_check)
app.command("inspect")(inspect_cmd.inspect_run)
app.command("retrieve")(retrieve.retrieve)
app.command("evaluate-retrieval")(evaluate_retrieval.evaluate_retrieval)
