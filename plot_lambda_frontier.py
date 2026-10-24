import json
import sys
from pathlib import Path

from src.visualize import plot_lambda_frontier


def main() -> int:
    # Regenerate the lambda frontier figure from saved multi-seed results
    source = sys.argv[1] if len(sys.argv) > 1 else "results/lambda_multiseed.json"
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    lambdas = sorted(float(key) for key in payload["lambdas"])
    per_lambda = {
        key: {"summary": payload["lambdas"][key]["summary"]} for key in payload["lambdas"]
    }
    print(f"lambdas {lambdas}")
    print(f"seeds {payload['seeds']}  paths {payload['paths']:,}  epochs {payload['epochs']}")
    figure = plot_lambda_frontier(per_lambda, lambdas, Path("results"))
    print(f"saved {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
