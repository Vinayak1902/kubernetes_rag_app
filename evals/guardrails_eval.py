"""
Guardrails binary evaluation.
Sends each test input to the live /query API and checks if the guardrail fired.
Classifies each result as TP / TN / FP / FN and computes precision + recall.
"""

import time          # Used to introduce delays between API requests.
import copy          # Used to create deep copies of Python objects.
import requests      # Used to send HTTP requests to the FastAPI application.
import logfire       # Used for tracing and logging execution details.

# URL of the running FastAPI application.
API_URL = "http://localhost:8000/query"


# Determines whether the guardrail was triggered.
def _is_blocked(response_json: dict) -> bool:

    # Extract the thought process from the API response.
    tp = response_json.get("thought_process") or []

    # Return True if any step mentions that the guardrail fired.
    return any("guardrails fired" in step.lower() for step in tp)


# Executes all guardrail test cases.
def run_guardrails_eval(guardrails_samples: list, progress_callback=None) -> list:
    """
    Runs each guardrails test case against the live API.
    Adds actual_blocked and result (TP/TN/FP/FN) to each sample in place.
    Returns the enriched list.
    """

    # Create a deep copy so the original dataset remains unchanged.
    samples = copy.deepcopy(guardrails_samples)

    # Total number of guardrail test cases.
    n = len(samples)

    # Create a Logfire tracing span.
    with logfire.span("🛡️ Eval — Guardrails Tests", total=n):

        # Iterate over every guardrail sample.
        for i, sample in enumerate(samples):

            # Update progress bar if a callback is provided.
            if progress_callback:
                progress_callback(i, n, sample["input"])

            # Create a tracing span for this individual test.
            with logfire.span(

                # Span name.
                f"🛡️ Test {sample['id']}",

                # Log first 80 characters of the input.
                input_text=sample["input"][:80],

                # Log expected behaviour.
                expected_blocked=sample["expected_blocked"],
            ):

                try:

                    # Send the test input to the live FastAPI endpoint.
                    resp = requests.post(

                        # API URL.
                        API_URL,

                        # JSON request body.
                        json={
                            "q": sample["input"],                    # User query.
                            "thread_id": f"guardrail_eval_{i}"       # Unique conversation ID.
                        },

                        # Timeout after 30 seconds.
                        timeout=30,
                    )

                    # Raise an exception if HTTP status code indicates failure.
                    resp.raise_for_status()

                    # Determine whether the guardrail blocked the request.
                    blocked = _is_blocked(resp.json())

                # FastAPI server is unreachable.
                except requests.exceptions.ConnectionError:

                    # Log connection error.
                    logfire.error(
                        "❌ Cannot reach FastAPI — is the app running on :8000?"
                    )

                    # Treat as not blocked.
                    blocked = False

                # Handle any unexpected errors.
                except Exception as e:

                    # Log exception details.
                    logfire.error(f"❌ Guardrails test error: {e}")

                    # Treat as not blocked.
                    blocked = False

                # Expected behaviour for this test.
                expected = sample["expected_blocked"]

                # Store actual behaviour.
                sample["actual_blocked"] = blocked

                # ---------------- Classification ----------------

                # Expected block and actually blocked.
                if expected and blocked:

                    # True Positive.
                    sample["result"] = "TP"

                # Expected block but passed.
                elif expected and not blocked:

                    # False Negative.
                    sample["result"] = "FN"

                # Expected pass and actually passed.
                elif not expected and not blocked:

                    # True Negative.
                    sample["result"] = "TN"

                # Expected pass but blocked.
                else:

                    # False Positive.
                    sample["result"] = "FP"

                # Log evaluation result.
                logfire.info(

                    # Log message.
                    f"🛡️ {sample['result']}",

                    # Expected behaviour.
                    expected_blocked=expected,

                    # Actual behaviour.
                    actual_blocked=blocked,

                    # Store shortened input for easier debugging.
                    input_preview=sample["input"][:60],
                )

            # Wait 2 seconds before testing the next sample.
            time.sleep(2)

    # Return enriched guardrail results.
    return samples


# Computes overall evaluation metrics.
def compute_guardrails_metrics(results: list) -> dict:

    # Count True Positives.
    tp = sum(1 for r in results if r["result"] == "TP")

    # Count True Negatives.
    tn = sum(1 for r in results if r["result"] == "TN")

    # Count False Positives.
    fp = sum(1 for r in results if r["result"] == "FP")

    # Count False Negatives.
    fn = sum(1 for r in results if r["result"] == "FN")

    # Calculate Precision.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Calculate Recall.
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Calculate Accuracy.
    accuracy = (tp + tn) / len(results) if results else 0.0

    # Return all computed metrics.
    return {

        # Number of True Positives.
        "tp": tp,

        # Number of True Negatives.
        "tn": tn,

        # Number of False Positives.
        "fp": fp,

        # Number of False Negatives.
        "fn": fn,

        # Rounded Precision.
        "precision": round(precision, 3),

        # Rounded Recall.
        "recall": round(recall, 3),

        # Rounded Accuracy.
        "accuracy": round(accuracy, 3),

        # Total number of test cases.
        "total": len(results),

        # Number of correctly classified samples.
        "correct": tp + tn,
    }