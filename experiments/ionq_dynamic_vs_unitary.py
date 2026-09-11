"""
IonQ Forte Enterprise: Dynamic vs Pure Unitary comparison.
Supplements Table 1 (IBM Heron) with trapped-ion data.
Uses Open Quantum SDK + MCP service layer.
"""

import os, sys, json, time
from datetime import datetime

sys.path.insert(0, r"C:/Users/26427/AppData/Local/Programs/Python/Python312")

from openquantum_sdk.auth import ClientCredentialsAuth, ClientCredentials
from openquantum_sdk.clients import SchedulerClient, ManagementClient
from openquantum_sdk.models import JobPreparationCreate
from openquantum_mcp.service import OpenQuantumService
from openquantum_mcp.auth import SdkClients
from openquantum_mcp.config import McpConfig

# ---- Config ----
CLIENT_ID = 's_a7085abf63c64066856a957e85497daa'
CLIENT_SECRET = '07e1cd35d9c67a65e69fb5cbf3fc3bf037ee8e3a0b5b24e44955fb1eb22af552'
BACKEND_ID = 'cc924471-7661-4420-a4f4-88a46b6b168a'  # IonQ Forte Enterprise
BACKEND_NAME = 'IonQ Forte Enterprise'
SHOTS = 4000
DEPTHS = [1, 3, 5, 7, 9, 13, 17]


def make_qasm_pure_unitary(n: int) -> str:
    """Pure unitary: X(q0) → CNOT(q0, qi) for i=1..n → measure all."""
    lines = [
        'OPENQASM 3.0;',
        f'qubit[{n + 1}] q;',
        'bit[{n + 1}] c;',
        '',
        'x q[0];',
    ]
    for i in range(1, n + 1):
        lines.append(f'cx q[0], q[{i}];')
    for i in range(n + 1):
        lines.append(f'c[{i}] = measure q[{i}];')
    return '\n'.join(lines)


def make_qasm_dynamic(n: int) -> str:
    """Dynamic: measure q0, conditional X on each target."""
    lines = [
        'OPENQASM 3.0;',
        f'qubit[{n + 1}] q;',
        'bit[{n + 1}] c;',
        '',
        'x q[0];',
    ]
    for i in range(1, n + 1):
        lines.append(f'c[0] = measure q[0];')
        lines.append(f'if (c[0]) x q[{i}];')
    for i in range(n + 1):
        lines.append(f'c[{i}] = measure q[{i}];')
    return '\n'.join(lines)


def parse_counts(raw_output, n_qubits):
    """Parse job output into counts dict."""
    if raw_output is None:
        return None
    if isinstance(raw_output, str):
        raw_output = json.loads(raw_output)
    counts = {}
    if isinstance(raw_output, dict):
        for key, val in raw_output.items():
            bs = str(key)
            if bs.startswith('0x'):
                bs = format(int(bs, 16), f'0{n_qubits}b')
            elif bs.startswith('0b'):
                bs = bs[2:].zfill(n_qubits)
            counts[bs] = val
    return counts


def main():
    # Setup
    creds = ClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    auth = ClientCredentialsAuth(creds=creds)
    config = McpConfig.from_env()
    clients = SdkClients(
        management=ManagementClient(auth=auth),
        scheduler=SchedulerClient(auth=auth),
        config=config,
    )
    service = OpenQuantumService(clients=clients)

    # Resolve org
    org = service.resolve_organization()
    print(f"Organization: {org['name']} ({org['id']})")

    # Check credits
    credits_info = service.get_credits()
    print(f"Credits: spark={credits_info['spark_credits']}, full={credits_info['full_credits']}")

    print("\n" + "=" * 60)
    print(f"{BACKEND_NAME}: Dynamic vs Pure Unitary")
    print(f"Depths: {DEPTHS}, Shots: {SHOTS}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    results = {}

    for n in DEPTHS:
        print(f"\n{'='*40} N={n} {'='*40}")

        # --- Pure Unitary ---
        qasm_u = make_qasm_pure_unitary(n)
        print(f"\n  [Unitary] N={n}, {n} CNOT gates")
        try:
            prep_u = service.prepare_job(
                backend_class_id=BACKEND_ID,
                qasm=qasm_u,
                shots=SHOTS,
                name=f"unitary_N{n}",
            )
            print(f"    Prepared: cost={prep_u.get('estimated_credits', '?')} credits")
            job_u = service.submit_job(
                preparation_id=prep_u['preparation_id'],
                confirm_spend=True,
            )
            print(f"    Submitted: job_id={job_u.get('job_id', '?')}")

            # Wait
            final_u = service.wait_for_job(job_u['job_id'])
            output_u = service.get_job_output(job_u['job_id'])
            counts_u = parse_counts(output_u, n + 1)
            print(f"    Done: {dict(list(counts_u.items())[:5]) if counts_u else 'No output'}")
        except Exception as e:
            print(f"    ERROR: {e}")
            counts_u = None

        # --- Dynamic ---
        qasm_d = make_qasm_dynamic(n)
        print(f"\n  [Dynamic] N={n}, {n} measure-feedback cycles")
        try:
            prep_d = service.prepare_job(
                backend_class_id=BACKEND_ID,
                qasm=qasm_d,
                shots=SHOTS,
                name=f"dynamic_N{n}",
            )
            print(f"    Prepared: cost={prep_d.get('estimated_credits', '?')} credits")
            job_d = service.submit_job(
                preparation_id=prep_d['preparation_id'],
                confirm_spend=True,
            )
            print(f"    Submitted: job_id={job_d.get('job_id', '?')}")

            final_d = service.wait_for_job(job_d['job_id'])
            output_d = service.get_job_output(job_d['job_id'])
            counts_d = parse_counts(output_d, n + 1)
            print(f"    Done: {dict(list(counts_d.items())[:5]) if counts_d else 'No output'}")
        except Exception as e:
            print(f"    ERROR: {e}")
            counts_d = None

        # Ideal outcome
        ideal = '1' * (n + 1)
        p_u = counts_u.get(ideal, 0) / SHOTS if counts_u else None
        p_d = counts_d.get(ideal, 0) / SHOTS if counts_d else None
        ratio = p_u / p_d if (p_u and p_d and p_d > 0) else None

        print(f"\n  Results N={n}:")
        print(f"    Unitary P(|{ideal}>) = {p_u}")
        print(f"    Dynamic P(|{ideal}>) = {p_d}")
        print(f"    Ratio = {ratio}")

        results[n] = {
            "n": n,
            "qasm_unitary": qasm_u,
            "qasm_dynamic": qasm_d,
            "counts_unitary": counts_u,
            "counts_dynamic": counts_d,
            "p_unitary": p_u,
            "p_dynamic": p_d,
            "ratio": ratio,
        }

    # Save
    output = {
        "platform": BACKEND_NAME,
        "backend_id": BACKEND_ID,
        "shots": SHOTS,
        "timestamp": datetime.now().isoformat(),
        "results": {str(k): v for k, v in results.items()},
    }
    out_path = f"experiments/ionq_dynamic_vs_unitary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'N':>3} | {'Unitary':>10} | {'Dynamic':>10} | {'Ratio':>8}")
    print("-" * 40)
    for n in DEPTHS:
        r = results[n]
        u = f"{r['p_unitary']:.4f}" if r['p_unitary'] is not None else "N/A"
        d = f"{r['p_dynamic']:.4f}" if r['p_dynamic'] is not None else "N/A"
        ratio = f"{r['ratio']:.2f}x" if r['ratio'] is not None else "N/A"
        print(f"{n:>3} | {u:>10} | {d:>10} | {ratio:>8}")


if __name__ == "__main__":
    main()
