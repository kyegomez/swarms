# Signed decision receipts for Swarms — ScopeBlind integration

Add **Ed25519-signed, tamper-evident, offline-verifiable receipts** to every tool call a Swarms agent makes. Matches the [Veritas Acta receipt format](https://datatracker.ietf.org/doc/draft-farley-acta-signed-receipts/) used by Microsoft Agent Governance Toolkit, protect-mcp, sb-runtime, hermes-decision-receipts (aeoess), Signet (Prismer-AI), and the broader agent-governance ecosystem.

No fork of Swarms is required. No ScopeBlind infrastructure is in the verification path. Receipts verify with `npx @veritasacta/verify` against a public key the operator publishes out-of-band.

## What this gives you

- **Cryptographic evidence** of which tool your agent called, with what arguments hash, at what time, under what policy, with what result hash.
- **Chain linkage** across successive tool calls — tampering with any intermediate receipt breaks the chain at verification time.
- **Offline verification** with no dependency on Swarms, ScopeBlind, or any runtime service. Auditors get the receipts and a public key; they verify independently.
- **Compliance-ready audit trails** aligned with SOC 2 / ISO 42001 / EU AI Act requirements for agent-action logging.
- **Zero lock-in.** Apache-2.0 verifier, IETF-draft-spec'd receipt format, MIT adapter.

## Install

```bash
pip install scopeblind-swarms swarms
```

## Quick start

```python
from swarms import Agent
from scopeblind_swarms import ReceiptChain, sign_tool

# 1. Create a receipt chain (one per agent or per session)
chain = ReceiptChain.from_key_file(
    signer_key_path="/etc/scopeblind/issuer.key",
    agent_id="did:swarms:researcher",
    policy_id="allow-web-read",
)

# 2. Define tools as usual
def web_search(query: str) -> str:
    # your search logic
    return f"search results for {query}"

# 3. Wrap the tool to emit signed receipts
signed_search = sign_tool(web_search, chain=chain)

# 4. Use the wrapped tool in your Agent
agent = Agent(
    agent_name="researcher",
    tools=[signed_search],
    model_name="gpt-4",
    # ... rest of your Agent config
)
agent.run("find recent papers on agent governance")

# Every tool invocation now has a signed receipt:
print(signed_search.last_receipt)
print(f"Chain tip hash: {chain.current_tip}")
```

### Bulk wrapping

```python
from scopeblind_swarms import sign_tools

agent = Agent(
    agent_name="researcher",
    tools=sign_tools([web_search, summarize, post_draft], chain=chain),
    ...,
)
```

### Decorator form

```python
from scopeblind_swarms import ReceiptChain, sign_tool

chain = ReceiptChain.from_key_file("/etc/scopeblind/issuer.key", agent_id="did:swarms:writer")

@sign_tool(chain=chain, policy_id="allow-draft-only")
def post_draft(content: str) -> str:
    return f"drafted: {content[:50]}..."
```

## What's in a receipt

```json
{
  "payload": {
    "type": "scopeblind:swarms:tool-call",
    "agent_id": "did:swarms:researcher",
    "issuer_id": "swarms:agent:HJY4k2aN",
    "tool_name": "web_search",
    "action": "swarms:tool:web_search",
    "action_ref": "sha256:a8f3...c91e",
    "decision": "allow",
    "policy_id": "allow-web-read",
    "result_hash": "sha256:4b2c...d71f",
    "issued_at": "2026-04-19T15:42:01.773Z",
    "previousReceiptHash": "Zk4p..."
  },
  "signature": {
    "alg": "EdDSA",
    "kid": "HJY4k2aNqRwXcdEfGh...",
    "sig": "..."
  }
}
```

The `action_ref` is a SHA-256 of the JCS-canonicalized tool arguments — use it to correlate the same tool invocation across different audit systems. The `result_hash` is a SHA-256 of the tool's return value so the receipt attests to what came back without storing the raw output (privacy default).

## Offline verification

Any receipt verifies with the canonical verifier, zero dependencies on Swarms:

```bash
npx @veritasacta/verify receipt.json --key operator-public.pem
```

Exit code `0` = valid; `1` = proven tampering; `2` = undecidable.

Your operator public key is published via one of:

- An operator-signed JWKS URL (verifier: `--jwks https://you.example/jwks`)
- A DID document service endpoint (use any W3C DID resolver)
- A pinned trust anchor file
- Your GitHub release SBOM

Never embedded in the receipt itself (per draft-farley-acta-signed-receipts-02 §9).

## Composition with other Swarms-ecosystem components

`scopeblind-swarms` covers the **decision-receipt layer**. For full defense-in-depth, compose with:

- **Kernel sandboxing** — `sb-runtime` (Landlock+seccomp, Apache-2.0) or `nono` (capability-based). Your tool runs inside the sandbox; the sandbox backend is recorded in the receipt.
- **Policy evaluation** — `bindu-scopeblind` ships a Cedar evaluator; or call `cedarpy` directly before passing the decision into `sign_tool`.
- **Transparency-log anchoring** — submit receipts to Sigstore Rekor for timestamped immutability (DSSE wrapper; see `@veritasacta/verify --anchor rekor`).

## Thread safety

Swarms' async agents often run tools concurrently. `ReceiptChain` is safe to share across parallel tool invocations within a single agent; chain integrity is preserved via an internal lock. Successive receipts chain correctly even under concurrent signing.

## Related

| Resource | URL |
|---|---|
| `scopeblind-swarms` source | [github.com/ScopeBlind/scopeblind-gateway](https://github.com/ScopeBlind/scopeblind-gateway) (under `packages/verify-cli/ecosystem/adapters/swarms/`) |
| Reference verifier | [`@veritasacta/verify`](https://github.com/ScopeBlind/verify) (Apache-2.0, offline) |
| Receipt format spec | [draft-farley-acta-signed-receipts-02](https://datatracker.ietf.org/doc/draft-farley-acta-signed-receipts/) |
| Conformance profile | [VeritasActa/agt-integration-profile](https://github.com/VeritasActa/agt-integration-profile) |
| Microsoft AGT integration | [docs/integrations/sb-runtime.md](https://github.com/microsoft/agent-governance-toolkit/blob/main/docs/integrations/sb-runtime.md) (PR [#1202](https://github.com/microsoft/agent-governance-toolkit/pull/1202) merged) |

## License

MIT.
