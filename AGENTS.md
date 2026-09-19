# Agent Guide — `webtyp/weights`

Constraints for agents working on this library. **Read this before any change.**
For end-user docs see [README.md](README.md); the current work order is [docs/PLAN.md](docs/PLAN.md).

---

## What this library is

`weights` carries model parameters from a URL to typed Go slices, **once per browser**. It owns the
artifact file format, its reader, the fetch and the IndexedDB cache. It knows nothing about what the
numbers mean — no tokenization, no inference, no embeddings.

Its **primary runtime is a browser tab compiled with TinyGo**. The host (`go test`) is a convenience
for development, not the target. Any change that is green on the host and red under TinyGo is **not
done**, no matter what the host badge says.

---

## The three builds that define "done"

```bash
go vet ./...
gotest                                  # host: vet + race + cover + wasm
gotest -tinygo                          # MANDATORY — the real target
GOOS=js GOARCH=wasm go build ./...
tinygo build -target wasm -o /dev/null .
```

`gotest` alone passing means nothing here. TinyGo's stdlib is a **subset**: packages that compile
under `GOOS=js GOARCH=wasm` (which uses the full Go stdlib) can still fail under TinyGo. `net/http`
is the canonical example — it compiles for `js/wasm` and **fails to compile under TinyGo**.

---

## Never import these — reach for the ecosystem instead

| Never | Use instead | Why |
|---|---|---|
| `net/http` | `webtyp.com/fetch` | does not compile under TinyGo; `fetch` is isomorphic (XHR/`fetch()` in the browser, `net/http` natively) |
| `context` (stdlib) | `webtyp.com/context` | the ecosystem's `*context.Context` is what every webtyp API takes |
| `encoding/json` | `webtyp.com/json`, or a fixed binary layout | reflection-based `encoding/json` costs **~1 MB of wasm** on its own (measured: 114 KB → 1.11 MB for a hello-world) |
| `fmt`, `errors`, `strconv`, `strings` | `webtyp.com/fmt` | one small package replaces all four; stdlib `fmt` drags in reflection |
| `time` | `webtyp.com/time` | |
| `log` | return an error, or inject a logger | a library never writes to a global stream |
| `os`, `flag`, `io/ioutil` | nothing — they don't belong in this module | see "Host-only tools" below |
| `database/sql` and any driver | `webtyp.com/storage` | `storage.Conn` is THE storage port of the ecosystem |
| `map[K]V` | `fmt.KeyValue` or a slice scanned linearly | TinyGo's map runtime is a size tax paid by every binary that imports this, transitively |

`unsafe` is allowed for the documented zero-copy tensor views, and nowhere else.

---

## Do not invent a port that already exists

This is the rule that was broken in PR #1 and the most expensive mistake available in this repo.

- **Caching goes through `storage.Conn`**, injected by the caller. Do not declare a local
  `StorageConn`/`Cache`/`KV` interface. `webtyp/indexdb` already implements `storage.Conn` over
  IndexedDB and already handles `[]byte` payloads (`FieldBlob`). A second, parallel port means the
  application cannot hand `weights` the connection it already has, and a second IndexedDB connection
  to the same database is a source of version-change deadlocks.
- **Fetching goes through `webtyp.com/fetch`.** Do not declare a local `Fetcher` interface to "make
  it testable": `fetch` is already the seam, and it already works in both worlds.
- Before adding *any* interface, search the ecosystem for it. `storage`, `fetch`, `json`, `binary`,
  `crypto`, `context`, `fmt`, `time`, `model` cover most of what a library here needs.

An interface is justified only when the thing it abstracts **varies by consumer**. `storage.Conn`
varies (indexdb / sqlt / postgres / mem). An HTTP client does not vary; it is already abstracted.

---

## Host-only tools live in their own module

A converter that reads a `.safetensors` from disk is legitimate work, but it **must not sit inside
this module**. `./...` includes `cmd/`, so one `os`/`flag`/`log` import there is enough to break
`gotest -tinygo` and `GOOS=js GOARCH=wasm go build ./...` for the whole repo, and to make a browser
library look like a CLI.

The ecosystem already has the pattern: build-time tools are separate repos with a `c` suffix —
`webtyp/ormc` for `orm`, `webtyp/ddlc` for `ddl`, `webtyp/sitec` for `site`. The weights converter
belongs in **`webtyp/weightsc`**, which may import `weights` (for the writer) plus anything host-only
it needs. `weights` itself stays importable from a browser binary.

The artifact **writer** (`writer.go`) stays here: it has no host dependency and the round-trip tests
need it.

---

## Format invariants (do not weaken)

- **Verification is not optional.** `Open` must reject a truncated or corrupt artifact
  unconditionally. A `checksum == 0` or `total_len == 0` that *skips* the check turns the one
  guarantee this package offers into a coin flip — a half-cached 100 MB artifact that loads without
  error and produces garbage vectors is the worst outcome available here.
- **The checksummed range is declared by the header**, never inferred from the first tensor's
  offset. Deriving it from data the header controls means a corrupt header can shrink the region it
  is meant to protect.
- **64-byte alignment per tensor**, little-endian everywhere — the same endianness decision as
  `webtyp/vector`'s codec. One endianness decision in the whole system.
- **Never cache before verifying.** Read the full body, check length and checksum, *then* write.
- **Never write past the quota.** Consult the browser's storage estimate first; when the estimate is
  unavailable, do **not** fall back to writing anyway — an eviction triggered here destroys the
  user's document corpus, which is far more valuable than a re-downloadable artifact.

---

## Layout & tests

- Flat hierarchy: Go files in the repo root. No subdirectories for library code.
- Max 500 lines per file; split by domain and rename when exceeded.
- More than 5 test files in the root → move **all** of them to `tests/`, package `tests`, consuming
  only the public API.
- Tests use the standard library only. No external assertion packages.
- The converter is tested against a **synthetic artifact built in the test**, never a downloaded
  model: no test in this repo may touch the network.

Publish with `gopush 'message'` — never `git commit`/`git push` directly.

---

## Common mistakes to avoid

- Declaring a local interface for something `storage`/`fetch` already abstracts.
- Believing `GOOS=js GOARCH=wasm go build ./...` proves TinyGo compatibility. It does not.
- Putting a `cmd/` with `os`/`flag`/`log` in this module.
- Making a correctness check conditional on a field being non-zero.
- Reaching for `map[K]V` "just for a lookup" — a linear scan over ~100 tensors costs nothing next to
  the I/O that produced them.
