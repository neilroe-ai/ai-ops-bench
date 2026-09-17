# Diagrams

`ai-ops-bench.architecture.json` is the source; `ai-ops-bench.html` is the rendered,
self-contained viewer. Open the HTML directly in a browser — no server, no build step.

Pinned to commit `b04a7afec74561d448b207d3d38a5421ace22b34`. The `sources` entries in the
JSON link to real files and line numbers at that revision, so they go stale when those
files move. Re-render after a structural change:

    node <archify>/bin/archify.mjs deliver architecture \
        docs/diagrams/ai-ops-bench.architecture.json \
        docs/diagrams/ai-ops-bench.html \
        --quality showcase --repo-root .

Edit the JSON, never the HTML. The viewBox is 1340x480: wider drops node text below the
6px readability floor, taller overflows a 1440x900 desktop viewport. Both are checked.
