# ICRA manuscript workflow

- After every writing or editing pass on this manuscript, compile `root.tex`
  and regenerate `root.pdf` before reporting completion.
- Run the compiler from `icra2027/` so section and figure paths resolve.
- Tectonic is available in this environment at
  `/tmp/opencode/porygon-tex/bin/tectonic`; build with:

  ```bash
  /tmp/opencode/porygon-tex/bin/tectonic root.tex
  ```

- If that temporary environment is unavailable, use an installed LaTeX compiler
  or restore Tectonic. Check the build output and inspect the resulting PDF.
- Preserve the supplied `ieeeconf.cls` and its page geometry.
