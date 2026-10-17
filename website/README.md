# imajev-rs website

The marketing and documentation site for [imajev-rs](https://github.com/apiplant/imajev-rs), served at
<https://imajev-rs.apiplant.com>. Solid 2 RC + Tailwind v4 + Vite, a static build deployed as a Cloudflare
Workers assets-only site.

```bash
pnpm install
pnpm dev       # http://127.0.0.1:5276
pnpm build     # → dist/
pnpm check     # types only
```

There is no WebAssembly build: the model is a 4B-parameter vision-language model, far too large to ship to a browser.
Instead, the **Try it** page is a request builder that talks to an `imajev serve` you run yourself
(`imajev serve --cors '*'`), straight from the browser.

## The version is not copied here

`vite.config.ts` reads the version from `../Cargo.toml` (`[package] version`) at build time and injects it as
`__VERSION__`, so the install section and release links never go stale.

## Documentation pages

`src/components/docs/` holds one page per topic (`Overview`, `Library`, `Cli`, `Server`), a shared sidebar
(`DocsLayout.tsx`) and the prose primitives (`Prose.tsx`). There is no MDX pipeline: each page is hand-written TSX kept
in sync with `../README.md` by hand.

## Deploying

```bash
pnpm build && npx wrangler deploy
```

`wrangler.jsonc` binds the `imajev-rs.apiplant.com` custom domain and sends unknown paths to `index.html`
(`not_found_handling: single-page-application`) so deep links resolve on a cold load. `index.html`,
`public/robots.txt` and `public/sitemap.xml` name that domain; update all three if the site moves.
