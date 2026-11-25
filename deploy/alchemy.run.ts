// A preview of the web app on a Cloudflare Worker, for a link to share before a branch is merged.
// The site itself is deployed by Vercel from the upstream repository (vercel.json); this is only for
// the branch `preview` of a fork, which .github/workflows/preview.yml deploys on each push.
//
// The Worker has no code of its own: it serves the build (`bun run build`, ../dist) as static
// assets, with the same cache rules and the same /plugin → /plugin/ redirect as vercel.json.
//
// The state of the stack is on the branch `alchemy-state-store` of the repository that deploys it
// (@samebase/alchemy-kit/github-state): no bucket and no other account. A run needs GITHUB_TOKEN
// (write access to the repository's contents), ALCHEMY_GITHUB_STATE_KEY (32 random bytes in
// base64, a repository secret), and CLOUDFLARE_API_TOKEN with CLOUDFLARE_ACCOUNT_ID (a token from
// the "Edit Cloudflare Workers" template). From a laptop, in this folder:
//   GITHUB_TOKEN="$(gh auth token)" ALCHEMY_GITHUB_STATE_KEY=... npm run deploy
import * as Cloudflare from "@alchemy.run/cloudflare";
import * as GitHubState from "@samebase/alchemy-kit/github-state";
import * as Alchemy from "alchemy";
import * as Effect from "effect/Effect";

/** The cache rules of vercel.json: the hashed bundles for a year, fonts and data for a day. */
const HEADERS = `/assets/*
  Cache-Control: public, max-age=31536000, immutable
/fonts/*
  Cache-Control: public, max-age=86400
/data/*
  Cache-Control: public, max-age=86400
`;

export default Alchemy.Stack(
  "chimera-vocal-synth-preview",
  {
    providers: Cloudflare.providers(),
    state: GitHubState.state(),
  },
  Effect.gen(function* () {
    const { stage } = yield* Alchemy.Stack;
    const site = yield* Cloudflare.Worker("Site", {
      name: stage === "prod" ? "chimera-vocal-synth-preview" : `chimera-vocal-synth-preview-${stage}`,
      assets: {
        directory: "../dist",
        // /plugin → /plugin/, as vercel.json redirects it
        htmlHandling: "auto-trailing-slash",
        notFoundHandling: "none",
        headers: HEADERS,
      },
      compatibility: { date: "2026-11-01" },
    });
    return { url: site.url };
  }),
);
