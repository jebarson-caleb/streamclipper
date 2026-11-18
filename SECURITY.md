# Security policy

## Reporting a vulnerability

Please **don't open a public issue** for security problems. Report them privately through GitHub's
[security advisory form](https://github.com/jebarson-caleb/streamclipper/security/advisories/new), with steps to
reproduce and the version you tested. You'll get a reply as soon as possible; there is no bug bounty.

Only the latest release is supported with fixes.

## How StreamClipper protects you

- The dashboard listens on `127.0.0.1` only and has no login, so it refuses requests addressed to any other host
  (DNS rebinding) and form posts coming from other websites (CSRF).
- Links you paste are validated, and every call to yt-dlp, streamlink and ffmpeg passes them after `--` / `-i`,
  so a link can never be read as a command-line option.
- The bundled AI engine listens on `127.0.0.1` with a random key per launch.
- Notification tokens and webhook URLs are never shown on the Settings page or written to the log.
- Links are stripped from AI-written titles and descriptions, since the AI reads viewers' chat and speech.
- Installer builds pin and SHA-256-check every bundled tool, and the build fails if Windows Application Control
  would block any bundled file.

## Your part

- Keep `dashboard.host` at `127.0.0.1`. If you expose it to a network, anyone who can reach it controls the app.
- `client_secret.json` and the OAuth tokens in `data\tokens\` give access to your YouTube channel: don't share
  them or commit them (they're in `.gitignore`).
- Turn on `review_mode` if you want to approve every clip before it's published.
