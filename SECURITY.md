# Security Policy

## Reporting a Vulnerability

If you find a security vulnerability in PriorPatch, please report it privately rather than opening a public issue.

**Contact**: Open a GitHub Security Advisory at <https://github.com/jayashankarvr/priorpatch/security/advisories/new>

Please include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)

## What qualifies as a security issue?

For a forensics tool, security issues include:

- **Bypass vulnerabilities**: Ways to make manipulated images appear authentic
- **Path traversal**: Ability to read/write files outside intended directories
- **Code injection**: Ability to execute arbitrary code through inputs
- **DoS vulnerabilities**: Inputs that crash the tool or consume excessive resources
- **False negatives**: Manipulation techniques that reliably evade detection (especially if exploitable at scale)

## Response timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix timeline**: Depends on severity
  - Critical: 1-7 days
  - High: 1-4 weeks
  - Medium: 1-3 months
  - Low: Best effort

## Disclosure policy

We follow coordinated disclosure:

1. You report the issue privately
2. We confirm and develop a fix
3. We release a patched version
4. We publish a security advisory
5. You get credit (if you want it)

**Please don't publish the vulnerability before we've had a chance to fix it.**

## Security best practices for users

When using PriorPatch:

1. **Don't trust it alone**: Always use multiple detection methods
2. **Validate inputs**: Ensure image files come from trusted sources
3. **Sandbox if needed**: Run in a container/VM if processing untrusted images at scale
4. **Update regularly**: Security and detection improvements happen in new versions
5. **Review results**: Don't automatically act on detections without human review

## Known limitations (not security bugs)

These are design limitations, not vulnerabilities:

- **Can be fooled**: Tool isn't designed to resist targeted attacks
- **PRNU detector is basic**: Current implementation is a stub, not production-quality
- **No anti-tampering**: Configuration files can be modified
- **No rate limiting**: CLI can be called repeatedly
- **No authentication**: Tool is meant to run locally

If you need these features, they're valid feature requests but not security issues.

## Security features we implement

- Path validation to prevent directory traversal
- Input validation for image dimensions
- Safe file I/O operations
- No eval() or exec() on user input
- Dependencies pinned to known versions

## Dependencies

We rely on:

- numpy
- Pillow
- matplotlib
- scipy

If any of these have security advisories, we'll update promptly.

Run `pip list --outdated` regularly to check for updates.

## Questions?

Not sure if something is a security issue? Report it anyway. Better safe than sorry.
