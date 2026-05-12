# Swarms Documentation Bounty Program

The Swarms documentation bounty program rewards contributors who improve the quality, completeness, and usability of the Swarms docs. The program is intended for practical documentation work that helps developers install Swarms, understand core concepts, use agents and tools correctly, and contribute to the project with less friction.

This page explains what usually qualifies, how to prepare a bounty-eligible pull request, and what maintainers look for during review.

## Eligible Contributions

Documentation contributions are most useful when they remove a real source of confusion for users or maintainers. Eligible work can include:

- New tutorials, examples, or deep dives for Swarms agents, tools, workflows, and integrations.
- Updates to outdated API references, installation steps, command examples, or screenshots.
- Fixes for broken navigation entries, broken links, missing pages, formatting issues, and unclear contributor guidance.
- Translations or localization improvements that preserve the technical meaning of the original docs.
- Troubleshooting pages that document common setup errors and reliable recovery steps.

Small typo fixes are welcome, but larger bounty awards normally require a change that improves a complete user workflow.

## Reward Tiers

Maintainers assess the final tier after reviewing the merged contribution. The expected ranges are:

| Tier | Typical contribution | Expected payout |
| --- | --- | --- |
| Bronze | Typos or minor enhancements under 100 words | $1-$5 |
| Silver | Small tutorials, API examples, or focused fixes of 100-500 words | $5-$20 |
| Gold | Major updates or guides over 500 words | $20-$50 |
| Platinum | Multi-part guides, new documentation verticals, or substantial cross-page improvements | $50-$300 |

These ranges are guidelines, not automatic guarantees. A concise fix to an important broken page can be more valuable than a long guide that is hard to verify.

## How to Claim Eligibility

When opening a pull request, include a short bounty section in the PR description:

```markdown
## Bounty eligibility
- Label requested: `bounty-eligible`
- Expected tier: Silver
- Rationale: this fixes a broken docs navigation entry and adds a missing contributor page.
```

If the label is not available to external contributors, request it in the PR body and leave the final label decision to maintainers. Do not include public payout details in the pull request. If a reward is approved after merge, share PayPal, crypto, or wire details privately through the channel requested by the maintainers.

## Review Expectations

Good bounty PRs are focused, verifiable, and easy to merge. Before opening a PR:

1. Keep the scope narrow enough that maintainers can review it quickly.
2. Update navigation if you add a page.
3. Check Markdown formatting and code fences.
4. Confirm links and commands are still current.
5. Explain any local build limitations in the PR body.

For code examples, prefer small runnable snippets over long illustrative blocks. Include required environment variables, dependency names, and expected outputs when they matter.

## Quality Bar

Maintainers may decline or lower a bounty if the work is difficult to verify, duplicates existing content, introduces broken links, includes inaccurate technical claims, or appears to be generated without testing. A good contribution should make the documentation easier to trust.

If you are unsure whether a larger docs idea fits the program, open a short issue or discussion first. For small fixes and missing pages, a focused pull request with clear verification notes is usually enough.

## Examples

Examples of strong bounty candidates:

- A new guide that demonstrates a complete Swarms workflow from setup to output.
- A troubleshooting page for a common installation or provider configuration problem.
- A missing contributor page that is already referenced by the MkDocs navigation.
- A set of broken-link fixes with a short explanation of how they were verified.

Examples of weak candidates:

- Unverified code copied from another source.
- Broad rewrites that change tone without improving accuracy.
- Large PRs that mix unrelated docs, formatting, and code changes.
- Public comments asking for payment before the work has been reviewed or merged.
