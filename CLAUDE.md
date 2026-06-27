# QQ Chat Bot

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming -> invoke /office-hours
- Strategy/scope -> invoke /plan-ceo-review
- Architecture -> invoke /plan-eng-review
- Design system/plan review -> invoke /design-consultation or /plan-design-review
- Full review pipeline -> invoke /autoplan
- Bugs/errors -> invoke /investigate
- QA/testing site behavior -> invoke /qa or /qa-only
- Code review/diff check -> invoke /review
- Visual polish -> invoke /design-review
- Ship/deploy/PR -> invoke /ship or /land-and-deploy
- Save progress -> invoke /context-save
- Resume context -> invoke /context-restore

## File reading discipline

- Never manually construct a Read `offset` value. Omit offset entirely, or copy the number directly from a Grep output line number.
- Read from 0 by default (omit offset). Use `limit` to control how many lines to fetch.
- For long files, first use Grep to find the target region, then Read with the exact line number from Grep as offset.
- If Read fails because of an invalid offset, use the reported file length to correct it once; do not retry nearby guessed offsets.
- Prefer Grep for symbol/content lookup and Read only the small surrounding region needed.
- For broad codebase exploration requiring multiple searches, use the Explore agent instead of repeated manual reads.
