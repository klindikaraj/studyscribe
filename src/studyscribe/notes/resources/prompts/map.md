You turn one part of a recorded lecture into study notes. The input is an automatic
transcript: lines "S<n> text" (speech recognition, it can contain errors), with [H:MM:SS]
markers. {lecture}. Part {part} of {parts}, from {start} to {end}.

Rules
- Use only what the lecturer says in this part. Never add facts, definitions, examples or code
  that are not in the transcript, even if you know them. If something is unclear, {unclear}
  instead of guessing.
- "ref" and "refs" hold the S-numbers of the lines an item comes from, the line where it
  starts first. Cite only S-numbers of this part ({first} to {last}). S-numbers go only in
  "ref" and "refs", never inside names, explanations or code.
- Cover the whole part, from {first} to {last}: {coverage}, the last minutes included.
- Write in {language}. Keep technical terms as the lecturer uses them.
- Every section is filled on its own: one section never replaces another. Aim for the number
  of items given for each section below; fewer only when the lecturer really says less.
- Keep the transcript's words. Never fix a word you think was misheard, in the text or in the
  code, and never replace it with another word: a word or a passage that looks misrecognised
  stays as it is, and you {unclear_quote}. Writing spoken code in code form is not a fix.
{code_rule}- Definitions keep the lecturer's wording; remove only spoken fillers.

Sections ("key": what to write, how much):
{sections}

Answer with JSON that follows this schema:
{schema}
