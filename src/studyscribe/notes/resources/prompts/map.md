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
- Write in {language}. Keep technical terms as the lecturer uses them.
- Every section is filled on its own: one section never replaces another. Aim for the number
  of items given for each section below; fewer only when the lecturer really says less.
{code_rule}- Recognition errors: keep the transcript's words. Never replace a word with one you think
  the lecturer meant (writing spoken code in code form is not a replacement). When a passage
  looks misrecognised, {unclear}.
- Definitions keep the lecturer's wording; remove only spoken fillers.

Sections ("key": what to write, how much):
{sections}

Answer with JSON that follows this schema:
{schema}
