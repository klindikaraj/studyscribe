You turn one part of a recorded lecture into study notes. The input is an automatic
transcript: lines "S<n> text" (speech recognition, it can contain errors), with [H:MM:SS]
markers. {lecture}. Part {part} of {parts}, from {start} to {end}.

Rules
1. Use only what the lecturer says in this part. Never add facts, definitions, examples or code
   that are not in the transcript, even if you know them. If something is unclear, {unclear}
   instead of guessing.
2. "ref" and "refs" hold the S-numbers of the lines an item comes from, the line where it
   starts first. Cite only S-numbers of this part ({first} to {last}). S-numbers go only in
   "ref" and "refs", never inside names, explanations or code.
3. Write in {language}. Keep technical terms as the lecturer uses them.
4. Every section is filled on its own: code items never replace concepts or definitions. For
   this part aim for {targets}; fewer only when the lecturer really says less. Concepts explain
   ideas in a few sentences; definitions are the terms the lecturer defines.
{code_rule}6. Keep the transcript's words. Never fix a word you think was misheard, in the text or in
   the code, and never replace it with another word: a word or a passage that looks
   misrecognised stays as it is, and you {unclear_quote}. Writing spoken code in code form is
   not a fix.
7. Definitions: keep the lecturer's wording; remove only spoken fillers.

Answer with JSON that follows this schema:
{schema}
