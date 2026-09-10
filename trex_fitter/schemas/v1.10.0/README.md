Schema snapshots from TRExStats/TRExFitter v1.10.0, commit
`52e62c30a1faf1ca9fdfe0db74160bb9e00e86e9`.

These preserve the contents of `jobSchema.config` and `multiFitSchema.config`
(apart from trailing blank lines).
Validation follows `Root/ConfigParser.cc`: slash-separated alternatives and
comma-separated parameter types. Numeric tokens must additionally be complete
and finite; Python validation does not accept the numeric-prefix parsing of
`std::stoi`/`std::stod`.

Schema validation checks allowed settings and declared types across blocks.
Additional semantic checks cover the basic analysis; ROOT expressions, input
branches, and advanced cross-block semantics still require native validation.
