# Configuration

How LLM-GC is told what to do: where its settings come from, which one wins when
several disagree, and what happens when one is wrong.

LLM-GC has two faces, and they are configured differently on purpose:

- **The library** (`llm_gc`) takes a `GCConfig` in code. It never reads a file or the
  environment, so importing it can never be changed by something outside the program.
- **The application** (the `llm-gc` command, the proxy and the dashboard) reads settings
  from five places, validates them as one object, and refuses to start on a bad one.

This document is about the application. The library's own settings are the fields of
`GCConfig`, described where each one acts.

## How it works

### Five layers, the later one winning

```
  built-in      config        .env         LLM_GC_*           flags
  defaults  <   file     <    file    <    environment   <    on the command
                                           variables          line
  ─────────     ──────        ─────        ───────────        ───────────────
  what works    what you      local        CI, containers,    a quick
  out of the    keep and      secrets      one-off            experiment
  box           share                      overrides
```

Each layer supplies only the settings it mentions. A setting no layer mentions takes its
default. The layers are combined key by key, including inside a section, so a flag that
sets one engine tunable leaves the others the file chose.

A flag the user did not type contributes nothing. This is what keeps a flag's own
default from overriding the file: the command-line library gives every option a
default, and treating that default as "the user said so" would make the file
unreachable.

### One object, validated before anything starts

The combined result is validated once, as a single settings object, before a port is
bound or a model is called. Every problem is reported together, each by the dotted name
of the setting, so a configuration with three mistakes is fixed in one restart, not
three. The checks cover types and ranges, an upstream that is not an http(s) URL, an
upstream that points back at the proxy itself, ports that collide, and the engine's own
rules (the thresholds must leave room for a span to move between levels).

An unknown setting is an error, naming it. A misspelt key in the file never looks
configured while doing nothing. The error also says which section the key belongs in, or
which known key it is closest to, because the usual cause is a real setting in the wrong
place: a line added at the end of the file lands in its last section, `[engine]`.

### Finding the config file

```
   --config PATH ──┐
                   ├─ named explicitly: it must exist, or startup stops
   LLM_GC_CONFIG ──┘
        │ neither given
        ▼
   ./llm-gc.toml        the project, in the directory the command runs from
        │ absent
        ▼
   ~/.config/llm-gc/config.toml      the user (XDG_CONFIG_HOME moves it)
        │ absent
        ▼
   no file: built-in defaults
```

The first file found is the only one read. Files are never merged, so which file applies
is always a question with one answer. The startup banner names the file it used, or says
there was none, because an auto-discovered file that silently changes behaviour is the
usability cost of searching at all.

### Credentials

LLM-GC's own credentials (the key or bearer token its summary and fact calls
authenticate with) come from the environment or `.env`, and from nowhere else. A
credential in the config file is refused, with a message saying where to put it. They are
never a flag and never part of what is printed, logged or reported in an error. Each
model vendor receives only its own credentials: one exported for another tool is
neither sent to it nor an obstacle to starting it.

### Choosing the model vendor

The settings name which vendor the summaries come from, the model, and optionally an
address. A single factory in the provider layer turns that into a client; the engine
never learns a vendor's name. See [llm-provider.md](llm-provider.md).

## Why it exists

An operator needs to run the same program in three situations without changing code: on
a laptop, where a file is natural; in CI or a container, where the environment is the
only channel; and in a quick experiment, where a flag is. The layers are what let one
binary serve all three.

Validation at load is the other half. The failure it prevents is the quiet one: a
wrong upstream, a refused model or an out-of-range threshold that does not stop the
proxy but makes every collection fail or never run, with the reason visible only on a
dashboard. Stopping at startup, with the setting named, moves that failure to the moment
it is cheapest to fix.

## Why this shape

**Defaults live in code, and the file holds only overrides.** The alternative is a
defaults file the program reads. It loses on four counts. The program must work when the
file is absent or partial, so code defaults are needed anyway and the file becomes a
second copy that can disagree. A user's copy of a defaults file freezes the defaults of
the release they copied it from, so improvements never reach them. The library has no file
and still needs working defaults. And a default in code is checked by the type checker,
where one in a file is checked only when read. The single visible list of settings comes
instead from a commented template, which shows every setting at its default, and a
command that prints the effective values.

**The template is a separate file, tested against the code.** It ships inside the
package, so it exists wherever LLM-GC is installed, and `config init` writes it out. A
test loads it and requires the result to equal the built-in defaults, so the template
cannot drift. The
alternative, generating it from the settings' field descriptions, makes drift impossible
by construction but gives up the grouping and prose that help a first-time reader. It is
worth revisiting if the settings grow past about twenty-five.

**TOML, not YAML or JSON.** JSON has no comments, and a hand-edited file needs somewhere
to say why a value is what it is. YAML's implicit typing turns an unquoted `no` into a
boolean and `3.10` into `3.1`, needs a third-party parser, and puts errors far from their
cause. TOML has comments and explicit types, is readable at a glance, and is read by the
standard library. The cost is that the standard library only reads TOML, which is all
that is needed.

**The first file wins; files are not merged.** Merging a user file under a project file
would make a setting's effective value depend on two files at once. The cost is that a
project file must restate any user-level setting it wants to keep. Setting the value as
an environment variable holds it whichever file wins.

**Credentials are not in the file or a flag.** The file is meant to be shared and
committed; a flag lands in shell history. The environment and `.env` (which is
git-ignored) are the two places a secret is conventionally kept without either risk.

**`.env` ignores variables it does not recognise; the config file does not.** `.env` is
shared with whatever else the user keeps in it, so an unrelated variable must not stop
LLM-GC starting. The config file belongs to LLM-GC alone, so an unknown key there is a
mistake worth reporting. The cost is that a misspelt LLM-GC variable in `.env` is ignored.

**An empty value is not set.** `.env.example` ships its credential lines blank, so a copy
of it holds `ANTHROPIC_API_KEY=` and the like. Read as values, those would be empty
credentials that `config show` reports as set while `start` refuses for having none, and
an empty `LLM_GC_UPSTREAM=` would replace the file's upstream with nothing. Ignoring empty
values at the source settles both at once, so every command asks one question: does any
layer give this a value? The cost is that a setting cannot be set to the empty string from
`.env` or the environment; none has a meaningful empty value.

**The engine's section is `GCConfig` with the proxy's window.** A layer that sets only
some engine keys builds the section from just those keys, so a default supplied any other
way would be replaced by the library's whenever any engine key was set. A subclass
changes the window's default and inherits everything else, so each setting is
defaulted independently.

**The proxy's default window is 200,000.** It is the smallest of the current models'.
Guessing too small costs a little compression. Guessing too large lets a recap, whose
budget is a share of the window, outgrow the model's real window, and requests fail. A
larger model is an explicit opt-in. The window is not read from the model being called,
because the model arrives with each request and changes within a session, and a gateway
may name models its own way.

**No credential means no start, unless the compactor is `noop`.** The alternative is to
run on canned summaries, which look like compaction working while nothing real is
summarising: hidden degradation, the thing this project exists to prevent. `noop` is the
honest way to run without a model, and also the baseline to measure against.

**Few settings are exposed.** Every setting is a promise to validate, document, test and
support. One is exposed when a real user has a reason to turn it and the trade-off can be
stated. Timeouts and batch sizes of the model calls stay in code until a slow gateway
makes one of them such a reason.

## What this costs

- Two places to look for a file, mitigated by the banner naming the one used.
- A project file restates user-level settings it wants to keep.
- A misspelt LLM-GC variable in `.env` is ignored, not reported.
- The template and the code are two things kept equal by a test, not one.

### The commands around the file

`config init` writes the starter file into the current directory, or the per-user location
with `--user`. It creates the file or fails in a single step, so it can never replace one
that appeared between a check and the write; replacing is a separate, explicit flag.
`config path` names the file the proxy would use and why, or says there is none.

`config show` prints every effective setting, with the layer it came from beside it. The
layers are merged before validation, so by then the origin of a value is gone; the origin
is found by asking each source what it contains, highest first, and taking the first that
mentions the setting. A setting no source mentions is a default. The output is valid
TOML, with the layer as a trailing comment, so it can be redirected into a file and loads
back as the same settings. A credential is reported as set or not, with the layer, and its
value is never printed. It runs the same validation as startup, so it also answers whether
a configuration is valid without starting anything. It takes no flags: a flag belongs to
one particular start, and the point is to see what the file, `.env` and environment give.
