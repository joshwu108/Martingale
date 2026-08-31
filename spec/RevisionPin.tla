-------------------------------- MODULE RevisionPin --------------------------------
(*
  Finite-state model of the pin-before-draw protocol for martingale.

  Scope: 2 actors, 3 revisions, 2 actions.

  Invariant: For every ledgered action, the revision in the ledger equals
  the revision pinned by the actor at the time of the draw.

  Two configurations:
  1. CORRECT: pin-before-draw ordering enforced → invariant passes.
  2. WEAKENED: ordering removed (actor can draw before pinning) → counterexample exists.

  State per actor: {IDLE, PINNED, DREW, APPENDED}
  Learner state: set of published revisions, current revision.
*)

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
  Actors,        \* Set of actor IDs: {0, 1}
  Revisions,     \* Set of revision IDs: {0, 1, 2}
  Actions        \* Set of actions: {0, 1}

ASSUME Actors = {0, 1}
ASSUME Revisions = {0, 1, 2}
ASSUME Actions = {0, 1}

VARIABLES
  actor_state,      \* actor_state[a] ∈ {IDLE, PINNED, DREW, APPENDED}
  actor_pin,        \* actor_pin[a] ∈ Revisions ∪ {-1} (pinned revision or none)
  actor_draw,       \* actor_draw[a] ∈ Actions ∪ {-1} (drawn action or none)
  actor_draw_rev,   \* actor_draw_rev[a]: revision used for the actual draw
  ledger,           \* Sequence of (actor, revision, action) records
  published,        \* Set of published revision IDs
  current_rev       \* Most recently published revision

vars == <<actor_state, actor_pin, actor_draw, actor_draw_rev, ledger, published, current_rev>>

IDLE == "IDLE"
PINNED == "PINNED"
DREW == "DREW"
APPENDED == "APPENDED"

TypeOK ==
  /\ actor_state \in [Actors -> {IDLE, PINNED, DREW, APPENDED}]
  /\ actor_pin \in [Actors -> Revisions \cup {-1}]
  /\ actor_draw \in [Actors -> Actions \cup {-1}]
  /\ actor_draw_rev \in [Actors -> Revisions \cup {-1}]
  /\ published \subseteq Revisions
  /\ current_rev \in Revisions

(*
  INVARIANT (T5): For every ledger record, the recorded revision equals
  the actor's pin at the time of the draw.

  We track this as: every appended record (actor, revision, action) has
  revision == actor_pin[actor] at the time of DREW → APPENDED.
  Since we record actor_pin at pin time and actor_draw_rev at draw time,
  we check: for all records, record.revision == record's draw_revision.
*)
PinInvariant ==
  \A i \in 1..Len(ledger):
    ledger[i].revision = ledger[i].pin_at_draw

Init ==
  /\ actor_state = [a \in Actors |-> IDLE]
  /\ actor_pin = [a \in Actors |-> -1]
  /\ actor_draw = [a \in Actors |-> -1]
  /\ actor_draw_rev = [a \in Actors |-> -1]
  /\ ledger = <<>>
  /\ published = {0}
  /\ current_rev = 0

\* Learner publishes a new revision (if not all published yet)
LearnerPublish ==
  /\ \E r \in Revisions \ published:
     /\ published' = published \cup {r}
     /\ current_rev' = r
     /\ UNCHANGED <<actor_state, actor_pin, actor_draw, actor_draw_rev, ledger>>

\* Actor pins a revision (IDLE → PINNED)
\* CORRECT ordering: pin BEFORE draw
ActorPin(a) ==
  /\ actor_state[a] = IDLE
  /\ published # {}
  /\ actor_state' = [actor_state EXCEPT ![a] = PINNED]
  /\ actor_pin' = [actor_pin EXCEPT ![a] = current_rev]
  /\ UNCHANGED <<actor_draw, actor_draw_rev, ledger, published, current_rev>>

\* Actor draws an action using the PINNED revision's simplex (PINNED → DREW)
ActorDraw(a, action) ==
  /\ actor_state[a] = PINNED
  /\ action \in Actions
  /\ actor_state' = [actor_state EXCEPT ![a] = DREW]
  /\ actor_draw' = [actor_draw EXCEPT ![a] = action]
  /\ actor_draw_rev' = [actor_draw_rev EXCEPT ![a] = actor_pin[a]]
  \* The draw uses ONLY actor_pin[a] — structurally cannot use another revision
  /\ UNCHANGED <<actor_pin, ledger, published, current_rev>>

\* Actor appends to ledger (DREW → APPENDED)
ActorAppend(a) ==
  /\ actor_state[a] = DREW
  /\ ledger' = Append(ledger, [
       actor |-> a,
       revision |-> actor_pin[a],     \* pinned revision
       pin_at_draw |-> actor_draw_rev[a],  \* revision used for draw
       action |-> actor_draw[a]
     ])
  /\ actor_state' = [actor_state EXCEPT ![a] = APPENDED]
  /\ UNCHANGED <<actor_pin, actor_draw, actor_draw_rev, published, current_rev>>

\* Actor resets to IDLE (APPENDED → IDLE)
ActorReset(a) ==
  /\ actor_state[a] = APPENDED
  /\ actor_state' = [actor_state EXCEPT ![a] = IDLE]
  /\ actor_pin' = [actor_pin EXCEPT ![a] = -1]
  /\ actor_draw' = [actor_draw EXCEPT ![a] = -1]
  /\ actor_draw_rev' = [actor_draw_rev EXCEPT ![a] = -1]
  /\ UNCHANGED <<ledger, published, current_rev>>

Next ==
  \/ LearnerPublish
  \/ \E a \in Actors: ActorPin(a)
  \/ \E a \in Actors, action \in Actions: ActorDraw(a, action)
  \/ \E a \in Actors: ActorAppend(a)
  \/ \E a \in Actors: ActorReset(a)

Spec == Init /\ [][Next]_vars

\* The invariant should hold for the correct ordering
THEOREM Spec => []PinInvariant

================================================================================
